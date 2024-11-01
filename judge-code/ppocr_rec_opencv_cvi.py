#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# -*- coding: utf-8 -*- 

import logging ; logging.basicConfig(level=logging.DEBUG)

import os
from time import time
from argparse import ArgumentParser
from typing import List

import cv2
import numpy as np
from numpy import ndarray
import onnxruntime as ort


class PPOCRv2Rec:

    def __init__(self, args):
        # load cvimodel
        model_path = args.cvimodel_rec
        logging.info("using model {}".format(model_path))
        # self.net = sail.Engine(model_path, args.dev_id, sail.IOMode.SYSIO)
        self.net = ort.InferenceSession(model_path)
        node_input = self.net.get_inputs()[0]
        node_output = self.net.get_outputs()[0]
        print(f'>> [input] name: {node_input.name}, shape: {node_input.shape}')
        print(f'>> [output] name: {node_output.name}, shape: {node_output.shape}')
        self.graph_name = 'ch_PP-OCRv3_rec'
        self.input_name = 'x'
        if 'v3' in model_path or 'v4' in model_path:
            self.input_shape = [1, 3, 48, 640]
        else:
            self.input_shape = [1, 3, 32, 320]
        self.rec_batch_size = 1
        logging.info("load cvimodel success!")
        # hparam
        self.img_size  = sorted(args.img_size)
        self.img_ratio = sorted([x[0]/x[1] for x in self.img_size])
        self.character = ['blank']
        with open(args.char_dict_path, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.strip('\r\n').strip('\n')
                self.character.append(line)
        if args.use_space_char:
            self.character.append(' ')
        # perfcnt
        self.preprocess_time  = 0.0
        self.inference_time   = 0.0
        self.postprocess_time = 0.0

    def preprocess(self, im:ndarray):
        h, w, _ = im.shape
        ratio = w / float(h)
        if ratio > self.img_ratio[-1]:
            logging.debug("Warning: ratio out of range: h = %d, w = %d, ratio = %f, cvimodel with larger width is recommended." % (h, w, ratio))
            resized_w = self.img_size[-1][0]
            resized_h = self.img_size[-1][1]
            padding_w = resized_w
        else:          
            for max_ratio in self.img_ratio:
                if ratio <= max_ratio:
                    resized_h = self.img_size[0][1]
                    resized_w = int(resized_h * ratio)
                    padding_w = int(resized_h * max_ratio)
                    break
        # 在 uint8 域上 resize (?)
        if h != resized_h or w != resized_w:
            im = cv2.resize(im, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        # to [C, H, W], f32
        im = im.astype(np.float32)
        im -= 127.5        # = 255/2
        im *= 0.0078125    # = 1/127
        im = im.transpose(2, 0, 1)

        # right pad 0
        im_pad = np.zeros((3, resized_h, padding_w), dtype=np.float32)
        im_pad[:, :, :resized_w] = im
        return im_pad

    def predict(self, x):
        start_infer = time()
        outputs =  self.net.run([self.net.get_outputs()[0].name], {self.input_name: x})[0]
        self.inference_time += time() - start_infer
        return outputs      # [B=1, L=40, NC=6625]

    def postprocess(self, outputs:ndarray):
        start_post = time()
        result_list = []

        preds_idx = outputs.argmax(axis=2)
        preds_prob = outputs.max(axis=2)
        for batch_idx, pred_idx in enumerate(preds_idx):
            char_list = []
            conf_list = []
            pre_c = pred_idx[0]
            #print("pre_c: ",pre_c)  
            if pre_c != 0:
                char_list.append(self.character[pre_c])
                conf_list.append(preds_prob[batch_idx][0])
            for idx, c in enumerate(pred_idx):
                if pre_c == c:
                    continue
                if c == 0:
                    pre_c = c
                    continue
                char_list.append(self.character[c])
                conf_list.append(preds_prob[batch_idx][idx])
                pre_c = c

            result_list.append((''.join(char_list), np.mean(conf_list)))

        self.postprocess_time += time() - start_post
        return result_list

    def __call__(self, img_list:List[ndarray]):
        img_dict = {}
        for img_size in self.img_size:
            img_dict[img_size[0]] = {"imgs":[], "ids":[], "res":[]}
        for id, img in enumerate(img_list):
            start_pre = time()
            img = self.preprocess(img)
            self.preprocess_time += time() - start_pre
            if img is None: continue
            img_dict[img.shape[2]]["imgs"].append(img)
            img_dict[img.shape[2]]["ids"].append(id)

        for size_w in img_dict.keys():
            if size_w > 640:
                for img_input in img_dict[size_w]["imgs"]:
                    img_input = np.expand_dims(img_input, axis=0)
                    outputs = self.predict(img_input)
                    res = self.postprocess(outputs)
                    img_dict[size_w]["res"].extend(res)
            else:
                img_num = len(img_dict[size_w]["imgs"])
                for beg_img_no in range(0, img_num, self.rec_batch_size):
                    end_img_no = min(img_num, beg_img_no + self.rec_batch_size)
                    if beg_img_no + self.rec_batch_size > img_num:
                        for ino in range(beg_img_no, end_img_no):
                            img_input = np.expand_dims(img_dict[size_w]["imgs"][ino], axis=0)
                            outputs = self.predict(img_input)
                            res = self.postprocess(outputs)
                            img_dict[size_w]["res"].extend(res)   
                    else:
                        img_input = np.stack(img_dict[size_w]["imgs"][beg_img_no:end_img_no])
                        outputs = self.predict(img_input)
                        res = self.postprocess(outputs)
                        img_dict[size_w]["res"].extend(res)

        rec_res = {"res":[], "ids":[]}
        for size_w in img_dict.keys():
            rec_res["res"].extend(img_dict[size_w]["res"])
            rec_res["ids"].extend(img_dict[size_w]["ids"])
        return rec_res


def main(opt):
    ppocrv2_rec = PPOCRv2Rec(opt)

    file_list = sorted(os.listdir(opt.input), key=(lambda e: int(e.split('.')[0].split('_')[-1])))
    img_list: List[ndarray] = []
    for img_name in file_list:
        img_file = os.path.join(opt.input, img_name)
        src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)   # [H, W, C=3], u8 RGB
        img_list.append(src_img)

    rec_res = ppocrv2_rec(img_list)
    for i, id in enumerate(rec_res.get("ids")):
        logging.info("img_name: {}, conf: {:.6f}, pred: {}".format(file_list[id], rec_res["res"][i][1], rec_res["res"][i][0]))


def img_size_type(arg):
    img_sizes = arg.strip('[]').split('],[')
    img_sizes = [size.split(',') for size in img_sizes]
    img_sizes = [[int(width), int(height)] for width, height in img_sizes]
    return img_sizes


if __name__ == '__main__':
    parser = ArgumentParser(prog=__file__)
    parser.add_argument('--dev_id', type=int, default=0, help='tpu card id')
    parser.add_argument('--input', type=str, default='../datasets/cali_set_rec', help='input image directory path')
    parser.add_argument('--cvimodel_rec', type=str, default='../models/ch_PP-OCRv3_rec_infer.onnx', help='recognizer cvimodel path')
    parser.add_argument('--img_size', type=img_size_type, default=[[640, 48],[320, 48]], help='You should set inference size [width,height] manually if using multi-stage cvimodel.')
    parser.add_argument("--char_dict_path", type=str, default="../datasets/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=bool, default=True)
    opt = parser.parse_args()

    main(opt)
