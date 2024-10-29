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
import math
from time import time
from argparse import ArgumentParser
from typing import List

import cv2
import numpy as np
from numpy import ndarray
import onnxruntime as ort


class PPOCRv2Cls:

    def __init__(self, args):
        # load cvimodel
        model_path = args.cvimodel_cls
        logging.info('using model {}'.format(model_path))
        # self.net = sail.Engine(model_path, args.dev_id, sail.IOMode.SYSIO)
        self.net = ort.InferenceSession(model_path)
        node_input = self.net.get_inputs()[0]
        node_output = self.net.get_outputs()[0]
        print(f'>> [input] name: {node_input.name}, shape: {node_input.shape}')
        print(f'>> [output] name: {node_output.name}, shape: {node_output.shape}')
        self.graph_name = 'ch_PP-OCRv3_cls'
        self.input_name = 'x'
        self.input_shape = [1, 3, 48, 640]
        self.cls_batch_size = 1
        logging.info('load cvimodel success!')
        # hparam
        self.cls_thresh = args.cls_thresh
        self.label_list = args.label_list
        # perfcnt
        self.preprocess_time  = 0.0
        self.inference_time   = 0.0
        self.postprocess_time = 0.0

    def preprocess(self, im:ndarray) -> ndarray:
        # 保宽高比，强制放缩高h到48，由此推断w
        # TODO: 先旋转竖条图像为水平
        h, w, _ = im.shape
        ratio = w / float(h)
        resized_h = self.input_shape[2]
        new_w = math.ceil(ratio * resized_h)
        if new_w > self.input_shape[3]:
            resized_w = self.input_shape[3]
        else:
            resized_w = new_w
        # 在 uint8 域上 resize (?)
        if h != resized_h or w != resized_w:
            im = cv2.resize(im, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        # to [C, H, W], f32
        im: ndarray
        im = im.astype(np.float32)
        im -= 127.5        # = 255/2
        im *= 0.0078125    # = 1/127
        im = im.transpose(2, 0, 1)

        # right pad 0
        im_pad = np.zeros((3, self.input_shape[2], self.input_shape[3]), dtype=np.float32)
        im_pad[:, :, :resized_w] = im
        return im_pad

    def predict(self, x:ndarray) -> ndarray:
        #outputs = model_inference({self.input_name: x}, self.net)
        # [B, C, H, W] => [B, NC=2], probs (with softmax embeded)
        return self.net.run([self.net.get_outputs()[0].name], {self.input_name: x})[0]

    def postprocess(self, outputs:ndarray):
        #print('outputs.shape:', outputs.shape)
        preds = outputs.argmax(axis=1)
        #print('preds.shape:', preds.shape)
        return [(self.label_list[idx], outputs[i, idx]) for i, idx in enumerate(preds)]

    def __call__(self, img_list:List[ndarray]):
        start_pre = time()
        img_input_list = [self.preprocess(img) for img in img_list]
        self.preprocess_time += time() - start_pre

        start_infer = time()
        img_num = len(img_list)
        cls_res = []
        for beg_img_no in range(0, img_num, self.cls_batch_size):
            end_img_no = min(img_num, beg_img_no + self.cls_batch_size)
            if beg_img_no + self.cls_batch_size > img_num:
                for ino in range(beg_img_no, end_img_no):
                    img_input = np.expand_dims(img_input_list[ino], axis=0)
                    outputs = self.predict(img_input)
                    res = self.postprocess(outputs)
                    cls_res.extend(res)
            else:
                img_input = np.stack(img_input_list[beg_img_no:end_img_no])
                outputs = self.predict(img_input)
                res = self.postprocess(outputs)
                cls_res.extend(res)
        self.inference_time += time() - start_infer

        start_post = time()
        for id, res in enumerate(cls_res):
            if res[0] == '180' and res[1] > self.cls_thresh:
                img_list[id] = cv2.rotate(img_list[id], cv2.ROTATE_180)
        self.postprocess_time += time() - start_post

        return cls_res, img_list


def main(opt):
    ppocrv2_cls = PPOCRv2Cls(opt)

    file_list = sorted(os.listdir(opt.input), key=(lambda e: int(e.split('.')[0].split('_')[-1])))
    img_list: List[ndarray] = []
    for img_name in file_list:
        img_file = os.path.join(opt.input, img_name)
        src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)   # [H, W, C=3], u8 RGB
        img_list.append(src_img)

    cls_res, img_list = ppocrv2_cls(img_list)
    for img_name, res in zip(file_list, cls_res):
        logging.info(f'img_name: {img_name}, pred: {res[0]}, conf: {res[1]}')


def list_type(arg):
    return arg.split(',')


if __name__ == '__main__':
    parser = ArgumentParser(prog=__file__)
    parser.add_argument('--dev_id', type=int, default=0, help='tpu card id')
    parser.add_argument('--input', type=str, default='../datasets/cali_set_rec', help='input image directory path')
    parser.add_argument('--cvimodel_cls', type=str, default='../models/ch_ppocr_mobile_v2.0_cls_infer.onnx', help='classifier cvimodel path')
    parser.add_argument('--cls_thresh', type=float, default=0.9)
    parser.add_argument('--label_list', type=list_type, default='0,180')
    opt = parser.parse_args()

    main(opt)
