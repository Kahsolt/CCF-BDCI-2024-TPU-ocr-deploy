#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# -*- coding: utf-8 -*-

import logging ; logging.basicConfig(level=logging.INFO)

import os
import math
import json
import random
from time import time
from copy import deepcopy
from argparse import ArgumentParser

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ppocr_det_opencv_cvi as predict_det
import ppocr_rec_opencv_cvi as predict_rec
import ppocr_cls_opencv_cvi as predict_cls

'''
[Test on i7-11700K@3.60GHz]
------------------ Image Load/Crop Time Info ----------------------
decode_time(ms): 6.35
crop_time(ms): 0.48
------------------ Det Predict Time Info ----------------------
preprocess_time(ms): 5.55
inference_time(ms): 46.84
postprocess_time(ms): 3.06
------------------ Rec Predict Time Info ----------------------
preprocess_time(ms): 0.10
inference_time(ms): 9.35
postprocess_time(ms): 0.24
------------------ Total Time Info ----------------------
total_inference_time(ms): 56.19
total_run_time(ms): 19935.76
per_img_run_time(ms): 155.75
'''

class TextSystem:

    def __init__(self, args):
        # sub models
        self.text_detector = predict_det.PPOCRv2Det(args)
        self.text_recognizer = predict_rec.PPOCRv2Rec(args)
        self.use_angle_cls = args.use_angle_cls
        if self.use_angle_cls:
            self.text_classifier = predict_cls.PPOCRv2Cls(args)
        # hparam
        self.rec_thresh = args.rec_thresh
        # perfcnt
        self.crop_num = 0
        self.crop_time = 0.0

    def __call__(self, img_list, cls=True):
        dt_boxes_list = self.text_detector(img_list)
        img_dict = {"imgs":[], "dt_boxes":[], "pic_ids":[]}
        for id, dt_boxes in enumerate(dt_boxes_list):
            img = img_list[id]
            self.crop_num += len(dt_boxes)
            start_crop = time()
            for box in dt_boxes:
                img_crop = get_rotate_crop_image(img, box)  # 仿射变换并裁剪，太高则旋转90度横置
                img_dict["imgs"]    .append(img_crop)
                img_dict["dt_boxes"].append(box)
                img_dict["pic_ids"] .append(id)
            self.crop_time += time() - start_crop

        if self.use_angle_cls and cls:
            _, img_dict["imgs"] = self.text_classifier(img_dict["imgs"])

        rec_res = self.text_recognizer(img_dict["imgs"])
        results_list = [{"dt_boxes":[], "text":[], "score":[]} for i in range(len(img_list))]
        for i, id in enumerate(rec_res.get("ids")):
            text, score = rec_res["res"][i]
            if score >= self.rec_thresh:
                pic_id = img_dict["pic_ids"][id]
                results_list[pic_id]["dt_boxes"].append(img_dict["dt_boxes"][id])
                results_list[pic_id]["text"]    .append(text)
                results_list[pic_id]["score"]   .append(score)

        return results_list


def get_rotate_crop_image(img, points):
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width  = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    # align with cpp
    img_crop_width  = max(16, img_crop_width)
    img_crop_height = max(16, img_crop_height)

    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)
    dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def draw_ocr_box_txt(image, boxes, txts, scores=None, rec_thresh=0.5, font_path="../datasets/fonts/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < rec_thresh:
            continue
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon([
            box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
            box[2][1], box[3][0], box[3][1]
        ], outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][1])**2)
        box_width  = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except OSError:
                print("无法打开字体文件，使用默认字体。")
                font = ImageFont.load_default()
            cur_y = box[0][1]
            for c in txt:
                try:    # pillow 9.5.0
                    char_size = font.getsize(c)
                except AttributeError:  # pillow 10
                    char_size = font.getbbox(c)[-2:]
                draw_right.text((box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except OSError:
                print("无法打开字体文件，使用默认字体。")
                font = ImageFont.load_default()
            draw_right.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.asarray(img_show)


def main(opt):
    ts_start = time()

    draw_img_save = "./results/inference_results"
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    ppocrv2_sys = TextSystem(opt)

    img_file_list = []
    file_list = sorted(os.listdir(opt.input), key=(lambda e: int(e.split('.')[0].split('_')[-1])))
    for img_name in file_list:
        img_file = os.path.join(opt.input, img_name)
        img_file_list.append(img_file)
    print('len(img_file_list):', len(img_file_list))

    decode_time = 0.0
    results = {}
    batch_size = opt.batch_size
    for batch_idx in range(0, len(img_file_list), batch_size):
        img_list = []
        # 不是整batch的，转化为1batch进行处理
        if batch_idx + batch_size >= len(img_file_list):
            batch_size = len(img_file_list) - batch_idx
        for idx in range(batch_size):
            start_time = time()
            src_img = cv2.imdecode(np.fromfile(img_file_list[batch_idx+idx], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            decode_time += time() - start_time
            img_list.append(src_img)

        results_list = ppocrv2_sys(img_list)
        for i, result in enumerate(results_list):
            img_name = file_list[batch_idx+i]
            #logging.info(img_name)
            logging.info(result["text"])
            img = Image.fromarray(cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB))

            img_id = os.path.splitext(img_name)[0]
            results[img_id] = [
                {
                    "illegibility": bool(result["score"][j] < opt.rec_thresh),
                    "points": result["dt_boxes"][j].tolist(),
                    "score": float(result["score"][j]),
                    "transcription": result["text"][j],
                }
                for j in range(len(result["text"]))
            ]

            if opt.save_draw:
                draw_img = draw_ocr_box_txt(img, result["dt_boxes"], result["text"], result["score"], rec_thresh=opt.rec_thresh)
                img_path = os.path.join(draw_img_save, "ocr_res_{}".format(img_name))
                cv2.imwrite(img_path, draw_img[:, :, ::-1])
                logging.info("The visualized image saved in {}".format(img_path))

    save_fp = f"./results/ppocr_system_results_b{opt.batch_size}.json"
    with open(save_fp, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    logging.info("result saved in {}".format(save_fp))

    ts_end = time()
    ts_run_total = ts_end - ts_start
    ts_run_per_img = ts_run_total / len(img_file_list)

    # calculate speed
    total_inference_time = 0
    logging.info("------------------ Image Load/Crop Time Info ----------------------")
    decode_time = decode_time / len(img_file_list)
    crop_time = ppocrv2_sys.crop_time / ppocrv2_sys.crop_num
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("crop_time(ms): {:.2f}".format(crop_time * 1000))
    logging.info("------------------ Det Predict Time Info ----------------------")
    preprocess_time = ppocrv2_sys.text_detector.preprocess_time / len(img_file_list)
    inference_time = ppocrv2_sys.text_detector.inference_time / len(img_file_list)
    total_inference_time += inference_time
    postprocess_time = ppocrv2_sys.text_detector.postprocess_time / len(img_file_list)
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
    if opt.use_angle_cls == True:
        logging.info("------------------ Cls Predict Time Info ----------------------")
        preprocess_time = ppocrv2_sys.text_classifier.preprocess_time / ppocrv2_sys.crop_num
        inference_time = ppocrv2_sys.text_classifier.inference_time / ppocrv2_sys.crop_num
        total_inference_time += inference_time
        postprocess_time = ppocrv2_sys.text_classifier.postprocess_time / ppocrv2_sys.crop_num
        logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
        logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
        logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
    logging.info("------------------ Rec Predict Time Info ----------------------")
    preprocess_time = ppocrv2_sys.text_recognizer.preprocess_time / ppocrv2_sys.crop_num
    inference_time = ppocrv2_sys.text_recognizer.inference_time / ppocrv2_sys.crop_num
    total_inference_time += inference_time
    postprocess_time = ppocrv2_sys.text_recognizer.postprocess_time / ppocrv2_sys.crop_num
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
    logging.info("------------------ Total Time Info ----------------------")
    logging.info("total_inference_time(ms): {:.2f}".format(total_inference_time * 1000))
    logging.info("total_run_time(ms): {:.2f}".format((ts_run_total) * 1000))
    logging.info("per_img_run_time(ms): {:.2f}".format((ts_run_per_img) * 1000))


def img_size_type(arg):
    img_sizes = arg.strip('[]').split('],[')
    img_sizes = [size.split(',') for size in img_sizes]
    img_sizes = [[int(width), int(height)] for width, height in img_sizes]
    return img_sizes


if __name__ == '__main__':
    parser = ArgumentParser(prog=__file__)
    parser.add_argument('--dev_id', type=int, default=0, help='tpu card id')
    parser.add_argument('--input', type=str, default='../datasets/cali_set_det', help='input image directory path')
    parser.add_argument("--batch_size", type=int, default=1, help='img num for a ppocr system process launch.')
    parser.add_argument("--save_draw", action='store_true', help='save visualize results')
    # params for text detector
    parser.add_argument('--cvimodel_det', type=str, default='../models/ch_PP-OCRv3_det_infer.onnx', help='detector cvimodel path')
    parser.add_argument('--det_limit_side_len', type=int, default=[640])
    # params for text recognizer
    parser.add_argument('--cvimodel_rec', type=str, default='../models/ch_PP-OCRv3_rec_infer.onnx', help='recognizer cvimodel path')
    parser.add_argument('--img_size', type=img_size_type, default=[[640, 48]], help='You should set inference size [width,height] manually if using multi-stage cvimodel.')
    parser.add_argument("--char_dict_path", type=str, default="../datasets/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=bool, default=True)
    parser.add_argument('--use_beam_search', action='store_const', const=True, default=False, help='Enable beam search')
    parser.add_argument("--beam_size", type=int, default=5, choices=range(1,41), help='Only valid when using beam search, valid range 1~40')
    parser.add_argument("--rec_thresh", type=float, default=0.5)
    # params for text classifier
    parser.add_argument("--use_angle_cls", action='store_true')
    parser.add_argument('--cvimodel_cls', type=str, default='../models/ch_ppocr_mobile_v2.0_cls_infer.onnx', help='classifier cvimodel path')
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_thresh", type=float, default=0.9)
    opt = parser.parse_args()

    main(opt)
