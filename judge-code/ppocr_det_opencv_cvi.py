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
import pyclipper
from shapely.geometry import Polygon
import onnxruntime as ort


class DBPostProcess:

    def __init__(self, thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=1.5):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3

    def boxes_from_bitmap(self, bitmap, dest_width, dest_height):
        ''' bitmap: single map with shape (1, H, W), whose values are binarized as {0, 1} '''
        height, width = bitmap.shape
        contours = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        num_contours = min(len(contours), self.max_candidates)
        boxes = []
        for i in range(num_contours):
            contour = contours[i]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size: continue
            score = self.box_score_fast(bitmap, points)
            #print("score:", score)
            if score < self.box_thresh or score > 2: continue       # FIXME: why >2, what case?
            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2: continue
            box[:, 0] = np.clip(np.round(box[:, 0] / width  * dest_width),  0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
        return np.asarray(boxes, dtype=np.int32)

    def unclip(self, box:ndarray):
        # 扩展 box 为新的 contour
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        poly = Polygon(box)
        distance = poly.area * self.unclip_ratio / poly.length
        expanded = offset.Execute(distance)
        return np.asarray(expanded, dtype=np.int32)

    def get_mini_boxes(self, contour):
        # 最小外接矩形 (带旋转)
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=tuple)

        # 保证顺序: ↖ ↙ ↘ ↗
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return np.asarray(box, dtype=np.int32), min(bounding_box[1])

    def box_score_fast(self, bitmap:ndarray, box:ndarray):
        ''' box_score_fast: use bbox mean score as the mean score '''
        h, w = bitmap.shape[:2]
        # 旋转框
        box = box.copy()
        xmin = np.clip(box[:, 0].min(), 0, w - 1)
        xmax = np.clip(box[:, 0].max(), 0, w - 1)
        ymin = np.clip(box[:, 1].min(), 0, h - 1)
        ymax = np.clip(box[:, 1].max(), 0, h - 1)
        # 旋转框的正外接矩形
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2), 1)
        # 外接矩形对应原图位置
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps'][:, 0, :, :]  # [B, C=1, H, W] => [B, H, W]
        segmentation = (pred > self.thresh).astype(np.uint8)    # binarize

        boxes_batch = []
        for i in range(pred.shape[0]):
            src_h, src_w = shape_list[i]
            mask = segmentation[i]
            boxes = self.boxes_from_bitmap(mask, src_w, src_h)
            boxes_batch.append({'points': boxes})
        return boxes_batch


class PPOCRv2Det:

    def __init__(self, args):
        # load cvimodel
        model_path = args.cvimodel_det
        logging.info("using model {}".format(model_path))
        # self.net = sail.Engine(model_path, args.dev_id, sail.IOMode.SYSIO)
        self.net = ort.InferenceSession(model_path)
        node_input = self.net.get_inputs()[0]
        node_output = self.net.get_outputs()[0]
        print(f'>> [input] name: {node_input.name}, shape: {node_input.shape}')
        print(f'>> [output] name: {node_output.name}, shape: {node_output.shape}')
        self.graph_name = 'ch_PP-OCRv3_det'
        self.input_name = 'x'
        self.input_shape = [1, 3, 640, 640]
        self.det_batch_size = 1
        logging.info("load cvimodel success!")
        # preprocess
        self.input_size_level = sorted([self.input_shape[2], self.input_shape[3]])
        self.mean  = np.asarray([0.485,   0.456,   0.406  ], dtype=np.float32).reshape((1, 1, 3)) * 255.0
        self.scale = np.asarray([1/0.229, 1/0.224, 1/0.225], dtype=np.float32).reshape((1, 1, 3)) / 255.0
        # postprocess
        self.postprocess_op = DBPostProcess()
        self.min_size = self.postprocess_op.min_size
        # perfcnt
        self.preprocess_time  = 0.0
        self.inference_time   = 0.0
        self.postprocess_time = 0.0

    def preprocess(self, im:ndarray):
        # 小图不放大，大图等比例缩小到模型输入上限
        h, w, _ = im.shape
        size_max = max(h, w)
        if size_max >= self.input_size_level[-1]:
            canvas_size = self.input_size_level[-1]
            ratio = float(canvas_size) / size_max
        else:
            for side_len in self.input_size_level:
                if size_max <= side_len:
                    canvas_size = side_len
                    ratio = 1.
                    break
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
        # 在 uint8 域上 resize (?)
        if h != resize_h or w != resize_w:
            im = cv2.resize(im, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

        # to [C, H, W], f32
        im: ndarray
        im = im.astype(np.float32)
        im = im - self.mean
        im = im * self.scale
        im = im.transpose(2, 0, 1)

        # right bottom pad 0
        im_pad = np.zeros((3, canvas_size, canvas_size), dtype=np.float32)
        im_pad[:, :resize_h, :resize_w] = im
        return im_pad, (h, w, resize_h, resize_w)

    def predict(self, x):
        # [B, C, H, W], logits
        return self.net.run([self.net.get_outputs()[0].name], {self.input_name: x})[0]

    def postprocess(self, outputs, src_h, src_w, resize_h, resize_w):
        preds = {}
        preds['maps'] = np.expand_dims(outputs[:, :resize_h, :resize_w], axis=0)
        shape_list = np.asarray([[src_h, src_w]])
        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        #print('dt_boxes:', dt_boxes)
        #dt_boxes = self.filter_tag_det_res(dt_boxes, (src_h, src_w, 3))
        return dt_boxes

    def filter_tag_det_res(self, dt_boxes:ndarray, image_shape:tuple) -> ndarray:
        img_height, img_width = image_shape[:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width  = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= self.min_size or rect_height <= self.min_size:
                continue
            dt_boxes_new.append(box)
        return np.asarray(dt_boxes_new, dtype=dt_boxes.dtype)

    def order_points_clockwise(self, pts:ndarray) -> ndarray:
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]
        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost  = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        (tl, bl) = leftMost [np.argsort(leftMost [:, 1]), :]
        (tr, br) = rightMost[np.argsort(rightMost[:, 1]), :]
        return np.asarray([tl, tr, br, bl], dtype=pts.dtype)

    def clip_det_res(self, points, img_height, img_width):
        for p in points:
            p[0] = int(min(max(p[0], 0), img_width  - 1))
            p[1] = int(min(max(p[1], 0), img_height - 1))
        return points

    def __call__(self, img_list:List[ndarray]):
        img_num = len(img_list)
        img_input_list = []
        img_size_list = []

        start_pre = time()
        for img in img_list:
            img, [src_h, src_w, resize_h, resize_w] = self.preprocess(img)
            img_input_list.append(img)
            img_size_list.append([src_h, src_w, resize_h, resize_w])
        self.preprocess_time += time() - start_pre

        start_infer = time()
        outputs_list = []
        for beg_img_no in range(0, img_num, self.det_batch_size):
            end_img_no = min(img_num, beg_img_no + self.det_batch_size)
            if beg_img_no + self.det_batch_size > img_num:
                for ino in range(beg_img_no, end_img_no):
                    img_input = np.expand_dims(img_input_list[ino], axis=0)
                    outputs = self.predict(img_input)
                    outputs_list.extend(outputs)
            else:
                img_input = np.stack(img_input_list[beg_img_no:end_img_no])
                outputs = self.predict(img_input)
                outputs_list.extend(outputs)
        self.inference_time += time() - start_infer

        start_post = time()
        dt_boxes_list = []
        for id, outputs in enumerate(outputs_list):
            src_h, src_w, resize_h, resize_w = img_size_list[id]
            dt_boxes = self.postprocess(outputs, src_h, src_w, resize_h, resize_w)
            dt_boxes_list.append(dt_boxes)
        self.postprocess_time += time() - start_post

        return dt_boxes_list


def draw_text_det_res(dt_boxes, img_path):
    im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.asarray(box, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(im, [box], True, color=(255, 255, 0), thickness=2)
    return im


def main(opt):
    ppocrv2_det = PPOCRv2Det(opt)

    file_list = sorted(os.listdir(opt.input), key=(lambda e: int(e.split('.')[0].split('_')[-1])))
    img_list: List[ndarray] = []
    for img_name in file_list:
        img_file = os.path.join(opt.input, img_name)
        src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)   # [H, W, C=3], u8 RGB
        img_list.append(src_img)

    draw_img_save = "./results/det_results"
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)

    dt_boxes_list = ppocrv2_det(img_list)
    for img_name, dt_boxes in zip(file_list, dt_boxes_list):
        image_file = os.path.join(opt.input, img_name)
        draw_im = draw_text_det_res(dt_boxes, image_file)
        img_name_pure = os.path.split(image_file)[-1]
        img_path = os.path.join(draw_img_save, "det_res_{}".format(img_name_pure))
        cv2.imwrite(img_path, draw_im)
        logging.info("The visualized image saved in {}".format(img_path))


if __name__ == '__main__':
    parser = ArgumentParser(prog=__file__)
    parser.add_argument('--dev_id', type=int, default=0, help='tpu card id')
    parser.add_argument('--input', type=str, default='../datasets/cali_set_det', help='input image directory path')
    parser.add_argument('--cvimodel_det', type=str, default='../models/ch_PP-OCRv3_det_infer.onnx', help='cvimodel path')
    opt = parser.parse_args()

    main(opt)
