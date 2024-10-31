#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/16 

# 基于开源repo推理单张图片

from time import time
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from tqdm import tqdm

from xutils import BASE_PATH, DATA_PATH, REPO_PATH


def run_ppocr(img_path:Path):
  '''
  # https://paddlepaddle.github.io/PaddleOCR/ppocr/quick_start.html
  | sample id | n_boxes | PP-OCRv4 | PP-OCRv3 | PP-OCRv2 |
  | :-: | :-: | :-: | :-: | :-: |
  | 1.jpg        |  2 | 129.70ms |  87.08ms |  78.88ms |
  | 11.jpg       | 16 | 519.30ms | 212.81ms | 164.08ms |
  | 12.jpg       |  4 | 178.11ms | 102.99ms |  87.35ms |
  | 00207393.jpg |  4 | 160.64ms |  74.43ms |  56.36ms |
  '''

  from paddleocr.paddleocr import PaddleOCR
  from paddleocr.tools.infer.utility import draw_ocr
  from paddleocr.tools.infer.predict_det import TextDetector
  from paddleocr.tools.infer.predict_cls import TextClassifier
  from paddleocr.tools.infer.predict_rec import TextRecognizer

  font_path = DATA_PATH / 'fonts/simfang.ttf'

  ocr = PaddleOCR(
    # version
    ocr_version='PP-OCRv4',
    # components
    lang='ch',
    use_angle_cls=False,
    # backends
    use_onnx=False,
    use_tensorrt=False,
    # precision
    precision='fp32',   # wtf, fp16/int8 makes no differences ??!
  )
  #det: TextDetector = ocr.text_detector
  #cla: TextClassifier = ocr.text_classifier
  #rec: TextRecognizer = ocr.text_recognizer

  # warmup
  im = np.array(Image.open(img_path))
  result = ocr.ocr(im, cls=False)
  ts_list = []
  # average 10 tests
  for _ in tqdm(range(10)):
    ts_start = time()
    result = ocr.ocr(im, cls=False)
    ts_end = time()
    ts_list.append(ts_end - ts_start)
  ts = sum(ts_list) / len(ts_list)
  print(f'>> time cost: {(ts) * 1000:.2f}ms')

  for idx in range(len(result)):
    res = result[idx]
    for line in res:
      print(line)

  if not 'draw':
    result = result[0]
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.show()


def run_cnocr(img_path:Path):
  '''
  # https://cnocr.readthedocs.io/zh-cn/stable/models/
  # NOTE: CnOCR 的所有自训练识别模型都 **不支持竖排文字**
  | sample id | n_boxes | ppocr_v3 (onnx) | ppocr_v2 (onnx) | shflnet_v2 (pytorch) | mbnet_v3 (pytorch) |
  | :-: | :-: | :-: | :-: | :-: | :-: |
  | 11.jpg       | 16/18/23 | 257.82ms | 235.94ms | 532.78ms | 478.98ms |
  | 12.jpg       |     4    | 126.57ms | 119.69ms | 391.74ms | 327.83ms |
  | 00207393.jpg |     4    | 144.20ms | 128.39ms | 393.02ms | 337.06ms |
  '''

  from cnocr import CnOcr

  ocr = CnOcr(
    det_model_name='ch_PP-OCRv3_det',               # 可以换
    rec_model_name='scene-densenet_lite_136-gru',   # 固定用 scene 权重
    det_model_backend='onnx',                       # 适配 det_model_name
    rec_model_backend='onnx',                       # 固定用 onnx 版本
  )

  # warmup
  im = np.array(Image.open(img_path))
  result = ocr.ocr(im)
  ts_list = []
  # average 10 tests
  for _ in tqdm(range(10)):
    ts_start = time()
    result = ocr.ocr(im)
    ts_end = time()
    ts_list.append(ts_end - ts_start)
  ts = sum(ts_list) / len(ts_list)
  print(f'>> time cost: {(ts) * 1000:.2f}ms')

  print('>> n_boxes:', len(result))
  for idx in range(len(result)):
    res = result[idx]
    print(res['text'], res['score'])


def run_rapidocr(img_path:Path):
  '''
  # https://rapidai.github.io/RapidOCRDocs/quickstart
  | sample id | n_boxes | PP-OCRv4 |
  | :-: | :-: | :-: |
  | 1.jpg        |  2 | 320.23ms |
  | 11.jpg       | 16 | 778.04ms |
  | 12.jpg       |  4 | 368.64ms |
  | 00207393.jpg |  4 | 420.30ms |
  '''

  from rapidocr_onnxruntime import RapidOCR

  # The default ckpt `ch_PP-OCRv4_*_infer.onnx` is bundled in site-packages
  # for other distro please donwload by yourself
  ocr = RapidOCR()

  # warmup
  im = np.array(Image.open(img_path))
  result, elapse = ocr(im)
  ts_list = []
  # average 10 tests
  for _ in tqdm(range(10)):
    ts_start = time()
    result, elapse = ocr(im)
    ts_end = time()
    ts_list.append(ts_end - ts_start)
  ts = sum(ts_list) / len(ts_list)
  print(f'>> time cost: {(ts) * 1000:.2f}ms')

  for idx in range(len(result)):
    res = result[idx]
    for line in res:
      print(line)


def run_chineseocr_lite(img_path:Path):
  '''
  # https://github.com/DayBreak-u/chineseocr_lite
  | sample id | n_boxes | short_size=960 | short_size=640 |
  | :-: | :-: | :-: | :-: |
  | 1.jpg        |  2 | 327.56ms | 143.78ms |
  | 11.jpg       | 16 | 262.11ms | 147.21ms |
  | 12.jpg       |  4 | 339.22ms | 156.48ms |
  | 00207393.jpg |  6 | 273.67ms | 147.34ms |
  '''

  REPO_LIB_PATH = REPO_PATH / 'chineseocr_lite'
  assert REPO_LIB_PATH.is_dir(), '>> You should first git clone https://github.com/DayBreak-u/chineseocr_lite'

  import sys ; sys.path.insert(0, str(REPO_LIB_PATH))
  import onnxruntime as ort
  ort.set_default_logger_severity(3)
  from model import OcrHandle
  np.int = np.int32

  short_size = 640
  ocr = OcrHandle()

  # warmup
  im = np.array(Image.open(img_path))
  result = ocr.text_predict(im, short_size)
  ts_list = []
  # average 10 tests
  for _ in tqdm(range(10)):
    ts_start = time()
    result = ocr.text_predict(im, short_size)
    ts_end = time()
    ts_list.append(ts_end - ts_start)
  ts = sum(ts_list) / len(ts_list)
  print(f'>> time cost: {(ts) * 1000:.2f}ms')

  for idx in range(len(result)):
    res = result[idx]
    for line in res:
      print(line)


if __name__ == '__main__':
  img_path = DATA_PATH / 'train_full_images_0' / 'gt_97.jpg'

  parser = ArgumentParser()
  parser.add_argument('-K', '--backend',  default='ppocr', choices=['ppocr', 'cnocr', 'rapidocr', 'chineseocr_lite'])
  parser.add_argument('-I', '--img_path', default=img_path, type=Path)
  args = parser.parse_args()

  assert Path(args.img_path).is_file()

  globals()[f'run_{args.backend}'](img_path)
