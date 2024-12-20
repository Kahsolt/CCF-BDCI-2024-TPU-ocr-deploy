#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/17 

# 基于开源repo推理图片文件夹，保存结果并估算得分

'''
| method | mean(f1) | mean(runtime) | estimated_score | submit_score (f1/prec/recall) |
| :-: | :-: | :-: | :-: | :-: |
| ppocr (v4)  | 0.6931 | 613.44ms |  65.58/100 | 0.67250719747/0.80938897168/0.57522639411 (100.00) |
| ppocr (v3)  | 0.6627 | 455.55ms |  77.79/100 | 0.64444587712/0.82346157015/0.52936503734 (100.00) |
| ppocr (v2)  | 0.6402 | 153.69ms | 100.00/100 | 0.62266139657/0.82356016381/0.50055605571 (100.00) |
| cnocr (v3)  | 0.4845 | 137.97ms |  97.66/100 | 0.46649698118/0.52874874203/0.41735952974 (100.00) |
| cnocr (v2)  | 0.4628 | 121.34ms |  98.20/100 | 0.44589615778/0.53674716756/0.38134830271 (100.00) |
| chocr (960) | 0.4205 | 334.60ms |  78.38/100 | 0.41097834417/0.44281345566/0.38341365249 (100.00) |
| chocr (640) | 0.4310 | 147.82ms |  94.67/100 | 0.40900939238/0.48002283105/0.35629931685 (100.00) |
'''

from time import time
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from tqdm import tqdm

from xutils import *


def run_ppocr(args) -> InferResults:
  from paddleocr.paddleocr import PaddleOCR

  ocr = PaddleOCR(
    ocr_version=f'PP-OCR{args.ppocr_ver}',
    lang='ch',
    use_angle_cls=False,
    precision='fp32',
    show_log=False,
  )
  ocr.ocr(np.random.randint(low=0, high=255, size=(256, 256, 3), dtype=np.uint8), cls=False)

  results: InferResults = {}
  fps = sorted(Path(args.img_folder).iterdir())
  for fp in tqdm(fps):
    im = np.array(Image.open(fp).convert('RGB'))
    ts_start = time()
    result = ocr.ocr(im, cls=False)[0] or []
    ts_end = time()

    results[fp.stem] = InferResult(
      [InferAnnot(e[1][0], e[0]) for e in result],
      (ts_end - ts_start) * 1000,
    )

  return results


def run_cnocr(args) -> InferResults:
  from cnocr import CnOcr

  ocr = CnOcr(
    det_model_name=f'ch_PP-OCR{args.cnocr_ver}_det',
    rec_model_name='scene-densenet_lite_136-gru',
    det_model_backend='onnx',
    rec_model_backend='onnx',
  )
  ocr.ocr(np.random.randint(low=0, high=255, size=(256, 256, 3), dtype=np.uint8))

  results: InferResults = {}
  fps = sorted(Path(args.img_folder).iterdir())
  for fp in tqdm(fps):
    im = np.array(Image.open(fp).convert('RGB'))
    ts_start = time()
    result = ocr.ocr(im) or []
    ts_end = time()

    results[fp.stem] = InferResult(
      [InferAnnot(e['text'], e['position'].tolist()) for e in result],
      (ts_end - ts_start) * 1000,
    )

  return results


def run_rapidocr(args) -> InferResults:
  from rapidocr_onnxruntime import RapidOCR

  ocr = RapidOCR()
  ocr(np.random.randint(low=0, high=255, size=(256, 256, 3), dtype=np.uint8))

  results: InferResults = {}
  fps = sorted(Path(args.img_folder).iterdir())
  for fp in tqdm(fps):
    im = np.array(Image.open(fp).convert('RGB'))
    ts_start = time()
    result, elapse = ocr(im)
    ts_end = time()

    # TODO: analyze result

  return results


def run_chineseocr_lite(args) -> InferResults:
  REPO_LIB_PATH = REPO_PATH / 'chineseocr_lite'
  assert REPO_LIB_PATH.is_dir(), '>> You should first git clone https://github.com/DayBreak-u/chineseocr_lite'

  # FIXME: repo/chineseocr_lite/model.py:71 行，去掉序号和顿号!!
  import sys ; sys.path.insert(0, str(REPO_LIB_PATH))
  from model import OcrHandle
  import onnxruntime as ort
  ort.set_default_logger_severity(3)
  np.int = np.int32

  short_size = args.chocr_short_size
  ocr = OcrHandle()
  ocr.text_predict(np.random.randint(low=0, high=255, size=(256, 256, 3), dtype=np.uint8), short_size)

  results: InferResults = {}
  fps = sorted(Path(args.img_folder).iterdir())
  for fp in tqdm(fps):
    im = np.array(Image.open(fp).convert('RGB'))
    ts_start = time()
    result = ocr.text_predict(im, short_size) or []
    ts_end = time()

    results[fp.stem] = InferResult(
      [InferAnnot(e[1], e[0].tolist()) for e in result],
      (ts_end - ts_start) * 1000,
    )

  return results


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-K', '--backend',    default='ppocr', choices=['ppocr', 'cnocr', 'rapidocr', 'chineseocr_lite'])
  parser.add_argument('-I', '--img_folder', default=DEFAULT_INPUT_FOLDER, type=Path)
  parser.add_argument('-O', '--save_file',  default=DEFAULT_SAVE_FILE,    type=Path)
  # repo-wise opts
  parser.add_argument('--ppocr_ver', default='v4', choices=['v4', 'v3', 'v2'])
  parser.add_argument('--cnocr_ver', default='v3', choices=['v3', 'v2'])
  parser.add_argument('--chocr_short_size', default=960, type=int)
  args = parser.parse_args()

  assert Path(args.img_folder).is_dir()

  results: InferResults = globals()[f'run_{args.backend}'](args)    # NOTE: This takes 15min~20min
  save_infer_results(results, args.save_file)
  metrics = infer_results_to_metrics(results)
  save_metrics(metrics, Path(args.save_file).with_suffix('.metrics.json'))
