#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/17 

# 基于开源repo推理图片文件夹

'''
| repo | avg(time) |
| :-: | :-: |
| ppocr (v4)      |  739.95ms |
| ppocr (v3)      |  594.67ms |
| ppocr (v2)      |  501.94ms |
| cnocr (v3)      |  196.97ms |
| cnocr (v2)      |  174.62ms |
| rapidocr        | 1433.46ms |
| chineseocr_lite |  209.94ms |
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
    ocr_version='PP-OCRv4',
    lang='ch',
    use_angle_cls=False,
    precision='fp32',
    show_log=False,
  )
  ocr.ocr(np.random.randint(low=0, high=255, size=(256, 256, 3), dtype=np.uint8), cls=False)

  results: InferResults = []
  fps = sorted(Path(args.img_folder).iterdir())
  for fp in tqdm(fps):
    im = np.array(Image.open(fp).convert('RGB'))
    ts_start = time()
    result = ocr.ocr(im, cls=False)
    ts_end = time()

    # TODO: analyze result

    results.append({
      'Precision': 0.0,
      'Recall': 0.0,
      'F1-Score': 0.0,
      'i_time': (ts_end - ts_start) * 1000,
    })

  return results


def run_cnocr(args) -> InferResults:
  from cnocr import CnOcr

  ocr = CnOcr(
    det_model_name='ch_PP-OCRv3_det',
    rec_model_name='scene-densenet_lite_136-gru',
    det_model_backend='onnx',
    rec_model_backend='onnx',
  )
  ocr.ocr(np.random.randint(low=0, high=255, size=(256, 256, 3), dtype=np.uint8))

  results: InferResults = []
  fps = sorted(Path(args.img_folder).iterdir())
  for fp in tqdm(fps):
    im = np.array(Image.open(fp).convert('RGB'))
    ts_start = time()
    result = ocr.ocr(im)
    ts_end = time()

    # TODO: analyze result

    results.append({
      'Precision': 0.0,
      'Recall': 0.0,
      'F1-Score': 0.0,
      'i_time': (ts_end - ts_start) * 1000,
    })

  return results


def run_rapidocr(args) -> InferResults:
  from rapidocr_onnxruntime import RapidOCR

  ocr = RapidOCR()
  ocr(np.random.randint(low=0, high=255, size=(256, 256, 3), dtype=np.uint8))

  results: InferResults = []
  fps = sorted(Path(args.img_folder).iterdir())
  for fp in tqdm(fps):
    im = np.array(Image.open(fp).convert('RGB'))
    ts_start = time()
    result, elapse = ocr(im)
    ts_end = time()

    # TODO: analyze result

    results.append({
      'Precision': 0.0,
      'Recall': 0.0,
      'F1-Score': 0.0,
      'i_time': (ts_end - ts_start) * 1000,
    })

  return results


def run_chineseocr_lite(args) -> InferResults:
  REPO_LIB_PATH = REPO_PATH / 'chineseocr_lite'
  assert REPO_LIB_PATH.is_dir(), '>> You should first git clone https://github.com/DayBreak-u/chineseocr_lite'

  import sys ; sys.path.insert(0, str(REPO_LIB_PATH))
  from model import OcrHandle
  import onnxruntime as ort
  ort.set_default_logger_severity(3)
  np.int = np.int32

  short_size = 640
  ocr = OcrHandle()
  ocr.text_predict(np.random.randint(low=0, high=255, size=(256, 256, 3), dtype=np.uint8), short_size)

  results: InferResults = []
  fps = sorted(Path(args.img_folder).iterdir())
  for fp in tqdm(fps):
    im = np.array(Image.open(fp).convert('RGB'))
    ts_start = time()
    result = ocr.text_predict(im, short_size)
    ts_end = time()

    # TODO: analyze result

    results.append({
      'Precision': 0.0,
      'Recall': 0.0,
      'F1-Score': 0.0,
      'i_time': (ts_end - ts_start) * 1000,
    })

  return results


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-K', '--backend',    default='ppocr', choices=['ppocr', 'cnocr', 'rapidocr', 'chineseocr_lite'])
  parser.add_argument('-I', '--img_folder', default=DEFAULT_INPUT_FOLDER, type=Path)
  parser.add_argument('-O', '--save_file',  default=DEFAULT_SAVE_FILE,    type=Path)
  args = parser.parse_args()

  assert Path(args.img_folder).is_dir()

  results = globals()[f'run_{args.backend}'](args)
  save_infer_results(results, args.save_file)
