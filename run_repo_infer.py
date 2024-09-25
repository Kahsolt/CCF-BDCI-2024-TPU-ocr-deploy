#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/17 

# 基于开源repo推理图片文件夹，保存结果并估算得分

'''
| method | mean(f1) | mean(runtime) | estimated_score | submit_score |
| :-: | :-: | :-: | :-: | :-: |
| ppocr (v4)  | 0.6931 |  620.44ms |  64.99/100.00 |             |
| ppocr (v3)  | 0.6627 |  472.93ms |  76.31/100.00 |     全      |
| ppocr (v2)  | 0.6402 |  157.85ms | 100.00/100.00 |     部      |
| cnocr (v3)  | 0.2423 |  136.68ms |  88.07/ 99.62 |     都      |
| cnocr (v2)  | 0.2314 |  116.86ms |  89.32/ 99.19 |     是      |
| chocr (960) | 0.4205 |  333.30ms |  78.49/100.00 | 89.94687500 |
| chocr (640) | 0.4310 |  142.11ms |  95.16/100.00 |             |
'''

from time import time
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from tqdm import tqdm

from xutils import *


def run_ppocr(args, annots:Annots) -> InferResults:
  from paddleocr.paddleocr import PaddleOCR

  ocr = PaddleOCR(
    ocr_version=f'PP-OCR{args.ppocr_ver}',
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
    result = ocr.ocr(im, cls=False)[0]
    ts_end = time()

    id = fp.stem
    annot_pred = {id: [BBox(e[1][0], e[0]) for e in result] if result else []}
    annot_true = {id: annots[id]}
    f_score, precision, recall = calc_f1(annot_pred, annot_true)

    results.append({
      'Precision': precision,
      'Recall': recall,
      'F1-Score': f_score,
      'i_time': (ts_end - ts_start) * 1000,
    })

  return results


def run_cnocr(args, annots:Annots) -> InferResults:
  from cnocr import CnOcr

  ocr = CnOcr(
    det_model_name=f'ch_PP-OCR{args.cnocr_ver}_det',
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

    id = fp.stem
    annot_pred = {id: [BBox(e['text'], e['position'].tolist()) for e in result] if result else []}
    annot_true = {id: annots[id]}
    f_score, precision, recall = calc_f1(annot_pred, annot_true)

    results.append({
      'Precision': precision,
      'Recall': recall,
      'F1-Score': f_score,
      'i_time': (ts_end - ts_start) * 1000,
    })

    results.append({
      'Precision': 0.0,
      'Recall': 0.0,
      'F1-Score': 0.0,
      'i_time': (ts_end - ts_start) * 1000,
    })

  return results


def run_rapidocr(args, annots:Annots) -> InferResults:
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


def run_chineseocr_lite(args, annots:Annots) -> InferResults:
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

  results: InferResults = []
  fps = sorted(Path(args.img_folder).iterdir())
  for fp in tqdm(fps):
    im = np.array(Image.open(fp).convert('RGB'))
    ts_start = time()
    result = ocr.text_predict(im, short_size)
    ts_end = time()

    id = fp.stem
    annot_pred = {id: [BBox(e[1], e[0].tolist()) for e in result] if result else []}
    annot_true = {id: annots[id]}
    f_score, precision, recall = calc_f1(annot_pred, annot_true)

    results.append({
      'Precision': precision,
      'Recall': recall,
      'F1-Score': f_score,
      'i_time': (ts_end - ts_start) * 1000,
    })

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

  annots = load_annots()
  results = globals()[f'run_{args.backend}'](args, annots)    # NOTE: This takes ~15min
  save_infer_results(results, args.save_file)

  f1 = mean([e['F1-Score'] for e in results])
  ts = mean([e['i_time']   for e in results])
  print('mean(f1)',      f1)
  print('mean(runtime)', ts)
  print('Estimated score:',          min(100, 90 + 40 * f1 - 0.085 * ts))
  print('Estimated score (scaled):', min(100, 90 + 40 * f1 - 0.085 * ts / 160))
