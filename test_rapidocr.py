#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/16 

# 实验性脚本: 跑 RapidOCR 推理 (它只是 PaddlePaddleOCR 的 onnx发行版)
# https://rapidai.github.io/RapidOCRDocs/quickstart

'''
| sample id | n_boxes | PP-OCRv4 |
| :-: | :-: | :-: |
| 1.jpg        |  2 | 320.23ms |
| 11.jpg       | 16 | 778.04ms |
| 12.jpg       |  4 | 368.64ms |
| 00207393.jpg |  4 | 420.30ms |
'''

from time import time
from pathlib import Path

from PIL import Image
import numpy as np
from tqdm import tqdm
from rapidocr_onnxruntime import RapidOCR

BASE_PATH = Path(__file__).parent
PP_DATA_PATH = BASE_PATH / 'data' / 'ppocr_img'
assert PP_DATA_PATH.is_dir(), '>> You should first download https://paddleocr.bj.bcebos.com/dygraph_v2.1/ppocr_img.zip'

#img_path = PP_DATA_PATH / 'imgs/1.jpg'
img_path = PP_DATA_PATH / 'imgs/11.jpg'
#img_path = PP_DATA_PATH / 'imgs/12.jpg'
#img_path = PP_DATA_PATH / 'imgs/00207393.jpg'

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
