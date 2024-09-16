#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/16 

# 实验性脚本: 跑 ChineseOCR-Lite 推理
# https://github.com/DayBreak-u/chineseocr_lite

'''
| sample id | n_boxes | short_size=960 | short_size=640 |
| :-: | :-: | :-: | :-: |
| 1.jpg        |  2 | 327.56ms | 143.78ms |
| 11.jpg       | 16 | 262.11ms | 147.21ms |
| 12.jpg       |  4 | 339.22ms | 156.48ms |
| 00207393.jpg |  6 | 273.67ms | 147.34ms |
'''

from time import time
from pathlib import Path

from PIL import Image
import numpy as np
from tqdm import tqdm

BASE_PATH = Path(__file__).parent
PP_DATA_PATH = BASE_PATH / 'data' / 'ppocr_img'
assert PP_DATA_PATH.is_dir(), '>> You should first download https://paddleocr.bj.bcebos.com/dygraph_v2.1/ppocr_img.zip'
REPO_PATH = BASE_PATH / 'repo' / 'chineseocr_lite'
assert REPO_PATH.is_dir(), '>> You should first git clone https://github.com/DayBreak-u/chineseocr_lite'

import sys ; sys.path.insert(0, str(REPO_PATH))
import onnxruntime as ort
ort.set_default_logger_severity(3)
from model import OcrHandle
np.int = np.int32

img_path = PP_DATA_PATH / 'imgs/1.jpg'
#img_path = PP_DATA_PATH / 'imgs/11.jpg'
#img_path = PP_DATA_PATH / 'imgs/12.jpg'
#img_path = PP_DATA_PATH / 'imgs/00207393.jpg'

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
