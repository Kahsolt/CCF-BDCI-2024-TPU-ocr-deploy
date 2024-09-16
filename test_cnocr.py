#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/16 

# 实验性脚本: 跑 CnOCR 推理
# https://cnocr.readthedocs.io/zh-cn/stable/models/
# NOTE: CnOCR 的所有自训练识别模型都 **不支持竖排文字**

'''
| sample id | n_boxes | ppocr_v3 (onnx) | ppocr_v2 (onnx) | shflnet_v2 (pytorch) | mbnet_v3 (pytorch) |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 11.jpg       | 16/18/23 | 257.82ms | 235.94ms | 532.78ms | 478.98ms |
| 12.jpg       |     4    | 126.57ms | 119.69ms | 391.74ms | 327.83ms |
| 00207393.jpg |     4    | 144.20ms | 128.39ms | 393.02ms | 337.06ms |
'''

from time import time
from pathlib import Path
from PIL import Image

from PIL import Image
import numpy as np
from tqdm import tqdm
from cnocr import CnOcr

BASE_PATH = Path(__file__).parent
PP_DATA_PATH = BASE_PATH / 'data' / 'ppocr_img'
assert PP_DATA_PATH.is_dir(), '>> You should first download https://paddleocr.bj.bcebos.com/dygraph_v2.1/ppocr_img.zip'

#img_path = PP_DATA_PATH / 'imgs/1.jpg'  # 忽略竖排样本
img_path = PP_DATA_PATH / 'imgs/11.jpg'
#img_path = PP_DATA_PATH / 'imgs/12.jpg'
#img_path = PP_DATA_PATH / 'imgs/00207393.jpg'

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
