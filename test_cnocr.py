#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/16 

# 实验性脚本: 跑 CnOCR 推理
# https://cnocr.readthedocs.io/zh-cn/stable/models/
# NOTE: CnOCR 的所有自训练识别模型都 **不支持竖排文字**

'''
| sample id | n_boxes | ppocr_v3 (onnx) | ppocr_v2 (onnx) | shflnet_v2 (pytorch) | mbnet_v3 (pytorch) |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 11.jpg       | 16/18/23 | 277.27ms | 250.89ms | 532.78ms | 478.98ms |
| 12.jpg       |     4    | 131.87ms | 122.05ms | 391.74ms | 327.83ms |
| 00207393.jpg |     4    | 144.38ms | 129.24ms | 393.02ms | 337.06ms |
'''

from time import time
from pathlib import Path
from PIL import Image

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

ts_start = time()
result = ocr.ocr(Image.open(img_path).convert('RGB'))
ts_end = time()
print(f'>> time cost: {(ts_end - ts_start) * 1000:.2f}ms')

print('>> n_boxes:', len(result))
for idx in range(len(result)):
  res = result[idx]
  print(res['text'], res['score'])
