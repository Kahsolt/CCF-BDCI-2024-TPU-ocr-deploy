#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/16 

# 实验性脚本: 跑 PaddlePaddleOCR 推理
# https://paddlepaddle.github.io/PaddleOCR/ppocr/quick_start.html

'''
| sample id | n_boxes | PP-OCRv4 | PP-OCRv3 | PP-OCRv2 |
| :-: | :-: | :-: | :-: | :-: |
| 1.jpg        |  2 |  547.33ms | 507.17ms | 484.80ms |
| 11.jpg       | 16 | 1085.22ms | 753.76ms | 569.25ms |
| 12.jpg       |  4 |  564.77ms | 530.41ms | 443.50ms |
| 00207393.jpg |  4 |  511.48ms | 473.04ms | 421.85ms |
'''

from time import time
from pathlib import Path

from PIL import Image
from paddleocr.paddleocr import PaddleOCR
from paddleocr.tools.infer.utility import draw_ocr
from paddleocr.tools.infer.predict_det import TextDetector
from paddleocr.tools.infer.predict_cls import TextClassifier
from paddleocr.tools.infer.predict_rec import TextRecognizer

BASE_PATH = Path(__file__).parent
PP_DATA_PATH = BASE_PATH / 'data' / 'ppocr_img'
assert PP_DATA_PATH.is_dir(), '>> You should first download https://paddleocr.bj.bcebos.com/dygraph_v2.1/ppocr_img.zip'

#img_path = PP_DATA_PATH / 'imgs/1.jpg'
img_path = PP_DATA_PATH / 'imgs/11.jpg'
#img_path = PP_DATA_PATH / 'imgs/12.jpg'
#img_path = PP_DATA_PATH / 'imgs/00207393.jpg'
font_path= PP_DATA_PATH / 'fonts/simfang.ttf'

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

ts_start = time()
result = ocr.ocr(str(img_path), cls=True)
ts_end = time()
print(f'>> time cost: {(ts_end - ts_start) * 1000:.2f}ms')

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
