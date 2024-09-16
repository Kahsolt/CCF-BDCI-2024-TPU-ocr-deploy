### 模型选型

索引

- OCR调研报告: https://blog.csdn.net/weixin_41021342/article/details/127203654
- 十二款开源OCR开箱测评: http://www.ceietn.com/a/zhengcefagui/guojiaji/930.html
- 6款开源中文OCR使用介绍: https://blog.csdn.net/bugang4663/article/details/131720149
- PaddleOCR 3.5M模型: https://blog.csdn.net/moxibingdao/article/details/108765380
- 开源OCR模型对比: https://www.cnblogs.com/shiwanghualuo/p/18139459

项目

- https://github.com/PaddlePaddle/PaddleOCR ⭐
  - https://arxiv.org/abs/2206.03001
- https://github.com/breezedeus/CnOCR
- https://github.com/RapidAI/RapidOCR
- https://github.com/chineseocr/chineseocr (停更)
- https://github.com/chineseocr/darknet-ocr (停更)
- https://github.com/ouyanghuiyu/chineseocr_lite (停更)
  - onnx port: https://github.com/benjaminwan/OcrLiteOnnx
  - ncnn port: https://github.com/benjaminwan/OcrLiteNcnn
- https://github.com/myhub/tr
  - webui: https://github.com/alisen39/TrWebOCR
- https://github.com/jaidedai/easyocr
- https://github.com/clovaai/deep-text-recognition-benchmark


### 前车之鉴

- 移植 https://docs.qq.com/pdf/DSUlabGVFRlBkQkZv
- 推理 https://github.com/sophgo/sophon-demo/tree/release/sample/PP-OCR
  - 将 ch_PP-OCRv3_xx 系列模型迁移至 BM1684/BM1684X/BM1688/CV186X
    - [BM1684](https://www.sophgo.com/sophon-u/product/introduce/bm1684.html) (16 TOPS INT8?)
    - [BM1684X](https://www.sophgo.com/sophon-u/product/introduce/bm1684x.html) (32 TOPS INT8; 新文档已删除此数据)
    - [BM1688](https://www.sophgo.com/sophon-u/product/introduce/bm1688.html) (16 TOPS INT8)
    - [CV186AH](https://www.sophgo.com/sophon-u/product/introduce/cv186ah.html) (7.2 TOPS INT8)
  - 非量化的 FP16/FP32 部署，测定 F1 score 和推理时长如下
    - BM1684 FP32: 0.573 / 109.8ms
    - BM1684X FP32: 0.573 / 91.37ms   (2 TFLOPS FP32)
    - BM1684X FP16:  0.558 / 41.44ms  (16 TFLOPS FP16)
    - BM1688 FP32: 0.571 / 381.5ms
    - BM1688 FP16: 0.571 / 108.97ms
    - CV186X FP32: 0.571 / 365.9ms
    - CV186X FP16: 0.571 / 101.11ms
    - In essay report: 0.629 / 330ms

⚠ 既然出题方已有一套完整的移植流程，他为什么还要出题，他做不到的事是什么

- 老板子算力高 (32 TOPS INT8)，新板子算力更低 (0.5 TOPS INT8)，对于推理速度是个挑战
  - [Milk-V Duo CV1800B](https://milkv.io/duo)
- 更新 PP-OCRv4 了，新模型 3.5M 更轻量化且号称精度指标更高
- 移植教程中提到了量化，但推理仓库中无量化实验，可能他量化失败了
