原评测代码由 sophgo 提供，使用 sophon-sail 运行时在 BM1684/CV186x 系列板子上跑 bmodel
  - ref: https://github.com/sophgo/sophon-demo/tree/release/sample/PP-OCR/python  

⚠ 但 CV1800 硬件根本不可能以 sophon-sail 为运行时跑 cvimodel，出题方把代码改造了一半发现跑不起来，就出作赛题让大家解决！

我们把这套代码改造为以 onnx 为后端，来方便把它的前后处理部分迁移为 C++ 代码 (跑 cvimodel 需要的 cviruntime 是 C++)


⚪ 模型评测结果 (CPU + onnx)

```
[ppocrv4]
  Inference time: 76.05
  F-score: 0.60724, Precision: 0.78855, Recall: 0.49372
[ppocrv3]
  Inference time: 58.68
  F-score: 0.57585, Precision: 0.80885, Recall: 0.44707
  F-score: 0.57360, Precision: 0.80604, Recall: 0.44522
[ppocrv2]
  Inference time: 43.02
  F-score: 0.52051, Precision: 0.78323, Recall: 0.38977
[ppocr_mb]
  Inference time: 41.61
  F-score: 0.34883, Precision: 0.69257, Recall: 0.23312

[ppocr v3_det + mb_rec]
  Inference time: 58.13
  F-score: 0.54064, Precision: 0.79092, Recall: 0.41069
[ppocr v2_det + mb_rec]
  Inference time: 41.59
  F-score: 0.49098, Precision: 0.78041, Recall: 0.35815
```
