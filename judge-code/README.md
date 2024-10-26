原评测代码由 sophgo 提供，使用 sophon-sail 运行时在 BM1684/CV186x 系列板子上跑 bmodel
  - ref: https://github.com/sophgo/sophon-demo/tree/release/sample/PP-OCR/python  

⚠ 但 CV1800 硬件根本不可能以 sophon-sail 为运行时跑 cvimodel，出题方把代码改造了一半发现跑不起来，就出作赛题让大家解决！

我们把这套代码改造为以 onnx 为后端，来方便把它的前后处理部分迁移为 C++ 代码 (跑 cvimodel 需要的 cviruntime 是 C++)


评测结果 (`eval_score.py`):

```
[ppocrv4 (no cls)]
  Inference time: 76.05
  F-score: 0.60724, Precision: 0.78855, Recall: 0.49372
[ppocrv3]
  Inference time: 62.15
  F-score: 0.57585, Precision: 0.80885, Recall: 0.44707
[ppocrv3 (no cls)]
  Inference time: 58.68
  F-score: 0.57585, Precision: 0.80885, Recall: 0.44707
[ppocrv2 (no cls)]
  Inference time: 54.55
  F-score: 0.08174, Precision: 0.37194, Recall: 0.04591
[ppocr_mb (no cls)]
  Inference time: 48.60
  F-score: 0.03312, Precision: 0.40547, Recall: 0.01726
```
