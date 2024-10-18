### 开发流程

⚠ 本次比赛主要任务: 调研轻量化模型, 模型训练(可跳过), 模型量化部署, 推理测试  

⚪ 第一阶段: 信息调研

- 数据集探索
  - 数据统计: 评测数据的尺寸
- 开源可用 OCR 模型
  - 找一个精度最好的预训练模型，直接推理并提交A榜的测试数据，试探评测数据的难度
  - 找一组(有预训练权重的)轻量化模型，直接推理并提交A榜的测试数据，试探未来TPU部署时的大致得分
- 申请领取板子，调研硬件 Milk-V Duo CV1800B 的计算性能
  - 整理官方文档给出的性能数据
  - 做基准测试，寻找TPU计算瓶颈

⚪ 第二阶段: 模型部署

- 将轻量化模型部署到 TPU，测推理速度
  - 如果没有预训练权重也没关系，直接用随机权重的模型测速度也行
- 评分公式 `score = 40 * f1_score - 0.085 * infer_time` 其中 `infer_time < 1000`
  - 最低 `-65`, 最高 `40`

⚪ 第三阶段: 模型微调

⚠ 在第二阶段中确定了模型结构可用、速度可以接受后，再研究如何微调权重以优化精度

- 使用给出的数据集微调模型，并重复模型部署步骤
- 优化模型的计算量


### 任务列表

- [*] 跑通各开源项目的推理 [run_repo_infer.cmd](./run_repo_infer.cmd)
- [ ] 跑通官方样例工程 [ppocr](/ppocr) (这个需要在板子上跑😈)
- [ ] 部署 & 性能测试官方样例工程
- [ ] 板子的基准性能测试
- [ ] 尝试迁移其他开源模型
  - [ ] ppocr v4
  - [ ] ppocr v3
  - [ ] ppocr v2 (good!)
  - [ ] chineseocr_lite (good!)


### 模型选型

⚪ 调研索引

- OCR调研报告: https://blog.csdn.net/weixin_41021342/article/details/127203654
- 十二款开源OCR开箱测评: http://www.ceietn.com/a/zhengcefagui/guojiaji/930.html
- 6款开源中文OCR使用介绍: https://blog.csdn.net/bugang4663/article/details/131720149
- PaddleOCR 3.5M模型: https://blog.csdn.net/moxibingdao/article/details/108765380
- 开源OCR模型对比: https://www.cnblogs.com/shiwanghualuo/p/18139459

⚪ 开源项目

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

⚪ 开源项目实测

```shell
# venv
conda create -y -n tpu python==3.10
conda activate tpu
pip install onnxruntime Shapely==2.0.6

# data
mkdir data & pushd data
wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.1/ppocr_img.zip
unzip ppocr_img.zip
# => manually download & unzip train_full_labels.json (183.86M) from https://aistudio.baidu.com/datasetdetail/177210
popd

# project
# (optional) packages, run PPOCR
pip install paddlepaddle paddleocr
# (optional) packages, run RapidOCR (NOT recommended, even twice slower than PPOCR!)
pip install rapidocr_onnxruntime
# (optional) packages, run CnOCR (NOT recommended, it depeneds on PyTorch and not support verticle texts!!)
pip install cnocr[ort-cpu]
# (optional) repository, run ChineseOCR-Lite
# => manually run repo/init_repos.cmd
```


### refenrence

- ICDAR2019-LVST dataset: https://rrc.cvc.uab.es/?ch=16&com=introduction
  - download: https://aistudio.baidu.com/datasetdetail/177210
  - ⚠ 该压缩包已损坏，建议使用赛方提供的子集，详见 `ppocr\downloads\download.cmd`
- PaddleOCR test: https://paddleocr.bj.bcebos.com/dygraph_v2.1/ppocr_img.zip
- tpu-mlir 资料
  - 算能云开发平台使用说明 https://tpumlir.org/index.html
  - Duo系列开发板
    - site: https://milkv.io/duo
    - doc: https://milkv.io/docs/duo/overview
- TPU-sr-deploy: https://github.com/Kahsolt/CCF-BDCI-2023-TPU-sr-deploy


### 前车之鉴

- 移植 https://docs.qq.com/pdf/DSUlabGVFRlBkQkZv
  - 将 ch_PP-OCRv3_xx 系列模型迁移至 CV181X
  - det(INT8) / rec(BF16) 部署
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

- 算能给出的[移植教程](https://docs.qq.com/pdf/DSUlabGVFRlBkQkZv) 实现了 `CV181x` 部署，但 MilkV-Duo 是 `CV180x` 型号
  - [Milk-V Duo CV1800B](https://milkv.io/duo) == [SOPHGO CV180xB](https://en.sophgo.com/sophon-u/product/introduce/cv180xb.html)
- 老板子算力高 (32 TOPS INT8)，新板子算力更低 (0.5 TOPS INT8)，对于推理速度是个挑战
- 更新 PP-OCRv4 了，新模型 3.5M 更轻量化且号称精度指标更高
- 移植教程中提到了 INT8 量化，但推理仓库中相应实验，可能他量化失败了

----
by Armit
2024/09/14 
