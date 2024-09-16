# CCF-BDCI-2024-TPU-ocr-deploy

    CCF BDCI 2024 基于TPU平台的OCR模型性能优化

----

Contest page: https://www.datafountain.cn/competitions/1044  
Team Name: 识唔识得  


### setup dev env

我们的首要尝试是迁移部署 PP-OCRv4 系列模型

```shell
# venv
conda create -y -n tpu python==3.11
conda activate tpu
# test data
mkdir data & pushd data
wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.1/ppocr_img.zip
unzip ppocr_img.zip
popd
# (optional) packages, run PPOCR
pip install paddlepaddle
pip install paddleocr onnxruntime
# (optional) packages, run CnOCR (NOT recommended, it depeneds on PyTorch and not support verticle texts!!)
pip install cnocr[ort-cpu]
```


#### refenrence

- ICDAR2019-LVST dataset: https://rrc.cvc.uab.es/?ch=16&com=introduction
  - download: https://aistudio.baidu.com/datasetdetail/177210
- PaddleOCR test: https://paddleocr.bj.bcebos.com/dygraph_v2.1/ppocr_img.zip
- tpu-mlir 资料
  - 算能云开发平台使用说明 https://tpumlir.org/index.html
  - PP-OCR 模型部署参考示例
    - 移植 https://docs.qq.com/pdf/DSUlabGVFRlBkQkZv
      - 将 ch_PP-OCRv3_xx 系列模型迁移至 BM1684/BM1684X/BM1688/CV186X
      - FP16/FP32 部署，F1 score约
    - 推理 https://github.com/sophgo/sophon-demo/tree/release/sample/PP-OCR
  - Duo系列开发板
    - site: https://milkv.io/duo
    - doc: https://milkv.io/docs/duo/overview
- TPU-sr-deploy: https://github.com/Kahsolt/CCF-BDCI-2023-TPU-sr-deploy

----
by Armit
2024/09/14 
