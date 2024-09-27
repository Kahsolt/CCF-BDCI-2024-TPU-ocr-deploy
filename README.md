# CCF-BDCI-2024-TPU-ocr-deploy

    CCF BDCI 2024 基于TPU平台的OCR模型性能优化

----

Contest page: https://www.datafountain.cn/competitions/1044  
Team Name: 识唔识得  

Todo List:

- [*] 跑通各开源项目的推理 [run_repo_infer.cmd](./run_repo_infer.cmd)
- [ ] 跑通官方样例工程 [ppocr](/ppocr) (这个需要在板子上跑😈)
- [ ] 部署 & 性能测试官方样例工程
- [ ] 板子的基准性能测试
- [ ] 尝试迁移其他开源模型
  - [ ] ppocr v4
  - [ ] ppocr v3
  - [ ] ppocr v2 (good!)
  - [ ] chineseocr_lite (good!)


### Quickstart

⚪ install

```shell
# venv
conda create -y -n tpu python==3.10
conda activate tpu
pip install onnxruntime

# data
mkdir data & pushd data
wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.1/ppocr_img.zip
unzip ppocr_img.zip
# => manually download & unzip train_full_labels.json (183.86M) from https://aistudio.baidu.com/datasetdetail/177210
# => manually follow ppocr\README.md 数据准备 section
popd
```

⚪ run

- 跑开源仓库: 参考 [NOTE.md](./NOTE.md) 中的 **开源项目实测**
- 跑比赛样例工程: 参考 [ppocr/README.md](./ppocr/README.md)


#### refenrence

- ICDAR2019-LVST dataset: https://rrc.cvc.uab.es/?ch=16&com=introduction
  - download: https://aistudio.baidu.com/datasetdetail/177210
  - ⚠ 该压缩包已损坏，建议使用赛方提供的子集，详见 `ppocr\downloads\download.cmd`
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
