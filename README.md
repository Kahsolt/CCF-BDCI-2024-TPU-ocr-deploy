# CCF-BDCI-2024-TPU-ocr-deploy

    CCF BDCI 2024 基于TPU平台的OCR模型性能优化

----

Contest page: https://www.datafountain.cn/competitions/1044  
Team Name: 识唔识得  


ℹ 本仓库部署 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 项目到 [MilkV-Duo](https://milkv.io/zh/duo) 板上运行

### 性能评估 & 对比

ℹ 单元格内数值为 `f-score/precsion/recall : infer_time`

| det | rec | CPU + onnx (fp32) | TPU + cvimodel (int8 + bf16) | score | comment |
| :-: | :-: | :-: | :-: | :-: | :-: |
| v4 | v4 | 0.60724/0.78855/0.49372 : 76.05 | | | not run on chip |
| v3 | v3 | 0.57585/0.80885/0.44707 : 58.68 | | | not run on chip |
| v2 | v2 | 0.52051/0.78323/0.38977 : 43.02 | 0.44099/0.65593/0.33215 : 719.539 | 46.47884 | too slow |
| v3 | mb | 0.54064/0.79092/0.41069 : 58.13 | 0.42010/0.54821/0.34052 : 433.201 | 69.98178 | slow |
| v2 | mb | 0.49098/0.78041/0.35815 : 41.59 | 0.42781/0.63849/0.32166 : 367.944 | 75.83703 (⭐) | the most balanced solution |
| mb | mb | 0.34883/0.69257/0.23312 : 41.61 | 0.32475/0.57048/0.22698 : 344.309 | 73.72358 | too wrong |


### 环境搭建

⚪ 资源获取 (run on Windows)

- `downloads\download.cmd`
- `git clone https://github.com/Kahsolt/tpu-sdk-cv180x-ocr`

⚪ 上位机 (模型编译, 本仓库!)

ℹ 可跳过，直接使用我预编译的模型 [tpu-sdk-cv180x-ocr/cvimodels](./tpu-sdk-cv180x-ocr/cvimodels/)  

- 下载并转换模型: paddle -> onnx (run on Windows)
  - `pip install -r requirements.txt`
  - run `models\download_and_convert.cmd`
- 编译模型文件: onnx -> cvimodel (run in Docker container [tpu-mlir](./run_docker.cmd))
  - `bash ./compile_cvimodel_all.sh`

⚪ 上位机 (运行时编译, 子仓库 tpu-sdk-cv180x-ocr)

ℹ 可跳过，直接使用我预编译的运行时 [tpu-sdk-cv180x-ocr/samples/ppocr_*](./tpu-sdk-cv180x-ocr/samples/)

- 参考各子项目的说明文件 `tpu-sdk-cv180x-ocr/samples/ppocr_*/README.md`


#### references

- https://github.com/Kahsolt/MilkV-Duo-init
- https://github.com/Kahsolt/tpu-sdk-cv180x-ocr
- https://community.milkv.io/t/duo-linux-fdisk-resize2fs-root/42
  - 板上TF卡分区扩容，参考，记得分区不能太大！（1.5G 安全）
- https://github.com/ZhangGe6/onnx-modifier
- https://github.com/zcswdt/OCR_ICDAR_label_revise

----
by Armit
2024/09/14 
