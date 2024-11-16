# CCF-BDCI-2024-TPU-ocr-deploy

    CCF BDCI 2024 基于TPU平台的OCR模型性能优化

----

Contest page: https://www.datafountain.cn/competitions/1044  
Team Name: 识唔识得  


ℹ 本仓库部署 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 项目到 [MilkV-Duo](https://milkv.io/zh/duo) 板上运行

### 性能评估 & 对比

⚪ A榜 (ICDAR2019-LVST, `n_sample=2350`)

ℹ 单元格内数值为 `f-score/precsion/recall : valid_infer_time/contest_infer_time : real_fps`

| det | rec | CPU + onnx (fp32) | TPU + cvimodel (int8 + bf16) | valid score | contest score | comment |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| v4 | v4 | 0.60724/0.78855/0.49372 : 76.05 | | | | not run on chip |
| v3 | v3 | 0.57585/0.80885/0.44707 : 58.68 | | | | not run on chip |
| v2 | v2 | 0.52051/0.78323/0.38977 : 43.02 | 0.44099/0.65593/0.33215 : 719.539/333.937 : 0.88 | 46.47884 | 79.25498 | too slow |
| v3 | mb | 0.54064/0.79092/0.41069 : 58.13 | 0.42010/0.54821/0.34052 : 433.201/277.942 : 1.22 | 69.98178 | 83.17896 | slow |
| v2 | mb | 0.49098/0.78041/0.35815 : 41.59 | 0.42781/0.63849/0.32166 : 367.944/256.211 : 1.42 | 75.83703 | 85.33433 (⭐) | the most balanced solution |
| mb | mb | 0.34883/0.69257/0.23312 : 41.61 | 0.32475/0.57048/0.22698 : 344.309/256.930 : 1.47 | 73.72358 | 81.15095 | too wrong |

```
注: 
- valid score 计算中: infer_time = ts_det_infer + ts_rec_infer * (n_crop / n_img), 似乎应用上更合理
- contest score 计算中: infer_time = ts_det_infer + ts_rec_infer, 应比赛要求
- real_fps = n_img / (ts_total - (ts_model_load + ts_model_unload)) * 1000
```

一些(确实很!!)难蚌的比赛刷分设置，调整精度-时间平衡：

| input size | f1 | infer_time | real_fps | score | comment |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 640 | 0.42781 | 256.211 | 1.42  | 85.33433 | v2-mb baseline |
| 480 | 0.33901 | 155.279 | 1.885 | 90.36170 | 综合考虑最优 ⭐ |
| 320 | 0.20613 |  75.951 | 2.954 | 91.78934 | 很快，但质量下降很厉害 |

⚪ B榜 (MSRA-TD500, `n_sample=500`)

| input size | infer_time | real_fps | comment |
| :-: | :-: | :-: | :-: |
| 640 | 313.969 | 0.42 | 原图较大，ts_det_infer 比 A 榜多 |
| 480 | 198.213 | 0.52 | 相比 640 质量下降不大 (⭐) |
| 320 | 115.033 | 0.58 | 相比 480 有文本框粘连/更多的漏检；原图较大，load_img 严重拉低了 real_fps |

⚪ B2榜 (unknown, `n_sample=3992 (resampled under 640x640)`)

| input size | infer_time | real_fps | comment |
| :-: | :-: | :-: | :-: |
| 640 | 255.359 | 1.96 | 提前降采样后少了很多mem swap，总体吞吐量提高 |
| 480 | 152.944 | 2.70 |  |


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
- https://github.com/xinke-wang/OCRDatasets
  - MSRA-TD500 dataset: http://pages.ucsd.edu/%7Eztu/publication/MSRA-TD500.zip
  - MSRA-TD500 essay: https://pages.ucsd.edu/~ztu/publication/cvpr12_textdetection.pdf

----
by Armit
2024/09/14 
