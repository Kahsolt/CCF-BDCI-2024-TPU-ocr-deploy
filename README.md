# CCF-BDCI-2024-TPU-ocr-deploy

    CCF BDCI 2024 基于TPU平台的OCR模型性能优化

----

Contest page: https://www.datafountain.cn/competitions/1044  
Team Name: 识唔识得  


### 环境搭建

⚪ 上位机

- 运行 `downloads\download.cmd` 下载文件资源材料
- 下载 paddle 模型文件并转 onnx
  - `pip install -r requirements.txt`
  - run `models\download.cmd`
- 运行 `run_docker.cmd` 以启动 docker 环境 (下列命令在在 Docker 中进行)
  - `bash ./convert_cvimodel.sh det`
  - `bash ./convert_cvimodel.sh rec`
  - `bash ./convert_cvimodel.sh cls mb`

⚪ 板子

- (TODO) 用 cviruntime 运行框架进行 cvimodel 的部署+运行


#### references

- https://github.com/Kahsolt/MilkV-Duo-init
- https://github.com/sophgo/cviruntime

----
by Armit
2024/09/14 
