# CCF-BDCI-2024-TPU-ocr-deploy

    CCF BDCI 2024 基于TPU平台的OCR模型性能优化

----

Contest page: https://www.datafountain.cn/competitions/1044  
Team Name: 识唔识得  


### 环境搭建

- 上位机: 用 tpu_mlir 编译工具箱进行 cvimodel 的仿真+编译
- 板子: 用 sophon.sail 运行框架进行 cvimodel 的部署+运行

⚪ 上位机

- 运行 `downloads\download.cmd` 下载文件资源材料
- 运行 `run_docker.cmd` 以启动 docker 环境 (**此后一切操作都在 Docker 中进行**)
  - `pip install -r requirements_tpuc_dev.txt`
  - ``
  - 该工具箱的安装路径为 `/usr/local/lib/python3.10/dist-packages/tpu_mlir` 以后可能会用到
  - 可执行脚本在其 `$TPU_MLIR_ROOT/bin` 和 `$TPU_MLIR_ROOT/python/tools` 目录下

⚪ 板子

- 暂时不知道阿
- 运行时框架 sophon.sail??!

----
by Armit
2024/09/14 
