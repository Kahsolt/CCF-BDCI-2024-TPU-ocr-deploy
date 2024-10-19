# CCF-BDCI-2024-TPU-ocr-deploy

    CCF BDCI 2024 基于TPU平台的OCR模型性能优化

----

Contest page: https://www.datafountain.cn/competitions/1044  
Team Name: 识唔识得  


### 环境搭建

⚪ 资源获取

- 运行 `downloads\download.cmd` 下载资源材料文件

⚪ 上位机

ℹ 运行 `run_docker.cmd` 以启动 docker 环境，下列命令在 Docker 中进行

- 下载并转换模型: paddle -> onnx (可跳过，直接使用我预编译的模型 [cvimodel](./cvimodel/))
  - `pip install -r requirements.txt`
  - read and run `models\download_and_convert.cmd` line by line
- 编译模型文件: onnx -> cvimodel (可跳过，直接使用我预编译的模型 [cvimodel](./cvimodel/))
  - `bash ./convert_cvimodel.sh det v3`
  - `bash ./convert_cvimodel.sh rec v3`
  - `bash ./convert_cvimodel.sh cls mb`
- 交叉编译板上模型的运行时 sophon-sail (可跳过，直接使用我预编译的运行时库 [runtime](./runtime/))
  - read and run [compile_runtime.sh](./compile_runtime.sh) line by line

⚪ 板子

- 用 cviruntime 运行框架进行 cvimodel 的部署+运行
- 参考子项目 https://github.com/Kahsolt/tpu-sdk-cv180x-ocr


#### references

- https://github.com/Kahsolt/MilkV-Duo-init
- https://github.com/Kahsolt/tpu-sdk-cv180x-ocr
- https://github.com/sophgo/sophon-sail
- https://github.com/sophgo/cviruntime
- https://community.milkv.io/t/duo-linux-fdisk-resize2fs-root/42
  - 板上TF卡分区扩容，参考，记得分区不能太大！（1.5G 安全）
- https://github.com/ZhangGe6/onnx-modifier

----
by Armit
2024/09/14 
