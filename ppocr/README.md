# ppocr demo project

    本目录为赛方给出的样例工程

----

### 开发流程模型

- 上位机: 用 tpu_mlir 编译工具箱进行 cvimodel 的仿真+编译
- 板子: 用 sophon.sail 运行框架进行 cvimodel 的部署+运行


### 环境搭建

ℹ 安装流程只进行一次

#### 数据准备

- 运行 `downloads\download.cmd` 下载文件资源材料
- 从 `ocr-595521.zip` 中解压出单个文件 `tpu_mlir-1.9b0-py3-none-any.whl` 备用
- 解压 `datasets-101982.zip` 到目录 `datasets\`

#### 编译工具链 tpu-mlir (上位机)

⚪ Windows (暂时不知道是否 work 🤔)

```shell
conda create -y -n tpu python==3.10
conda activate tpu
pip install -r requirements.txt
pip install .\downloads\tpu_mlir-1.9b0-py3-none-any.whl
```

- 该工具箱的安装路径为 `{MINICONDA_ROOT}\envs\tpu\lib\site-packages\tpu_mlir` 以后可能会用到
- 可执行脚本在其 `/bin` 和 `/python/tools` 目录下

⚪ Docker

ℹ 运行 `run_docker.cmd` 以启动 docker 环境

```shell
pip install -r requirements.txt
pip install ./downloads/tpu_mlir-1.9b0-py3-none-any.whl
```

- 该工具箱的安装路径为 `/usr/local/lib/python3.10/dist-packages/tpu_mlir` 以后可能会用到
- 可执行脚本在其 `/bin` 和 `/python/tools` 目录下

#### 运行时框架 sophon.sail (板子)

- 暂时不知道阿
