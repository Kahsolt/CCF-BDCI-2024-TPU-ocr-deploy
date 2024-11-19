### TPU-ocr-deploy (submit version)

    完整仓库地址: https://github.com/Kahsolt/CCF-BDCI-2024-TPU-ocr-deploy，封榜后推送更新 :)

----

队名: 识唔识得


#### 解决方案

⚪ 方案概述

本项目将预训练的文本识别模型 [Paddle-OCR](https://github.com/PaddlePaddle/PaddleOCR) 通过量化技术转换，部署运行在附有 TPU 硬件的 [Milk-V Duo](https://milkv.io/duo) 开发板上。  
我们将工程重点放在模型量化策略、计算性能优化以及 TPU 的工程部署上，确保在保持较高识别精度的同时，提高处理速度和资源利用效率。  

⚪ 方案实现步骤

- 预研
  - MilkV Duo 软硬件环境和性能基准评估
  - 模型选型，确定迁移 paddle-ocr 和 chinese_ocr_lite
  - 测定 CPU+onnx 基线
- 开发-评估 迭代
  - 模型转换、量化、部署 cvimodel
  - 开发板上CVI模型运行时 cvirunner
  - 测定 TPU+cvimodel 基线
  - 迭代优化模型 cvimodel 和运行时 cvirunner
- 推理比赛数据集
  - 编写实验手册

⚪ 难点

- 板上运行时开发框架的最终确立因相关资料杂乱、匮乏而颇费周折
- 板上计算资源紧张，无法做到很高的召回率，否则严重影响实时性
- 基于 CRNN 的识别模型难以用 INT8 量化，手工设计的混精度量化依然收益不明显

⚪ 创新点

- 用 C++ 近似实现检测模型后处理中的 unclip 操作，详见 [unclip_algo.pdf](user_data/files/unclip_algo.pdf)
- 移除模型中不必要的最后一层 softmax，降低计算量
- 对于检测模型，启用 `--quant_output` 以保持输出不被反量化，降低数据传输量
- 预处理识别模型的校准数据集，使其更接近真实推理时候的输入预填充情况
- 超参调优，发现进一步降低模型输入尺寸 (640->480) 可以牺牲少量 F1 分数换取更高的端到端 FPS


#### 推理运行指南 (跑通demo样例)

该小节介绍基本的运行环境搭建和配置，并跑通 3 张样例图 [exam/exam_input](exam/exam_input) 的推理，结果输出到 [output](output) 目录下

⚪ 主机运行 (环境安装)

```shell
# 创建虚拟环境 (可选)
conda create -n tpu python==3.10.0
conda activate tpu

# 安装第三方包依赖 (后处理+可视化)
pip install -r requirements.txt

# 本测试目录整体上传到板子
ssh root@192.168.42.1 "mkdir -p /root/data"
scp -r ./* root@192.168.42.1:/root/data
```

⚪ 板上运行 (推理)

```shell
# 切入工作目录
cd /root/data
# 记录常用目录
export BASE_PATH=$PWD
export BIN_PATH=$BASE_PATH/exam/code
export MODEL_PATH=$BASE_PATH/user_data/cvimodel
# 配置动态链接库路径
export LD_LIBRARY_PATH=$BASE_PATH/user_data/lib:$LD_LIBRARY_PATH
# 设置可执行文件权限
chmod +x $BIN_PATH/run_ppocr_sys

# 切入输出目录
mkdir -p output
cd output

# 运行推理，这里使用默认的模型配置 v2_det(640) + mb_rec
# ./run_ppocr_sys <det_model> <rec_model> <img_folder>
nice -n -19 $BIN_PATH/run_ppocr_sys $MODEL_PATH/ppocrv2_det_int8_640.cvimodel $MODEL_PATH/ppocr_mb_rec_bf16.cvimodel $BASE_PATH/exam/exam_input

# 从上述命令的标准输出读取推理时间 infer_time，此处须注意！！
# 按照比赛代码定义 infer_time = ts_det_infer + ts_rec_infer (一次det + 一次rec)
# 而非直接读取 ts_infer 字段 (一次det + 多次rec)

# 查看是否产生了裸推理结果
cat results.txt
```

⚪ 主机运行 (后处理)

```shell
# 下载裸推理结果
scp root@192.168.42.1:/root/data/output/results.txt ./output/results_demo.txt

# 转换文件格式
python exam/code/run_utils.py --cvtres -I ./output/results_demo.txt -O ./output/results_demo.json -R ./user_data/files/ppocr_keys_v1.txt
# 查看最终推理结果文件
cat ./output/results_demo.json

# 可视化结果推理结果 (可选)
python exam/code/run_utils.py --visres -I ./output/results_demo.json -O ./output/results_demo -R ./exam/exam_input
# 查看可视化结果 (应与 exam/exam_output 中的结果一致)
ls ./output/results_demo
```


#### 推理运行指南 (复现B榜)

⚠ B榜数据分辨率较大，其推理流程包含更多的前后处理步骤，请仔细阅读下面的手册 ;)

⚪ 主机/板子运行 (下采样)

⚠ 为什么一定要提前下采样，不可以直接处理大图吗？ 👉 看[这里](#b榜数据预处理说明)

如果完整数据集**不在板子上**，那么可以在主机处理后上传 (推荐)：

```shell
# 数据集降采样
python exam/code/run_utils.py --resize -I ./database/test_images -O ./user_data/files/test_images_640x640
# 上传到板子
scp -r ./user_data/files/test_images_640x640 root@192.168.42.1:/root/data/user_data/files/test_images_640x640
```

如果完整数据集**已经在板子上**，那么可以直接在板子上处理 (可能很慢/炸内存)：

```shell
# 数据集降采样
python $BIN_PATH/run_utils.py --resize -I $BASE_PATH/database/test_images -O $BASE_PATH/user_data/files/test_images_640x640
```

⚪ 板上运行 (推理)

```shell
# 切入工作目录
cd /root/data
# 记录常用目录
export BASE_PATH=$PWD
export BIN_PATH=$BASE_PATH/exam/code
export MODEL_PATH=$BASE_PATH/user_data/cvimodel
# 配置动态链接库路径
export LD_LIBRARY_PATH=$BASE_PATH/user_data/lib:$LD_LIBRARY_PATH
# 设置可执行文件权限
chmod +x $BIN_PATH/run_ppocr_sys

# 切入输出目录
mkdir -p output
cd output

# 运行推理，注意有很多模型配置可以使用 (见后面章节)
# ↓↓↓ 这是 640 尺寸的模型，能获得最好的 f1，也是我们 B 榜提交的版本 (⭐)
nice -n -19 $BIN_PATH/run_ppocr_sys $MODEL_PATH/ppocrv2_det_int8_640.cvimodel $MODEL_PATH/ppocr_mb_rec_bf16.cvimodel $BASE_PATH/user_data/files/test_images_640x640
# ↓↓↓ 这是 480 尺寸的模型，牺牲部分 f1 换取更快的速度，按加权评分公式计算的话预计会高5分
nice -n -19 $BIN_PATH/run_ppocr_sys $MODEL_PATH/ppocrv2_det_int8_480.cvimodel $MODEL_PATH/ppocr_mb_rec_bf16.cvimodel $BASE_PATH/user_data/files/test_images_640x640

# 从上述命令的标准输出读取推理时间 infer_time，此处须注意！！
# 按照比赛代码定义 infer_time = ts_det_infer + ts_rec_infer (一次det + 一次rec)
# 而非直接读取 ts_infer 字段 (一次det + 多次rec)

# 查看是否产生了裸推理结果
cat results.txt
```

⚪ 主机运行 (后处理)

```shell
# 下载裸推理结果
scp root@192.168.42.1:/root/data/output/results.txt ./output/results_lowres.txt

# 转换文件格式
python exam/code/run_utils.py --cvtres -I ./output/results_lowres.txt -O ./output/results_lowres.json -R ./user_data/files/ppocr_keys_v1.txt
# 下采样坐标尺度修复
python exam/code/run_utils.py --fixres -I ./output/results_lowres.json -O ./output/results.json -R ./database/test_images
# 查看最终推理结果文件
cat ./output/results.json

# 可视化结果推理结果-降采样版本 (可选)
python exam/code/run_utils.py --visres -I ./output/results_lowres.json -O ./output/results_lowres -R ./user_data/files/test_images_640x640
# 可视化结果推理结果-原图版本 (可选)
python exam/code/run_utils.py --visres -I ./output/results.json -O ./output/results -R ./database/test_images
```


#### 参考推理结果

注意下列定义，相应数据见 `run_ppocr_sys` 的标准输出：

```
TPU推理时间 infer_time := ts_det_infer + ts_rec_infer
端到端帧率 real_fps := n_img / (ts_total - (ts_model_load + ts_model_unload)) * 1000
```

⚪ A榜 (ICDAR2019-LVST, `n_sample=2350`)

| input size | f1 | infer_time | real_fps | score | comment |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 640 | 0.42781 | 256.211 | 1.42  | 85.33433 | v2-mb baseline |
| 480 | 0.33901 | 155.279 | 1.885 | 90.36170 | 综合考虑最优 ⭐ |
| 320 | 0.20613 |  75.951 | 2.954 | 91.78934 | 很快，但质量下降很厉害 |

⚪ B榜 (MSRA-TD500, `n_sample=500`)

| input size | infer_time | real_fps | comment |
| :-: | :-: | :-: | :-: |
| 640 | 313.969 | 0.42 | 原图较大，ts_det_infer 比 A 榜长 |
| 480 | 198.213 | 0.52 | 综合考虑最优 ⭐ |
| 320 | 115.033 | 0.58 | 有文本框粘连/更多的漏检；原图较大，load_img 严重拉低了 real_fps |

⚪ B2榜 (unknown, `n_sample=3992 (resampled under 640x640)`)

| input size | infer_time | real_fps | comment |
| :-: | :-: | :-: | :-: |
| 640 | 255.359 | 1.96 | 提前降采样避免 Mem Swap，总体吞吐量提高 (提交版本！) |
| 480 | 152.944 | 2.70 | 更快，不知 f1 下降多少 |


#### 模型支持情况

ℹ 所使用的 PPOCR 开源模型权重直接来源于官网的发布版本: https://paddlepaddle.github.io/PaddleOCR/main/model/index.html

可以任意组合下列受支持的 det + rec 模型，我们的实验主要用 `v2_det(640/480) + mb_rec` 这个设置  
在无 Mem Swap 的理想情况下，各模型的推理时间几乎**仅与输入数据尺寸相关**，可以认为直接等于下表所测定的常数 :)  

⚪ det

| version | dtype | shape | infer_time | comment |
| :-: | :-: | :-: | :-: | :-: |
| v3 | int8 | 640x640 | 270.805 | 慢，分割质量不好 |
| v2 | int8 | 640x640 | 247.386 | 原ppocr默认设置 |
| v2 | int8 | 480x480 | 128.963 | 质量相比640下降不多，完全可接受 |
| v2 | int8 | 320x320 |  46.317 | 质量不好 |

⚪ rec

| version | dtype | shape | infer_time | comment |
| :-: | :-: | :-: | :-: | :-: |
| v3 | bf16 | 32x320 | 95.523 | 很慢 |
| v3 | mix  | 32x320 | 65.314 | 慢，错字率更低 |
| mb | bf16 | 32x320 | 33.612 | 快，但有错字 |


#### 我想要自己编译CVI模型和运行时

⚠ 如果这部分代码还不存在、不是最新的，或者不完全work，请联系我们 ;)

- CVI模型编译: 参考 https://github.com/Kahsolt/CCF-BDCI-2024-TPU-ocr-deploy/blob/master/compile_cvimodel.sh
- 运行时编译: 参考 https://github.com/Kahsolt/tpu-sdk-cv180x-ppocr/tree/master/samples/ppocr_sys_many


#### Folder Layout

```
data
├── README.md                       // 说明文件
├── database
│   ├── data.jsonl                  // 标签GT (仅评测机持有)
│   └── test_images                 // 测试图片数据集
│       └── *.jpg
├── exam
│   ├── code                        // 可执行代码
│   └── exam_input                  // 样例输入文件
│   └── exam_output                 // 样例输出可视化图
├── output                          // 复现输出结果
│   ├── results.txt                 // 裸结果
│   ├── results_lowres.json         // 中间结果
│   ├── results.json                // 最终结果
│   └── results                     // 可视化图
└── user_data
    ├── lib/                        // 运行时动态链接库
    ├── cvimodel                    // 模型文件
    │   └── *.cvimodel
    └── files/                      // 中间文件、字典、外部数据等
        ├── test_images_640x640     // 降采样版本数据集
        └── ppocr_keys_v1.txt       // 字典
```

----

#### B榜数据预处理说明

Q: 为什么一定要提前下采样，不可以直接处理大图吗？
A: 我们的推理启动器 `run_ppocr_sys` 毫无疑问是支持直接吃大图的，但 MilkV-Duo 的硬件性能大家都知道……我们不建议这样做的原因有很多：

- TF卡: 我们的 Duo 板子做不到挂载一个超过 `1.5GB` 的分区，系统会直接挂掉；而 B 榜数据集原始大小为 `3.91GB` 无法直接上传，只有下采样后不到 `800MB` 是可行的
- Mem Swap: 高分辨率图像在板上的性能瓶颈完全在 数据加载 + JPEG Decode，配合 CVI 模型的内存占用情况之后会出现很严重的 Mem Swap，这就是三倍伟大的魔鬼之处：磨损 TF 寿命，干扰计时测定，有概率内存炸了 SegFault

Q: 那如果我就是就是就是想要直接跑原图分辨率尺寸呢？
A: 把命令行里的图片文件夹路径改为原始数据集如 `$BASE_PATH/database/test_images` 就可以了，但如果撞上 SegFault 请自求多福 (

----
by Armit
2024年11月13日
