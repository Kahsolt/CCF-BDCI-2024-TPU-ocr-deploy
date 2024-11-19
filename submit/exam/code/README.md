#### run_ppocr_sys / ppocr_sys_many.cpp

> 用于运行 cvimodel 推理的启动器 (在 SoC 上运行！！)

⚠ 注意: 该C++项目的编译依赖 cviruntime 环境，复现请参考 https://github.com/Kahsolt/tpu-sdk-cv180x-ppocr/tree/master/samples/ppocr_sys_many; 本目录中的cpp代码仅作展示使用，请不要在这里尝试编译(

CVI启动器 `run_ppocr_sys` 的命令行用法：

```shell
export LD_LIBRARY_PATH=/path/to/project/lib:$LD_LIBRARY_PATH
run_ppocr_sys <ppocr_det.cvimodel> <ppocr_rec.cvimodel> <image_folder>
```

预测结果将固定输出到**当前工作目录**下的裸预测结果文件：

```shell
$PWD/results.txt
```

需要将该结果下载到 Host 做进一步后处理

---- 

#### run_utils.py

> 一组前后处理/可视化小工具 (在 Host 上运行！！)

下列为命令用法示意，按需修改为正确的路径：

```shell
# 数据集降采样 (test_images => test_images_640x640)
## -I 原始数据集目录 -O 降采样数据集目录
python run_utils.py --resize -I ./test_images -O ./test_images_640x640

# 预测文件格式转换 (results.txt => results_lowres.json)
## -I 裸结果文件 -O 降采样结果文件/最终结果文件 -R 字典文件
python run_utils.py --cvtres -I ./results.txt -O ./results_lowres.json -R ./ppocr_keys_v1.txt

# 对于降采样过的数据集，还原原图尺寸的坐标位置 (results_lowres.json -> results.json)
## -I 降采样结果文件 -O 最终结果文件 -R 原始数据集目录
python run_utils.py --fixres -I ./results_lowres.json -O ./results.json -R ./test_images

# 预测结果可视化 (results*.json => results/*.png)
## -I 降采样数据集目录 -O 可视化图输出目录 -R 降采样结果文件
python run_utils.py --visres -I ./results_lowres.json -O ./results_lowres/ -R ./test_images_640x640
## -I 原始数据集目录 -O 可视化图输出目录 -R 最终结果文件
python run_utils.py --visres -I ./results.json -O ./results/ -R ./test_images
```

----
by Armit
2024年11月19日
