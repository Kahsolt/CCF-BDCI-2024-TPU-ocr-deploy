### TPU-ocr-deploy (submit version)

    完整仓库地址: https://github.com/Kahsolt/CCF-BDCI-2024-TPU-ocr-deploy，封榜后推送更新 :)

----

#### 推理运行指南

主机运行：

```shell
# 创建虚拟环境 (可选)
conda create -n tpu python==3.10.0
conda activate tpu

# 安装第三方包依赖
pip install -r requirements.txt

# 本目录整体上传到板子
ssh root@192.168.42.1 "mkdir -p /root/submit"
scp -r ./* root@192.168.42.1:/root/submit
```

板上运行：

```shell
# 切入工作目录
cd /root/submit

# 配置动态链接库路径
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH

# 运行推理，注意 det 模型有不同的尺寸：
#   - 640: 更高的 F1  (f1 ~= 0.43)
#   - 480: 均衡的设置 (推荐 ⭐)
#   - 320: 更快的速度 (fps ~= 3)
# 命令参数：
#   ./run_ppocr_sys <det_model> <rec_model> <img_folder>
./run_ppocr_sys ./cvimodels/ppocrv2_det_int8_640.cvimodel ./cvimodels/ppocr_mb_rec_bf16.cvimodel ./test_img_dir
./run_ppocr_sys ./cvimodels/ppocrv2_det_int8_480.cvimodel ./cvimodels/ppocr_mb_rec_bf16.cvimodel ./test_img_dir
./run_ppocr_sys ./cvimodels/ppocrv2_det_int8_320.cvimodel ./cvimodels/ppocr_mb_rec_bf16.cvimodel ./test_img_dir

# 从上调命令的运行结果读取 infer_time，此处须注意！！
# 按照比赛代码定义 infer_time = ts_det_infer + ts_rec_infer (一次det + 一次rec)
# 而非直接读取 ts_infer 字段 (一次det + 多次rec)

# 解析推理结果 (产生 results.json 文件)
python run_cvt_results.py -F ./results.txt
```

主机运行：

```shell
# 下载推理结果
mkdir -p results
scp root@192.168.42.1:/root/submit/results.json ./results

# 可视化结果推理结果
python run_vis_results.py -B ./results/results.json -O ./ocr_result_dir
```


#### 参考推理结果

```
infer_time = ts_det_infer + ts_rec_infer
real_fps = n_img / (ts_total - (ts_model_load + ts_model_unload)) * 1000
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


#### 我想要自己编译CVI模型和运行时

⚠ 如果这部分代码还不存在、不是最新的，或者不完全work，请联系我们；比赛结束后的合理时间，这些会全部开源发布

- CVI模型编译: 参考 https://github.com/Kahsolt/CCF-BDCI-2024-TPU-ocr-deploy/blob/master/compile_cvimodel.sh
- 运行时编译: 参考 https://github.com/Kahsolt/tpu-sdk-cv180x-ppocr/tree/master/samples/ppocr_sys_many

----
by Armit
2024年11月13日
