#!/usr/bin/env bash

# 在板子上跑完整数据集！

#=====================================================================
# Step 1: 新增分区 (run on chip)

# add new partition
fdisk /dev/mmcblk0
p           # print partition
n           # new partition
<Enter>     # partition type (default: primary)
<Enter>     # partition number (default: 3)
<Enter>     # First sector (use default)
+1.5G       # Last sector
w           # save and exit

# should see mmcblk0p3
lsblk
fdisk -l
ll /dev/mmcblk0p3

# format
mkfs.ext4 /dev/mmcblk0p3

# mount at /data
mkdir -p /data
mount /dev/mmcblk0p3 /data

# set auto mount
echo "/dev/mmcblk0p3 /data ext4 defaults 0 0" >> /etc/fstab


#=====================================================================
# Step 2: 上传数据/运行时

# run on host (Windows)
scp -r .\datasets\train_full_images_0 root@192.168.42.1:/data
scp -r .\datasets\MSRA_Photo root@192.168.42.1:/data
scp -r tpu-sdk-cv180x-ocr root@192.168.42.1:/root


#=====================================================================
# Step 3: 跑推理鸭！

# run on chip
cd tpu-sdk-cv180x-ocr
source ./envs_tpu_sdk.sh
cd samples

# rank-A
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv3_det_int8.cvimodel     ../cvimodels/ppocr_mb_rec_bf16.cvimodel     /data/train_full_images_0
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel     ../cvimodels/ppocr_mb_rec_bf16.cvimodel     /data/train_full_images_0
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocr_mb_det_int8.cvimodel    ../cvimodels/ppocr_mb_rec_bf16.cvimodel     /data/train_full_images_0
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel     ../cvimodels/ppocrv2_rec_bf16.cvimodel      /data/train_full_images_0
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel     ../cvimodels/ppocr_mb_rec_mix_fine.cvimodel /data/train_full_images_0
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8_480.cvimodel ../cvimodels/ppocr_mb_rec_bf16.cvimodel     /data/train_full_images_0
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8_320.cvimodel ../cvimodels/ppocr_mb_rec_bf16.cvimodel     /data/train_full_images_0

# rank-B
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel     ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/MSRA_Photo
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8_480.cvimodel ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/MSRA_Photo
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8_320.cvimodel ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/MSRA_Photo

# run on host (Windows)
pushd results
scp root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/samples/results.txt .\results.txt
python ..\tpu-sdk-cv180x-ocr\samples\ppocr_sys_many\convert_results.py -F .\results.txt
python ..\judge-code\eval_score.py ^
  --gt_path ..\datasets\train_full_images.json ^
  --result_json ..\results\results.json ^
  --inference_time 1000
popd


#=====================================================================
# Step 3.5 跑 B2 榜数据需要一些额外的处理！！

# run on host (Windows)
python run_utils_B2.py --resize -I .\datasets\test_images -R .\datasets\test_images_640x640
scp -r .\datasets\test_images_640x640 root@192.168.42.1:/data

# run on chip
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel     ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/test_images_640x640
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8_480.cvimodel ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/test_images_640x640

# run on host (Windows)
pushd results
scp root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/samples/results.txt .\resB2.txt
python ..\tpu-sdk-cv180x-ocr\samples\ppocr_sys_many\convert_results.py -F .\resB2.txt
python run_utils_B2.py --fixres -r .\results\resB2.json
ls .\results\resB2_fix.json
popd

# 基于降采样图
python vis_results_B.py -I .\datasets\test_images_640x640 -B .\results\resB2.json
# 基于原图
python vis_results_B.py -I .\datasets\test_images         -B .\results\resB2_fix.json
