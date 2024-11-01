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
+1G         # Last sector
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
scp -r tpu-sdk-cv180x-ocr root@192.168.42.1:/root


#=====================================================================
# Step 3: 跑推理鸭！

# run on chip
cd tpu-sdk-cv180x-ocr
source ./envs_tpu_sdk.sh
cd samples

nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv3_det_int8.cvimodel  ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/train_full_images_0
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel  ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/train_full_images_0
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocr_mb_det_int8.cvimodel ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/train_full_images_0

# run on host (Windows)
pushd tpu-sdk-cv180x-ocr\samples\ppocr_sys_many
scp root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/samples/results.txt .
python convert_results.py .\results.txt
MOVE .\results.json ..\..\..\results
popd

pushd judge-code
python eval_score.py ^
  --gt_path ..\datasets\train_full_images.json ^
  --result_json ..\results\results.json ^
  --inference_time 1000
popd
