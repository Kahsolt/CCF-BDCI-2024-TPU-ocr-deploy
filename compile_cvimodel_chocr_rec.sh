#!/usr/bin/env bash

# 编译来自 chineseocr_lite 仓库的 rec 模型: onnx -> cvimodel

BASE_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# tpu-mlir compile tools
[ ! -d tpu-mlir ] && git clone -q https://github.com/milkv-duo/tpu-mlir
source ./tpu-mlir/envsetup.sh
# runtime / distro folder
[ ! -d tpu-sdk-cv180x-ocr ] && git clone -q https://github.com/Kahsolt/tpu-sdk-cv180x-ocr
DISTRO_PATH=$BASE_PATH/tpu-sdk-cv180x-ocr/cvimodels

# [bf16, int8]
DTYPE=${1:-'bf16'}

MODEL_NAME=chocr_rec
MODEL_DEF=$BASE_PATH/models/crnn_lite_lstm.onnx
INPUT_SHAPE='[[1,3,32,320]]'
MEAN=127.5,127.5,127.5
SCALE=0.0078125,0.0078125,0.0078125
CALI_DATASET=$BASE_PATH/datasets/cali_set_rec_32x320
TEST_INPUT=$CALI_DATASET/crop_9.jpg

# predefine all generated filenames
TEST_INPUT_FP32=${MODEL_NAME}_in_f32.npz
TEST_RESULT=${MODEL_NAME}_outputs.npz
MLIR_MODEL_FILE=${MODEL_NAME}.mlir
CALI_TABLE_FILE=${MODEL_NAME}_cali_table
CVI_MODEL_FILE=${MODEL_NAME}_${DTYPE}.cvimodel
CVI_INFO_FILE=${MODEL_NAME}_${DTYPE}.info

echo ">> Compiling model $MODEL_DEF"
mkdir -p tmp ; pushd tmp > /dev/null
if [ ! -f $MLIR_MODEL_FILE ]; then
model_transform.py \
  --model_name $MODEL_NAME \
  --model_def $MODEL_DEF \
  --input_shapes $INPUT_SHAPE \
  --mean $MEAN \
  --scale $SCALE \
  --keep_aspect_ratio \
  --test_input $TEST_INPUT \
  --test_result $TEST_RESULT \
  --debug \
  --mlir $MLIR_MODEL_FILE
fi
if [ ! -f $CALI_TABLE_FILE ]; then
run_calibration.py $MLIR_MODEL_FILE \
  --dataset $CALI_DATASET \
  --input_num 300 \
  -o $CALI_TABLE_FILE
fi
if [ ! -f $CVI_MODEL_FILE ]; then
model_deploy.py \
  --chip cv180x \
  --mlir $MLIR_MODEL_FILE \
  --quantize $DTYPE \
  --quant_input \
  $QUANT_OUTPUT \
  --calibration_table $CALI_TABLE_FILE \
  --test_input $TEST_INPUT \
  --test_reference $TEST_RESULT \
  --tolerance 0.85,0.45 \
  --compare_all \
  --fuse_preprocess \
  --customization_format BGR_PACKED \
  --ignore_f16_overflow \
  --op_divide \
  --debug \
  --model $CVI_MODEL_FILE
fi
if [ -f $CVI_MODEL_FILE ]; then
  echo ">> Compile model done!"
else
  echo ">> Compile model FAILED!"
  exit
fi

echo ">> Save model to $CVI_MODEL_FILE the runtime repo"
cp -u $CVI_MODEL_FILE $DISTRO_PATH
model_tool --info $CVI_MODEL_FILE > $DISTRO_PATH/$CVI_INFO_FILE

echo ">> Upload model $CVI_MODEL_FILE to MilkV-Duo!"
ssh root@192.168.42.1 "mkdir -p /root/tpu-sdk-cv180x-ocr/cvimodels"
scp $CVI_MODEL_FILE root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels
popd > /dev/null

echo ">> All done!"
