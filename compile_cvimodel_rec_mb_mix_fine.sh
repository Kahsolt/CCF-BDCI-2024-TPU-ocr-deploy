#!/usr/bin/env bash

# 针对性地优化 ppocr_mb_rec_mix
# 自动混精度不行，尝试手动混精度: 从 BF16 量化出发，把精度达标的层改成 INT8 (手搓 qtable！)

BASE_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DISTRO_PATH=$BASE_PATH/tpu-sdk-cv180x-ocr/cvimodels

TEST_INPUT=$BASE_PATH/datasets/cali_set_rec_32x320/crop_9.jpg
MODEL_NAME=ppocr_mb_rec
MODEL_DEF=$BASE_PATH/models/ch_ppocr_mobile_v2.0_rec_infer.modify.onnx

# predefine all generated filenames
TEST_INPUT_FP32=${MODEL_NAME}_in_f32.npz
TEST_RESULT=${MODEL_NAME}_outputs.npz
MLIR_MODEL_FILE=${MODEL_NAME}.mlir
CALI_TABLE_FILE=${MODEL_NAME}_cali_table
QTABLE_FILE=../${MODEL_NAME}_qtable     # manually fixed design
CVI_MODEL_FILE=${MODEL_NAME}_mix_fine.cvimodel
CVI_INFO_FILE=${MODEL_NAME}_mix_fine.info

echo ">> Compiling model $MODEL_DEF"
mkdir -p build/$MODEL_NAME
pushd build/$MODEL_NAME > /dev/null

if [ ! -f $CVI_MODEL_FILE ]; then
model_deploy.py \
  --chip cv180x \
  --mlir $MLIR_MODEL_FILE \
  --quantize BF16 \
  --quant_input \
  --quantize_table $QTABLE_FILE \
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
