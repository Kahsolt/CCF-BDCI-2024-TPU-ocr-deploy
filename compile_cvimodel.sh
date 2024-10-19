#!/usr/bin/env bash

# 编译 ch_PP-OCRv*_*_infer/ch_ppocr_mobile_v2.0_*_infer 系列模型: onnx -> cvimodel
# - https://docs.qq.com/pdf/DSUlabGVFRlBkQkZv
# - https://doc.sophgo.com/sdk-docs/v23.09.01-lts/docs_latest_release/docs/tpu-mlir/developer_manual/html/03_user_interface.html

# TODO: model_deploy.py --op_divide

BASE_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
pushd $BASE_PATH

# [det, rec, cls]
TASK=${1:-'det'}
# [v2, v3, v4, mb]
VERSION=${2:-'v3'}

if [ "$TASK" == "cls" ] && [ "$VERSION" != "mb" ]; then
  VERSION=mb
  echo ">> warn: force VERSION=mb when set TASK=cls!"
fi

if [ "$TASK" == "det" ]; then
  INPUT_SHAPE='[[1,3,640,640]]'
  CALI_DATASET=$BASE_PATH/datasets/cali_set_det
  TEST_INPUT=$CALI_DATASET/gt_97.jpg
else
  INPUT_SHAPE='[[1,3,48,640]]'
  CALI_DATASET=$BASE_PATH/datasets/cali_set_rec
  TEST_INPUT=$CALI_DATASET/crop_9.jpg
fi
if [ "$TASK" == "rec" ]; then
  DTYPE=BF16
else
  DTYPE=INT8
fi
if [ "$VERSION" == "mb" ]; then
  MODEL_NAME=ppocr_mb_${TASK}
  MODEL_DEF=$BASE_PATH/models/ch_ppocr_mobile_v2.0_${TASK}_infer.onnx
else
  MODEL_NAME=ppocr${VERSION}_${TASK}
  MODEL_DEF=$BASE_PATH/models/ch_PP-OCR${VERSION}_${TASK}_infer.onnx
fi

# predefine all generated filenames
TEST_INPUT_FP32=${MODEL_NAME}_in_f32.npz
TEST_RESULT=${MODEL_NAME}_outputs.npz
MLIR_MODEL_FILE=${MODEL_NAME}.mlir
CALI_TABLE_FILE=${MODEL_NAME}_cali_table
CVI_MODEL_FILE=${MODEL_NAME}.cvimodel

# tpu-mlir compile tools
git clone -q https://github.com/milkv-duo/tpu-mlir
source ./tpu-mlir/envsetup.sh

echo ">> Compiling model $MODEL_DEF"
mkdir -p tmp ; pushd tmp
# onnx -> mlir
if [ ! -f $MLIR_MODEL_FILE ]; then
model_transform.py \
  --model_name $MODEL_NAME \
  --model_def $MODEL_DEF \
  --input_shapes $INPUT_SHAPE \
  --mean 0.0,0.0,0.0 \
  --scale 0.0039216,0.0039216,0.0039216 \
  --keep_aspect_ratio \
  --pixel_format rgb \
  --test_input $TEST_INPUT \
  --test_result $TEST_RESULT \
  --debug \
  --mlir $MLIR_MODEL_FILE
fi
# deplot INT8
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
  --calibration_table $CALI_TABLE_FILE \
  --test_input $TEST_INPUT \
  --test_reference $TEST_RESULT \
  --tolerance 0.45,0.45 \
  --fuse_preprocess \
  --debug \
  --model $CVI_MODEL_FILE
fi

model_tool --info $CVI_MODEL_FILE

model_runner.py \
  --input $TEST_INPUT_FP32 \
  --model $CVI_MODEL_FILE \
  --output $TEST_RESULT

mkdir -p $BASE_PATH/cvimodel
cp $CVI_MODEL_FILE $BASE_PATH/cvimodel
popd

echo Done!
