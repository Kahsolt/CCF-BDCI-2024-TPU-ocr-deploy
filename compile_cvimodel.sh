#!/usr/bin/env bash

# 编译 ch_PP-OCRv*_*_infer/ch_ppocr_mobile_v2.0_*_infer 系列模型: onnx -> cvimodel
# - https://docs.qq.com/pdf/DSUlabGVFRlBkQkZv
# - https://doc.sophgo.com/sdk-docs/v23.09.01-lts/docs_latest_release/docs/tpu-mlir/developer_manual/html/03_user_interface.html
# - https://tpumlir.org/docs/quick_start/07_fuse_preprocess.html

BASE_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# tpu-mlir compile tools
[ ! -d tpu-mlir ] && git clone -q https://github.com/milkv-duo/tpu-mlir
source ./tpu-mlir/envsetup.sh
# runtime project for distro
[ ! -d tpu-sdk-cv180x-ocr ] && git clone -q https://github.com/Kahsolt/tpu-sdk-cv180x-ocr
DISTRO_PATH=$BASE_PATH/tpu-sdk-cv180x-ocr/cvimodels

# [cv180x]
CHIP=cv180x
# [det, rec, cls, det_prune]
TASK=${1:-'det'}
# [v2, v3, v4, mb]
VERSION=${2:-'v3'}
# [bf16, int8, mix]
DTYPE=${3:-'bf16'}

if [ "$TASK" == "det" ] || [ "$TASK" == "det_prune" ]; then
  INPUT_SHAPE='[[1,3,640,640]]'
  MEAN=123.675,116.28,103.53
  SCALE=0.01712475,0.017507,0.01742919
  CALI_DATASET=$BASE_PATH/datasets/cali_set_det
  TEST_INPUT=$CALI_DATASET/gt_97.jpg
elif [ "$TASK" == "rec" ]; then
  INPUT_SHAPE='[[1,3,32,320]]'
  MEAN=127.5,127.5,127.5
  SCALE=0.0078125,0.0078125,0.0078125
  CALI_DATASET=$BASE_PATH/datasets/cali_set_rec_32x320
  TEST_INPUT=$CALI_DATASET/crop_9.jpg
  ONNX_FILE_SUFFIX=.modify
else    # cls
  INPUT_SHAPE='[[1,3,48,640]]'
  MEAN=127.5,127.5,127.5
  SCALE=0.0078125,0.0078125,0.0078125
  CALI_DATASET=$BASE_PATH/datasets/cali_set_rec_48x640
  TEST_INPUT=$CALI_DATASET/crop_9.jpg
  ONNX_FILE_SUFFIX=.modify
fi
if [ "$DTYPE" == "mix" ]; then
  QTYPE=int8
else
  QTYPE=$DTYPE
fi
if [ "$DTYPE" == "int8" ]; then
  QUANT_OUTPUT=--quant_output
fi
if [ "$TASK" == "cls" ] && [ "$VERSION" != "mb" ]; then
  VERSION=mb
  echo ">> warn: force VERSION=mb when set TASK=cls!"
fi
if [ "$VERSION" == "mb" ]; then
  MODEL_NAME=ppocr_mb_$TASK
  MODEL_DEF=$BASE_PATH/models/ch_ppocr_mobile_v2.0_${TASK}_infer${ONNX_FILE_SUFFIX}.onnx
else
  MODEL_NAME=ppocr${VERSION}_${TASK}
  MODEL_DEF=$BASE_PATH/models/ch_PP-OCR${VERSION}_${TASK}_infer${ONNX_FILE_SUFFIX}.onnx
fi

# predefine all generated filenames
TEST_INPUT_FP32=${MODEL_NAME}_in_f32.npz
TEST_RESULT=${MODEL_NAME}_outputs.npz
MLIR_MODEL_FILE=${MODEL_NAME}.mlir
CALI_TABLE_FILE=${MODEL_NAME}_cali_table
QTABLE_FILE=${MODEL_NAME}_qtable

echo ">> Compiling model $MODEL_DEF"
mkdir -p build/$MODEL_NAME
pushd build/$MODEL_NAME > /dev/null

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
if [ "$DTYPE" == "mix" ] && [ ! -f $QTABLE_FILE ]; then
run_qtable.py $MLIR_MODEL_FILE \
  --chip $CHIP \
  --fp_type BF16 \
  --dataset $CALI_DATASET \
  --calibration_table $CALI_TABLE_FILE \
  --min_layer_cos 0.9999 \
  --expected_cos 0.9999 \
  --input_num 10 \
  -o $QTABLE_FILE
fi
if [ "$DTYPE" == "mix" ] && [ -f $QTABLE_FILE ]; then
  QUANTIZE_TABLE=--quantize_table $QTABLE_FILE
  CVI_MODEL_FILE=${MODEL_NAME}_mix.cvimodel
  CVI_INFO_FILE=${MODEL_NAME}_mix.info
else
  CVI_MODEL_FILE=${MODEL_NAME}_$QTYPE.cvimodel
  CVI_INFO_FILE=${MODEL_NAME}_$QTYPE.info
fi
if [ ! -f $CVI_MODEL_FILE ]; then
model_deploy.py \
  --chip $CHIP \
  --mlir $MLIR_MODEL_FILE \
  --quantize $QTYPE \
  --quant_input \
  $QUANT_OUTPUT \
  $QUANTIZE_TABLE \
  --calibration_table $CALI_TABLE_FILE \
  --test_input $TEST_INPUT \
  --test_reference $TEST_RESULT \
  --tolerance 0.85,0.45 \
  $COMPARE_ALL \
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
