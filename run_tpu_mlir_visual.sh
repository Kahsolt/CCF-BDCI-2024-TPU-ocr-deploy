#!/usr/bin/env bash

# 启动 tpu-mlir 工具箱的 visual 工具，逐层对比 mlir 量化前后的中间值
# the mlirs: *_origin.mlir => *.mlir => *_cv180x_bf16|int8[_sym]_tpu.mlir => *_cv180x_bf16|int8[_sym]_final.mlir

BASE_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# tpu-mlir compile tools
[ ! -d tpu-mlir ] && git clone -q https://github.com/milkv-duo/tpu-mlir
source ./tpu-mlir/envsetup.sh


MODEL=ppocr_mb_rec

pushd build/$MODEL > /dev/null

#visual.py \
#  --f32_mlir $MODEL.mlir \
#  --quant_mlir ${MODEL}_cv180x_int8_sym_tpu.mlir \
#  --input $BASE_PATH/datasets/cali_set_det/gt_97.jpg \
#  -p 10000 \
#  --debug

visual.py \
  --f32_mlir $MODEL.mlir \
  --quant_mlir ${MODEL}_cv180x_bf16_tpu.mlir \
  --input $BASE_PATH/datasets/cali_set_rec/crop_177.jpg \
  -p 10000 \
  --debug

popd > /dev/null
