#!/usr/bin/env bash

# 一键编译所有ppocr模型！

# 量化情况 (所有模型均有 --quant_input --fuse_preprocess, 且为对称量化)
#      IN  PROCESS   OUT
# det INT8   INT8   INT8
# rec INT8   BF16   FP32
# cls INT8   INT8   FP32
# (*) rec-v3/v4 无法部署，不支持 softmax 操作
#
# 前处理情况
# - det 使用一套 mean/var, rec/cls 使用另一套 
# - 模型直接处理 BGR 通道顺序的图像，无需转换
# - pack2planer 需要在板上CPU处理，TPU算子好像做不了这件事 (TPU前处理仅包含 mean/scale/channel_swap)

bash ./compile_cvimodel.sh det v4
#bash ./compile_cvimodel.sh rec v4
bash ./compile_cvimodel.sh det v3
#bash ./compile_cvimodel.sh rec v3
bash ./compile_cvimodel.sh det v2
bash ./compile_cvimodel.sh rec v2

bash ./compile_cvimodel.sh det mb
bash ./compile_cvimodel.sh rec mb
bash ./compile_cvimodel.sh cls mb
