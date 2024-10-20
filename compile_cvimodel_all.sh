#!/usr/bin/env bash

# 一键编译所有ppocr模型！

# 量化情况 (所有模型均有 --fuse_preprocess --quant_input --quant_output, 且为对称量化)
#      IN   PROCESS   OUT
# det UINT8   INT8   INT8     // --quantize INT8
# rec UINT8   BF16   FP32     // --quantize BF16  (out is force-casted to FP32 due to chip cpu not support BF16 :(
# cls UINT8   INT8   INT8     // --quantize INT8
# (*) rec-v3/v4 无法部署，不支持 attn-softmax 操作 (疑似爆内存)
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
