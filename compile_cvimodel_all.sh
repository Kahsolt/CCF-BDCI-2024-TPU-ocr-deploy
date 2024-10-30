#!/usr/bin/env bash

# 一键编译所有ppocr模型！

# [量化情况]
#      IN   PROCESS   OUT
# det UINT8   INT8   FP32
# rec UINT8   BF16   FP32     // out is force-casted to FP32 due to chip cpu not support BF16 :(
# cls UINT8   INT8   FP32
# (*) rec-v3/v4 无法部署，不支持 attn-softmax 操作 (疑似爆内存)
# (*) det 必须 INT8 量化否则炸 ION 内存；rec 必须 BF16 量化否则精度堪忧 :(
#
# [前处理情况]
# - det 使用一套 mean/var, rec/cls 使用另一套 
# - TPU模型处理 BGR通道顺序 + HWC维度顺序 的转换，无需在CPU转换
# - 所有模型均有 --fuse_preprocess --quant_input
# - 对于 INT8 使用对称量化，且启用 --quant_ouput
# 后处理情况
# - rec/cls 模型使用 *.modify.onnx，即网络结构中去掉了最后一层 softmax
# 
# [部署时精度校验情况]
# | det-v4-int8 | --tolerance 0.85,0.45               |
# | det-v3-int8 | --tolerance 0.85,0.45               |
# | det-v2-int8 | --tolerance 0.85,0.45 --compare_all |
# | det-mb-int8 | --tolerance 0.85,0.45 --compare_all |
# | cls-mb-int8 | --tolerance 0.85,0.45 --compare_all |
# | rec-mb-bf16 | --tolerance 0.85,0.45 --compare_all |

# 常规量化到 INT8
bash ./compile_cvimodel.sh det v4 int8
bash ./compile_cvimodel.sh det v3 int8
bash ./compile_cvimodel.sh det v2 int8
bash ./compile_cvimodel.sh det mb int8
bash ./compile_cvimodel.sh cls mb int8

# rec 模型输出必须是浮点数
bash ./compile_cvimodel.sh rec mb bf16
# tpu-mlir 错误，疑似 softmax 操作爆内存
#bash ./compile_cvimodel.sh rec v4 bf16
#bash ./compile_cvimodel.sh rec v3 bf16
