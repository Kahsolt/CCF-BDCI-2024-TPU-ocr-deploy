#!/usr/bin/env bash

# 一键编译所有ppocr模型！

# [量化情况]
#      IN   PROCESS   OUT
# det UINT8   INT8   INT8
# rec UINT8   BF16   FP32     // out is force-casted to FP32 due to chip cpu not support BF16 :(
# cls UINT8   INT8   INT8
#
# [前处理情况]
# - 所有模型均有 --fuse_preprocess --quant_input
#   - det 和 rec/cls 分别使用一套 mean/var
#   - TPU模型处理 BGR通道顺序 + HWC维度顺序 的转换，无需在CPU转换
# - 对于 INT8 使用对称量化，且启用 --quant_ouput
# 后处理情况
# - rec/cls 模型使用 *.modify.onnx，即网络结构中去掉了最后一层 softmax
# 
# [部署时精度校验情况]
# | det-v4-int8    | --tolerance 0.85,0.45 --compare_all |
# | det-v3-int8    | --tolerance 0.85,0.45               |
# | det-v2-int8    | --tolerance 0.85,0.45 --compare_all |
# | det-mb-int8    | --tolerance 0.85,0.45 --compare_all |
# | cls-mb-int8    | --tolerance 0.85,0.45 --compare_all |
# | rec-mb-bf16    | --tolerance 0.85,0.45 --compare_all |

# rec/cls 模型常规量化到 INT8
bash ./compile_cvimodel.sh det v4 int8
bash ./compile_cvimodel.sh det v3 int8
bash ./compile_cvimodel.sh det v2 int8
bash ./compile_cvimodel.sh det mb int8
bash ./compile_cvimodel.sh cls mb int8

# rec 模型用 BF16 更加保险
bash ./compile_cvimodel.sh rec mb bf16

# 对不起做不到
#bash ./compile_cvimodel.sh rec v4 bf16         # model too large
#bash ./compile_cvimodel.sh rec v3 bf16         # model too large
#bash ./compile_cvimodel.sh rec v2 bf16         # model too large (?)
#bash ./compile_cvimodel.sh det_prune mb int8   # similarity check failed
