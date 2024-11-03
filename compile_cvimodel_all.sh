#!/usr/bin/env bash

# 一键编译所有ppocr模型！

# [前处理情况]
# - 所有模型均有 --fuse_preprocess --quant_input --quant_ouput
#   - det 和 rec/cls 分别使用一套 mean/var
#   - TPU模型处理 BGR通道顺序 + HWC维度顺序 的转换，无需在CPU转换
# - 对 det 模型使用 INT8-sym 量化，不然炸 ION 内存
# - 对 rec 模型使用自动混精度量化: INT8-sym -> BF16
# [后处理情况]
# - rec/cls 模型使用 *.modify.onnx，即网络结构中去掉了最后一层 softmax

# [部署版本情况]
# | model  | int8 | bf16 | mix |
# | det-v4 |   √  |      |  -  |
# | det-v3 |   √  |      |  -  |
# | det-v2 |   √  |      |  -  |
# | det-mb |   √  |      |  -  |
# | rec-mb |   x  |   √  |  √  |
# | cls-mb |   √  |      |  -  |
# (*) 减号表示 int8 精度已达标，无需混精度 mix

# gogogo!
bash ./compile_cvimodel.sh det v4 int8
bash ./compile_cvimodel.sh det v3 int8
bash ./compile_cvimodel.sh det v2 int8
bash ./compile_cvimodel.sh det mb int8
bash ./compile_cvimodel.sh rec mb bf16
bash ./compile_cvimodel.sh cls mb int8

# could be better?
#bash ./compile_cvimodel.sh rec mb mix    # This do not work properly :(
bash ./compile_cvimodel_rec_mb_mix_fine.sh

# prec-chk not pass, explosive numerical in network :(
#bash ./compile_cvimodel.sh det_prune mb mix

# model too large!!
#bash ./compile_cvimodel.sh rec v4
#bash ./compile_cvimodel.sh rec v3
#bash ./compile_cvimodel.sh rec v2
