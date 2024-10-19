⚪ 使用 tpu-mlir 工具链，所有 rec 模型都无法部署 model_deploy

https://github.com/sophgo/tpu-mlir/issues/193

运行

```shell
model_deploy.py \
  --chip cv180x \
  --quantize INT8 \
  --mlir ppocrv3_rec.mlir \
  --calibration_table ppocrv3_rec_cali_table \
  --model ppocrv3_rec.cvimodel
```

报错：

```
cvkcv180x tiu xor: wrong parameter
tpuc-opt: /home/jenkins/workspace/tpu-mlir/lib/Backend/CV18xx/Kernel/TgBf16SoftmaxKernel.cpp:256: void tpu_mlir::backend::TgSoftmaxKernel::softmaxLargeSizeHandler(): Assertion `tl_lut_reciprocal_result' failed.
```

报错处代码：

```cpp
// https://github.com/sophgo/tpu-mlir/blob/master/lib/Backend/CV18xx/Kernel/TgBf16SoftmaxKernel.cpp
cvk_tl_t *tl_lut_reciprocal_result =
   CV18xx::lmem_alloc_tensor(lut_result_shape, fmt, eu_align);
ASSERT(tl_lut_reciprocal_result);
```

其他参数设置的对照：

| quantize \ chip | cv180x | cv181x |
| :-: | :-: | :-: |
| INT8 | `cvkcv180x tiu xor: wrong parameter` | √ |
| BF16 | `cvkcv180x tiu xor: wrong parameter` | √ |
|  F16 | `Not Implemented` | `Not Implemented` |
|  F32 | `Not Implemented` | `Not Implemented` |

可能躲避 softmax 吗：

- PPOCR-rec v3/v4 架构为 SVTRNet，里面有 Dot-Attn 机制：https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/backbones/rec_svtrnet.py#L419
- PPOCR-rec v2/mb 架构为 CRNN，其 head=CTC，最后一层是 softmax: https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/heads/rec_ctc_head.py
  - 尝试魔改 mlir 去掉最后一层softmax

[可行!] 魔改 mlir 去掉最后一层后执行 (注意量化到BF16，识别模型INT8量化将无意义，尤其对于接近输出端的层):

```
model_deploy.py \
  --chip cv180x \
  --quantize BF16 \
  --quant_input \
  --mlir ppocr_mb_rec.mlir \
  --calibration_table ppocr_mb_rec_cali_table \
  --model ppocr_mb_rec.cvimodel
model_deploy.py \
  --chip cv180x \
  --quantize BF16 \
  --quant_input \
  --mlir ppocrv2_rec.mlir \
  --calibration_table ppocrv2_rec_cali_table \
  --model ppocrv2_rec.cvimodel
```
