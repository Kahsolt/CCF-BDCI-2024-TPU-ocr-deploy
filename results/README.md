### 板上运行+评测结果

ℹ 如无特殊说明，以下实验均使用超参数 `DET_SEG_THRESH=0.5` 和 `DET_MIN_SIZE=5`  

关于评分 & 得分的讨论

- 评分公式中，判为 TP 的条件是: `iou(box) > 0.5` && `strsim(txt) > 0.5` (严格大于!!)
  - box条件相对容易达成 (det模型要求低)，txt条件不容易达成 (rec模型要求高)
  - `v3_det + mb_rec` 和 `v2_det + mb_rec` 的得分差异虽然不大，但是前者错字率确实更低，而后者召回的框更多
- 板上精度不容易提高的原因
  - recall: 要求 det 模型 分割阈值低 + 预测碎片少、要求 rec 模型推理快；这两条都很难做到
  - precision: rec 模型量化后预测就是不太好，甚至有字典外词
- 刷分思路 & 矛盾
  - 尽可能多召回框，每个框的文本错字率少于一半就成功！
  - CRNN太慢了、有的框里其实没有字，为了降低板上计算量、加快推理速度，我们不得不提高阈值减少待 rec 框的数量

推理时间分析

- det
  - v3: 244.60 ms/img
  - v2: 228.71 ms/img (⭐)
  - mb: 223.63 ms/img
- rec
  - v2: 105.21 ms/crop
  - mb (bf16): 33.35 ms/crop (⭐)
  - mb (mix-fine): 33.17 ms/crop (精度很烂)
  - mb (mix): 18.07 ms/crop (精度很烂)

----

⚪ v3_det + mb_rec

⚠ 这条路线精度收益不大，可能 det 模型的预测还是太支离破碎了  

```shell
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv3_det_int8.cvimodel  ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/train_full_images_0
version: 1.4.0
ppocrv3_det Build at 2024-11-01 13:08:39 For platform cv180x
Max SharedMem size:8793600
version: 1.4.0
ppocr_mb_rec Build at 2024-11-01 12:56:51 For platform cv180x
Max SharedMem size:1075360
find shared memory(8793600),  saved:1075360
ts_model_load: 824.000 ms
ts_model_unload: 118.216 ms
================================
n_img:        2350
n_crop:       13293
--------------------------------
ts_img_load:  459602.250 ms
ts_img_crop:  72313.211 ms
ts_det_pre:   213001.766 ms
ts_det_infer: 574809.625 ms
ts_det_post:  61579.309 ms
ts_rec_pre:   18722.221 ms
ts_rec_infer: 443212.906 ms
ts_rec_post:  78747.188 ms
--------------------------------
ts_avg_pre:   98.606 ms
ts_avg_infer: 433.201 ms
ts_avg_post:  59.713 ms
================================
Total time:   1925107.125 ms
```

Judge score:

```shell
python eval_score.py ^
  --gt_path ..\datasets\train_full_images.json ^
  --result_json ..\results\res_v3.json ^
  --inference_time 433.201
100%|████████████████████████████████████████████████████████████████| 2350/2350 [00:07<00:00, 333.88it/s]
F-score: 0.42010, Precision: 0.54821, Recall: 0.34052
Inference time: 433.20100
Score: 69.98178
```

⚪ v2_det + mb_rec/v2_rec

⚠ 这条路线争取更高的精度  

```shell
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel  ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/train_full_images_0
version: 1.4.0
ppocrv2_det Build at 2024-11-01 13:47:17 For platform cv180x
Max SharedMem size:8179200
version: 1.4.0
ppocr_mb_rec Build at 2024-11-01 12:56:51 For platform cv180x
Max SharedMem size:1075360
find shared memory(8179200),  saved:1075360
ts_model_load: 1272.000 ms
ts_model_unload: 112.709 ms
================================
n_img:        2350
n_crop:       10963
--------------------------------
ts_img_load:  483269.844 ms
ts_img_crop:  68317.258 ms
ts_det_pre:   225042.297 ms
ts_det_infer: 533648.875 ms
ts_det_post:  75878.125 ms
ts_rec_pre:   16939.449 ms
ts_rec_infer: 365749.844 ms
ts_rec_post:  64983.641 ms
--------------------------------
ts_avg_pre:   102.971 ms
ts_avg_infer: 382.723 ms
ts_avg_post:  59.941 ms
================================
Total time:   1837247.125 ms

# with DET_SEG_THRESH=0.6
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel  ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/train_full_images_0
version: 1.4.0
ppocrv2_det Build at 2024-11-03 17:18:35 For platform cv180x
Max SharedMem size:8179200
version: 1.4.0
ppocr_mb_rec Build at 2024-11-01 12:56:51 For platform cv180x
Max SharedMem size:1075360
find shared memory(8179200),  saved:1075360 
ts_model_load: 625.641 ms
ts_model_unload: 99.197 ms
================================
n_img:        2350
n_crop:       10232
------------[Total]-------------
ts_img_load:  433020.969 ms
ts_img_crop:  55789.320 ms
ts_det_pre:   181388.703 ms
ts_det_infer: 523809.938 ms
ts_det_post:  43271.453 ms
ts_rec_pre:   14800.529 ms
ts_rec_infer: 340858.406 ms
ts_rec_post:  60596.555 ms
-----------[Average]------------
ts_det_pre:   77.187 ms
ts_rec_pre:   1.446 ms
ts_pre:       83.485 ms
ts_det_infer: 222.898 ms
ts_rec_infer: 33.313 ms
ts_infer:     367.944 ms
ts_det_post:  18.413 ms
ts_rec_post:  5.922 ms
ts_post:      44.199 ms
================================
Total time:   1655957.500 ms

# with dilate
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel  ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/train_full_images_0
version: 1.4.0
ppocrv2_det Build at 2024-11-01 13:47:17 For platform cv180x
Max SharedMem size:8179200
version: 1.4.0
ppocr_mb_rec Build at 2024-11-01 12:56:51 For platform cv180x
Max SharedMem size:1075360
find shared memory(8179200),  saved:1075360
ts_model_load: 746.000 ms
ts_model_unload: 104.360 ms
================================
n_img:        2350
n_crop:       12632
--------------------------------
ts_img_load:  532231.375 ms
ts_img_crop:  95815.852 ms
ts_det_pre:   251827.391 ms
ts_det_infer: 541911.875 ms
ts_det_post:  191669.469 ms
ts_rec_pre:   18334.904 ms
ts_rec_infer: 422575.875 ms
ts_rec_post:  74878.844 ms
--------------------------------
ts_avg_pre:   114.963 ms
ts_avg_infer: 410.420 ms
ts_avg_post:  113.425 ms
================================
Total time:   2132573.750  ms

# with rec_mix_fine
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel ../cvimodels/ppocr_mb_rec_mix_fine.cvimodel /data/train_full_images_0
version: 1.4.0
ppocrv2_det Build at 2024-11-03 17:18:35 For platform cv180x
Max SharedMem size:8179200
version: 1.4.0
ppocr_mb_rec Build at 2024-11-03 18:35:08 For platform cv180x
Max SharedMem size:1075360
find shared memory(8179200),  saved:1075360 
ts_model_load: 736.000 ms
ts_model_unload: 108.799 ms
================================
n_img:        2350
n_crop:       10963
------------[Total]-------------
ts_img_load:  449695.469 ms
ts_img_crop:  63493.660 ms
ts_det_pre:   206047.172 ms
ts_det_infer: 527193.500 ms
ts_det_post:  53778.926 ms
ts_rec_pre:   16207.931 ms
ts_rec_infer: 363686.875 ms
ts_rec_post:  64920.508 ms
-----------[Average]------------
ts_det_pre:   87.680 ms
ts_rec_pre:   1.478 ms
ts_pre:       94.577 ms
ts_det_infer: 224.338 ms
ts_rec_infer: 33.174 ms
ts_infer:     379.098 ms
ts_det_post:  22.885 ms
ts_rec_post:  5.922 ms
ts_post:      50.510 ms
================================
Total time:   1747858.750 ms

# with v2_rec
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel  ../cvimodels/ppocrv2_rec_bf16.cvimodel /data/train_full_images_0
version: 1.4.0
ppocrv2_det Build at 2024-11-01 13:47:17 For platform cv180x
Max SharedMem size:8179200
version: 1.4.0
ppocrv2_rec Build at 2024-11-02 22:19:55 For platform cv180x
Max SharedMem size:1075360
find shared memory(8179200),  saved:1075360
ts_model_load: 1208.000 ms
ts_model_unload: 92.979 ms
================================
n_img:        2350
n_crop:       10963
--------------------------------
ts_img_load:  494276.375 ms
ts_img_crop:  71197.633 ms
ts_det_pre:   240060.234 ms
ts_det_infer: 537510.000 ms
ts_det_post:  86418.023 ms
ts_rec_pre:   17251.906 ms
ts_rec_infer: 1153406.375 ms
ts_rec_post:  64942.500 ms
--------------------------------
ts_avg_pre:   109.495 ms
ts_avg_infer: 719.539 ms
ts_avg_post:  64.409 ms
================================
Total time:   2668221.000 ms
```

Judge score:

```shell
python eval_score.py ^
  --gt_path ..\datasets\train_full_images.json ^
  --result_json ..\results\res_v2.json ^
  --inference_time 382.723
100%|████████████████████████████████████████████████████████████████| 2350/2350 [00:06<00:00, 387.58it/s]
F-score: 0.41775, Precision: 0.60048, Recall: 0.32029
Inference time: 382.72300
Score: 74.17861

# with DET_SEG_THRESH=0.6
python ..\judge-code\eval_score.py ^
  --gt_path ..\datasets\train_full_images.json ^
  --result_json ..\results\results.json ^
  --inference_time 367.944
100%|████████████████████████████████████████████████████████████████| 2350/2350 [00:05<00:00, 428.34it/s]
F-score: 0.42781, Precision: 0.63849, Recall: 0.32166
Inference time: 367.94400
Score: 75.83703 (⭐)

# with dilate
python eval_score.py ^
  --gt_path ..\datasets\train_full_images.json ^
  --result_json ..\results\res_v2_dilate.json ^
  --inference_time 410.420
100%|████████████████████████████████████████████████████████████████| 2350/2350 [00:07<00:00, 297.74it/s]
F-score: 0.20217, Precision: 0.27758, Recall: 0.15898
Inference time: 410.42000
Score: 63.20104

# with mb_rec_mix_fine
python eval_score.py ^
  --gt_path ..\datasets\train_full_images.json ^
  --result_json ..\results\res_v2_mix_fine.json ^
  --inference_time 379.098
100%|████████████████████████████████████████████████████████████████| 2350/2350 [00:05<00:00, 442.32it/s]
F-score: 0.35770, Precision: 0.57243, Recall: 0.26013
Inference time: 379.09800
Score: 72.08486

# with v2_rec
python eval_score.py ^
  --gt_path ..\datasets\train_full_images.json ^
  --result_json ..\results\res_v2_recv2.json ^
  --inference_time 719.539
100%|████████████████████████████████████████████████████████████████| 2350/2350 [00:05<00:00, 457.92it/s]
F-score: 0.44099, Precision: 0.65593, Recall: 0.33215
Inference time: 719.53900
Score: 46.47884
```

⚪ mb_det + mb_rec

⚠ 这条路线主打更快的速度  

```shell
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocr_mb_det_int8.cvimodel ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/train_full_images_0
version: 1.4.0
ppocr_mb_det Build at 2024-11-01 13:21:52 For platform cv180x
Max SharedMem size:8179200
version: 1.4.0
ppocr_mb_rec Build at 2024-11-01 12:56:51 For platform cv180x
Max SharedMem size:1075360
find shared memory(8179200),  saved:1075360
ts_model_load: 800.000 ms
ts_model_unload: 91.092 ms
================================
n_img:        2350
n_crop:       8516
--------------------------------
ts_img_load:  441576.031 ms
ts_img_crop:  48579.242 ms
ts_det_pre:   191497.203 ms
ts_det_infer: 525525.688 ms
ts_det_post:  45419.188 ms
ts_rec_pre:   12404.003 ms
ts_rec_infer: 283600.375 ms
ts_rec_post:  50385.211 ms
--------------------------------
ts_avg_pre:   86.766 ms
ts_avg_infer: 344.309 ms
ts_avg_post:  40.768 ms
================================
Total time:   1601302.000 ms
```

Judge score:

```shell
python eval_score.py ^
  --gt_path ..\datasets\train_full_images.json ^
  --result_json ..\results\res_mb.json ^
  --inference_time 344.309
100%|████████████████████████████████████████████████████████████████| 2350/2350 [00:04<00:00, 564.50it/s]
F-score: 0.32475, Precision: 0.57048, Recall: 0.22698
Inference time: 344.30900
Score: 73.72358
```

----
by Armit
2024/11/01
