### 板上运行+评测结果

⚪ ppocr_v3

ℹ 使用超参数 `DET_SEG_THRESH=0.3` 和 `DET_MIN_SIZE=3`  
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
ts_model_load: 880.000 ms
ts_model_unload: 109.655 ms
================================
n_img:        2350
n_crop:       15581
--------------------------------
ts_img_load:  494600.969 ms
ts_img_crop:  78126.922 ms
ts_det_pre:   247432.531 ms
ts_det_infer: 579572.062 ms
ts_det_post:  84495.453 ms
ts_rec_pre:   21291.232 ms
ts_rec_infer: 519631.562 ms
ts_rec_post:  92250.250 ms
--------------------------------
ts_avg_pre:   114.351 ms
ts_avg_infer: 467.746 ms
ts_avg_post:  75.211 ms
================================
Total time:   2120964.250 ms
```

Judge score:

```shell
python eval_score.py ^
  --gt_path ..\datasets\train_full_images.json ^
  --result_json ..\results\res_v3.json ^
  --inference_time 468

100%|████████████████████████████████████████████████████████████████| 2350/2350 [00:08<00:00, 280.81it/s]
F-score: 0.41687, Precision: 0.51316, Recall: 0.35100
Inference time: 468.00000
Score: 66.89474
```

⚪ ppocr_v2

ℹ 使用超参数 `DET_SEG_THRESH=0.5` 和 `DET_MIN_SIZE=5`
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
ts_model_load: 709.500 ms
ts_model_unload: 122.508 ms
================================
n_img:        2350
n_crop:       10963
--------------------------------
ts_img_load:  495171.344 ms
ts_img_crop:  70669.398 ms
ts_det_pre:   233599.328 ms
ts_det_infer: 536907.812 ms
ts_det_post:  82139.875 ms
ts_rec_pre:   17274.988 ms
ts_rec_infer: 366274.719 ms
ts_rec_post:  65069.387 ms
--------------------------------
ts_avg_pre:   106.755 ms
ts_avg_infer: 384.333 ms
ts_avg_post:  62.642 ms
================================
Total time:   1870048.500 ms

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
Total time:   2132573.750 ms
```

Judge score:

```shell
python eval_score.py ^
  --gt_path ..\datasets\train_full_images.json ^
  --result_json ..\results\res_v2.json ^
  --inference_time 384.333
100%|████████████████████████████████████████████████████████████████| 2350/2350 [00:05<00:00, 414.35it/s]
F-score: 0.41665, Precision: 0.59889, Recall: 0.31944
Inference time: 384.33300
Score: 73.99756

# with dilate
python eval_score.py ^
  --gt_path ..\datasets\train_full_images.json ^
  --result_json ..\results\res_v2_dilate.json ^
  --inference_time 410.420
100%|████████████████████████████████████████████████████████████████| 2350/2350 [00:07<00:00, 297.74it/s]
F-score: 0.20217, Precision: 0.27758, Recall: 0.15898
Inference time: 410.42000
Score: 63.20104
```

⚪ ppocr_mb

ℹ 使用超参数 `DET_SEG_THRESH=0.5` 和 `DET_MIN_SIZE=5`  
⚠ 这条路线主打更快的速度  

```shell
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppoc
r_mb_det_int8.cvimodel ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/train_full_images_0
version: 1.4.0
ppocr_mb_det Build at 2024-11-01 13:21:52 For platform cv180x
Max SharedMem size:8179200
version: 1.4.0
ppocr_mb_rec Build at 2024-11-01 12:56:51 For platform cv180x
Max SharedMem size:1075360
find shared memory(8179200),  saved:1075360
ts_model_load: 680.000 ms
ts_model_unload: 97.938 ms
================================
n_img:        2350
n_crop:       8516
--------------------------------
ts_img_load:  490526.062 ms
ts_img_crop:  56534.555 ms
ts_det_pre:   229414.125 ms
ts_det_infer: 534560.562 ms
ts_det_post:  78513.445 ms
ts_rec_pre:   13347.812 ms
ts_rec_infer: 284500.500 ms
ts_rec_post:  50491.832 ms
--------------------------------
ts_avg_pre:   103.303 ms
ts_avg_infer: 348.537 ms
ts_avg_post:  54.896 ms
================================
Total time:   1740312.875 ms
```

Judge score:

```shell
-> python eval_score.py ^
  --gt_path ..\datasets\train_full_images.json ^
  --result_json ..\results\res_mb.json ^
  --inference_time 348.537

100%|████████████████████████████████████████████████████████████████| 2350/2350 [00:04<00:00, 554.24it/s]
F-score: 0.32444, Precision: 0.56995, Recall: 0.22676
Inference time: 348.53700
Score: 73.35208
```

----
by Armit
2024/11/01
