### 板上运行+评测结果

⚪ ppocr_v3

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

```shell
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel  ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/train_full_images_0
version: 1.4.0
ppocrv2_det Build at 2024-11-01 13:47:17 For platform cv180x
Max SharedMem size:8179200
version: 1.4.0
ppocr_mb_rec Build at 2024-11-01 12:56:51 For platform cv180x
Max SharedMem size:1075360
find shared memory(8179200),  saved:1075360 
ts_model_load: 700.000 ms
ts_model_unload: 102.037 ms
================================
n_img:        2350
n_crop:       12461
--------------------------------
ts_img_load:  486053.812 ms
ts_img_crop:  69860.852 ms
ts_det_pre:   235522.891 ms
ts_det_infer: 534852.000 ms
ts_det_post:  78957.102 ms
ts_rec_pre:   18113.322 ms
ts_rec_infer: 415385.375 ms
ts_rec_post:  73736.391 ms
--------------------------------
ts_avg_pre:   107.930 ms
ts_avg_infer: 404.356 ms
ts_avg_post:  64.976 ms
================================
Total time:   1915228.250 ms
```

Judge score:

```shell
python eval_score.py ^
  --gt_path ..\datasets\train_full_images.json ^
  --result_json ..\results\res_v2.json ^
  --inference_time 405

100%|████████████████████████████████████████████████████████████████| 2350/2350 [00:07<00:00, 332.85it/s]
F-score: 0.41318, Precision: 0.56653, Recall: 0.32516
Inference time: 405.00000
Score: 72.10203
```

⚪ ppocr_mb

```shell
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocr_mb_det_int8.cvimodel ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/train_full_images_0
version: 1.4.0
ppocr_mb_det Build at 2024-11-01 13:21:52 For platform cv180x
Max SharedMem size:8179200
version: 1.4.0
ppocr_mb_rec Build at 2024-11-01 12:56:51 For platform cv180x
Max SharedMem size:1075360
find shared memory(8179200),  saved:1075360 
ts_model_load: 738.000 ms
ts_model_unload: 110.439 ms
================================
n_img:        2350
n_crop:       10277
--------------------------------
ts_img_load:  489666.250 ms
ts_img_crop:  58280.066 ms
ts_det_pre:   238023.156 ms
ts_det_infer: 534782.938 ms
ts_det_post:  80013.414 ms
ts_rec_pre:   14772.701 ms
ts_rec_infer: 342676.438 ms
ts_rec_post:  60796.355 ms
--------------------------------
ts_avg_pre:   107.573 ms
ts_avg_infer: 373.387 ms
ts_avg_post:  59.919 ms
================================
Total time:   1821747.000 ms

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocr_mb_det_int8.cvimodel ../cvimodels/ppocr_mb_rec_bf16.cvimodel /data/train_full_images_0
version: 1.4.0
ppocr_mb_det Build at 2024-11-01 13:21:52 For platform cv180x
Max SharedMem size:8179200
version: 1.4.0
ppocr_mb_rec Build at 2024-11-01 12:56:51 For platform cv180x
Max SharedMem size:1075360
find shared memory(8179200),  saved:1075360 
ts_model_load: 960.000 ms
ts_model_unload: 98.064 ms
================================
n_img:        2350
n_crop:       10277
--------------------------------
ts_img_load:  483748.594 ms
ts_img_crop:  55850.594 ms
ts_det_pre:   226420.719 ms
ts_det_infer: 533198.312 ms
ts_det_post:  74040.242 ms
ts_rec_pre:   14540.577 ms
ts_rec_infer: 342629.875 ms
ts_rec_post:  60787.332 ms
--------------------------------
ts_avg_pre:   102.537 ms
ts_avg_infer: 372.693 ms
ts_avg_post:  57.373 ms
================================
Total time:   1794298.125 ms
```

Judge score:

```shell
python eval_score.py ^
  --gt_path ..\datasets\train_full_images.json ^
  --result_json ..\results\res_mb.json ^
  --inference_time 374

100%|████████████████████████████████████████████████████████████████| 2350/2350 [00:05<00:00, 409.34it/s]
F-score: 0.32552, Precision: 0.52379, Recall: 0.23614
Inference time: 374.00000
Score: 71.23088
```

----
by Armit
2024/11/01
