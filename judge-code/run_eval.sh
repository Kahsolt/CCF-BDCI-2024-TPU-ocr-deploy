# unit test

python3 ppocr_det_opencv_cvi.py \
  --input ./datasets/cali_set_det \
  --cvimodel_det ch_PP-OCRv3_det.cvimodel

python3 ppocr_cls_opencv_cvi.py \
  --input ./datasets/cali_set_rec \
  --cvimodel_cls ch_PP-OCRv3_cls.cvimodel \
  --cls_thresh 0.9 \
  --label_list 0,180

python3 ppocr_rec_opencv_cvi.py \
  --input ./datasets/cali_set_rec \
  --cvimodel_rec ch_PP-OCRv3_rec.cvimodel \
  --img_size [[640,48]] \
  --char_dict_path ./datasets/ppocr_keys_v1.txt


# integration test

python3 ppocr_system_opencv_cvi.py \
  --input=datasets/train_full_images_0 \
  --batch_size=1 \
  --cvimodel_det=ch_PP-OCRv3_det.cvimodel \
  --cvimodel_cls=ch_PP-OCRv3_cls.cvimodel \
  --cvimodel_rec=ch_PP-OCRv3_rec.cvimodel \
  --img_size [[640,48]] \
  --char_dict_path ./datasets/ppocr_keys_v1.txt \
  --use_angle_cls

python eval_score.py \
  --gt_path ./datasets/train_full_images.json \
  --result_json ./results/ppocr_system_results_b1.json \
  --inference_time 100
