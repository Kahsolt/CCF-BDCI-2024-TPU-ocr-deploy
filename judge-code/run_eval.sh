export DATA_PATH=../datasets
export MODEL_PATH=../models

# unit test

python3 ppocr_det_opencv_cvi.py \
  --input $DATA_PATH/cali_set_det \
  --cvimodel_det $MODEL_PATH/ch_PP-OCRv3_det_infer.onnx

python3 ppocr_cls_opencv_cvi.py \
  --input $DATA_PATH/cali_set_rec \
  --cvimodel_cls $MODEL_PATH/ch_ppocr_mobile_v2.0_cls_infer.onnx \
  --cls_thresh 0.9 \
  --label_list 0,180

python3 ppocr_rec_opencv_cvi.py \
  --input $DATA_PATH/cali_set_rec \
  --cvimodel_rec $MODEL_PATH/ch_PP-OCRv3_rec_infer.onnx \
  --img_size [[640,48]] \
  --char_dict_path $DATA_PATH/ppocr_keys_v1.txt


# integration test

python3 ppocr_system_opencv_cvi.py \
  --input=$DATA_PATH/train_full_images_0 \
  --cvimodel_det=$MODEL_PATH/ch_PP-OCRv3_det_infer.onnx \
  --cvimodel_cls=$MODEL_PATH/ch_ppocr_mobile_v2.0_cls_infer.onnx \
  --cvimodel_rec=$MODEL_PATH/ch_PP-OCRv3_rec_infer.onnx \
  --batch_size=1 \
  --img_size [[640,48]] \
  --char_dict_path $DATA_PATH/ppocr_keys_v1.txt \
  --use_angle_cls

python eval_score.py \
  --gt_path $DATA_PATH/train_full_images.json \
  --result_json ./results/ppocr_system_results_b1.json \
  --inference_time 100
