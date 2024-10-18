@ECHO OFF

REM ppocr releases: https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/model_list.html
REM We only need chinese models ;)

wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar
wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar
wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar
wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar

wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar
wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_prune_infer.tar
wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar
wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar

REM slim models are not supported
REM wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_infer.tar
REM wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.tar
REM wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_slim_quant_infer.tar
REM wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_infer.tar
REM wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_slim_infer.tar


REM paddle to onnx
FOR /F %%f in ('DIR /O:NG /T:W /B *.tar') DO (
  paddleocr_convert -p %%f -o . -txt_path ../datasets/ppocr_keys_v1.txt
  DEL /Q %%f
)

REM move onnx out and clean extracted files
FOR /F %%d in ('DIR /O:NG /T:W /B *_infer') DO (
  PUSHD %%d
  MOVE *.onnx ..
  POPD
  RMDIR /Q /S %%d
)
