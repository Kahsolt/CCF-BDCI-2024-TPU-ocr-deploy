@ECHO OFF

python run_repo_infer.py -K ppocr --ppocr_ver v4 -O output\val-ppocr_v4.json
python run_repo_infer.py -K ppocr --ppocr_ver v3 -O output\val-ppocr_v3.json
python run_repo_infer.py -K ppocr --ppocr_ver v2 -O output\val-ppocr_v2.json

python run_repo_infer.py -K cnocr --cnocr_ver v3 -O output\val-cnocr_v3.json
python run_repo_infer.py -K cnocr --cnocr_ver v2 -O output\val-cnocr_v2.json

python run_repo_infer.py -K chineseocr_lite --chocr_short_size 960 -O output\val-chocr_960.json
python run_repo_infer.py -K chineseocr_lite --chocr_short_size 640 -O output\val-chocr_640.json
