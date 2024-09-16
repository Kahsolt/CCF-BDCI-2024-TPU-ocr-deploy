import os
import sys
import json
import glob



import numpy as np


from PIL import Image
from time import time
from pathlib import Path
from tqdm import tqdm


# paddleocr
from paddleocr.paddleocr import PaddleOCR
from paddleocr.tools.infer.utility import draw_ocr
from paddleocr.tools.infer.predict_det import TextDetector
from paddleocr.tools.infer.predict_cls import TextClassifier
from paddleocr.tools.infer.predict_rec import TextRecognizer


# cnocr
#from cnocr import CnOcr

# rapidocr
#from rapidocr_onnxruntime import RapidOCR

# chineseocr_lite
#import onnxruntime as ort
#ort.set_default_logger_severity(3)
#from model import OcrHandle
#np.int = np.int32



parameters = {
    "Precision": 0,
    "Recall": 0,
    "F1-Score": 0,
    "i_time": 0
}



def model_ppocr(img_path):
    ocr     =   PaddleOCR( ocr_version='PP-OCRv4', lang='ch', use_angle_cls=False, use_onnx=False, use_tensorrt=False, precision='fp32')
    im = np.array(Image.open(img_path))
    result = ocr.ocr(im)
    ts_list =   []
    #还需要跑10遍吗
    for _ in tqdm(range(10)):
        ts_start    =   time()
        result      =   ocr.ocr(im)
        ts_end      =   time()
        ts_list.append(ts_end - ts_start)
    
    ts = sum(ts_list) / len(ts_list)
    print(f'>> time cost: {(ts) * 1000:.2f}ms')

    #for idx in range(len(result)):
    #    res = result[idx]
    #    for line in res:
    #        print(line)
    
    #这里没想清楚怎么改，先就加上了时间，然后也只是单张图片的时间
    #parameters["Precision"]    = 0
    #parameters["Recall"]       = 0
    #parameters["F1-Score"]     = 0
    parameters["i_time"]        = (ts) * 1000
    #return parameters



def model_chineseocr_lite(img_path):
    #类似于model_ppocr
    return parameters

def model_cnocr(img_path):
    #类似于model_ppocr
    return parameters

def model_rapidocr(img_path):
    #类似于model_ppocr
    return parameters


def get_arguments():

    if len(sys.argv) != 3:
        print("Usage: python run_repo_infer.py <image_folder_path> <model_name>")
        sys.exit(1)
    
    image_folder    =   sys.argv[1]
    model_name      =   sys.argv[2]
    
    return image_folder, model_name



def load_images(folder_path):
    image_pattern   =   os.path.join(folder_path, '*.jpg')  
    image_files     =   glob.glob(image_pattern)
    return image_files


def call_model(model_name, img_path):
    function = globals().get(model_name)
    if callable(function):
        return function(img_path)
    else:
        return f"Function '{model_name}' not found."


def main():


    image_folder, model_name    =   get_arguments()
    images                      =   load_images(image_folder)

    for image in images:
        call_model(model_name, image)
        

    data            =   { "Result": [parameters] }
    json_file_path  =   "./output/val.json"
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    main()