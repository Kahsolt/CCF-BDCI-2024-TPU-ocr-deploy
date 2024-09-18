#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/17 

import json
from pathlib import Path
from typing import List, Dict

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'data'
REPO_PATH = BASE_PATH / 'repo'
IMG_PATH = BASE_PATH / 'img'
OUT_PATH = BASE_PATH / 'output'
PP_DATA_PATH = DATA_PATH/ 'ppocr_img'
DEFAULT_INPUT_FOLDER = PP_DATA_PATH / 'imgs'
DEFAULT_SAVE_FILE = OUT_PATH / 'val.json'
IMAGE_FILE_SUFFIX = ['.jpg', '.jpeg', '.png']
INFER_RESULT_FIELDS = [
  'Precision',
  'Recall',
  'F1-Score',
  'i_time',
]

InferResult = Dict[str, float]
InferResults = List[InferResult]

mean = lambda x: sum(x) / len(x) if len(x) else 0.0


def save_infer_results(results:InferResults, fp:Path=DEFAULT_SAVE_FILE):
  assert isinstance(results, list) and len(results), 'data should be a non-empty list'
  for it in results:
    for fld in INFER_RESULT_FIELDS:
      assert isinstance(it.get(fld), float), f'{fld} should be float type but got {type(it.get(fld))}'
  print('>> mean(time):', mean([it['i_time'] for it in results]))

  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump({"Result": results}, fh, indent=2, ensure_ascii=False)
  print(f'>> [save_infer_data] {fp}')
