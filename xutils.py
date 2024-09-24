#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/17 

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

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
LABEL_FILE = DATA_PATH / 'train_full_labels.json'
PROCESSED_LABEL_FILE = LABEL_FILE.with_suffix('.jsonl')

InferResult = Dict[str, float]
InferResults = List[InferResult]

mean = lambda x: sum(x) / len(x) if len(x) else 0.0

@dataclass
class BBox:
  text: str
  bbox: List[Tuple[int]]

@dataclass
class Annot:
  id: int
  bbox_list: List[BBox]

def preprocess_annots(fp:Path=LABEL_FILE):
  assert fp.exists(), '>> You should first download the file "train_full_labels.json"'

  with open(fp, 'r', encoding='utf-8') as fh:
    data = json.load(fh)
  # len(data): 30000
  print('len(data):', len(data))
  #mean(n_bbox) before: 12.753533333333333
  print('mean(n_bbox) before:', mean([len(v) for v in data.values()]))

  items = []
  for k, v in data.items():
    bbox_list = []
    for gt in v:
      if gt['illegibility'] or len(gt['points']) > 4 or gt['transcription'] == "###":
        continue
      bbox_list.append({
        'text': gt['transcription'],
        'bbox': gt['points'],
      })
    items.append({
      'id': k,
      'bbox_list': bbox_list,
    })
  items.sort(key=lambda e: int(e['id'].split('_')[-1]))
  # mean(n_bbox) after: 8.003933333333332
  print('mean(n_bbox) after:', mean([len(it['bbox_list']) for it in items]))

  save_fp = fp.with_suffix('.jsonl')
  print(f'>> write to cache file {save_fp}')
  with open(save_fp, 'w', encoding='utf-8') as fh:
    for it in items:
      fh.write(json.dumps(it, indent=None, ensure_ascii=False))
      fh.write('\n')

def load_annots(fp:Path=PROCESSED_LABEL_FILE) -> List[Annot]:
  if not fp.is_file():
    preprocess_annots(fp.with_suffix('.json'))

  annots: List[Annot] = []
  with open(fp, 'r', encoding='utf-8') as fh:
    for line in fh.readlines():
      if not line: continue
      data = json.loads(line.strip())
      annot = Annot(data['id'], [BBox(bbox['text'], bbox['bbox']) for bbox in data['bbox_list']])
      annots.append(annot)
  return annots


def save_infer_results(results:InferResults, fp:Path=DEFAULT_SAVE_FILE):
  assert isinstance(results, list) and len(results), 'data should be a non-empty list'
  for it in results:
    for fld in INFER_RESULT_FIELDS:
      assert isinstance(it.get(fld), float), f'{fld} should be float type but got {type(it.get(fld))}'
  print('>> mean(time):', mean([it['i_time'] for it in results]))

  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump({"Result": results}, fh, indent=2, ensure_ascii=False)
  print(f'>> [save_infer_data] {fp}')


if __name__ == '__main__':
  annots = load_annots()
