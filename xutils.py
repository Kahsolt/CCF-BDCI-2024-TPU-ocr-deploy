#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/17 

import json
from pathlib import Path
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import List, Dict, Tuple

from tqdm import tqdm
from shapely.geometry import Polygon

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'data'
REPO_PATH = BASE_PATH / 'repo'
IMG_PATH = BASE_PATH / 'img'
OUT_PATH = BASE_PATH / 'output'
DEMO_PATH = BASE_PATH / 'ppocr'
PP_DATA_PATH = DATA_PATH/ 'ppocr_img'
DEFAULT_INPUT_FOLDER = DEMO_PATH / 'datasets' / 'train_full_images_0'
DEFAULT_SAVE_FILE = OUT_PATH / 'val.json'
INFER_RESULT_FIELDS = [
  'Precision',
  'Recall',
  'F1-Score',
  'i_time',
]
LABEL_FILE = DATA_PATH / 'train_full_labels.json'
PROCESSED_LABEL_FILE = LABEL_FILE.with_suffix('.jsonl')

Point = Tuple[int, int]
Points = List[Point]
InferResult = Dict[str, float]
InferResults = List[InferResult]

mean = lambda x: sum(x) / len(x) if len(x) else 0.0
safe_div = lambda x, y: x / y if y > 0 else 0.0


''' Annots '''

@dataclass
class BBox:
  text: str
  bbox: List[Tuple[int]]

Annots = Dict[str, List[BBox]]

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

def load_annots(fp:Path=PROCESSED_LABEL_FILE) -> Annots:
  if not fp.is_file():
    preprocess_annots(fp.with_suffix('.json'))

  annots: Annots = {}
  with open(fp, 'r', encoding='utf-8') as fh:
    for line in fh.readlines():
      if not line: continue
      data = json.loads(line.strip())
      annots[data['id']] = [BBox(bbox['text'], bbox['bbox']) for bbox in data['bbox_list']]
  return annots


''' Results '''

def save_infer_results(results:InferResults, fp:Path=DEFAULT_SAVE_FILE):
  assert isinstance(results, list) and len(results), 'data should be a non-empty list'
  for it in results:
    for fld in INFER_RESULT_FIELDS:
      assert isinstance(it.get(fld), float), f'{fld} should be float type but got {type(it.get(fld))}'
  print('>> mean(f1):',   mean([it['F1-Score'] for it in results]))
  print('>> mean(time):', mean([it['i_time']   for it in results]))

  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump({"Result": results}, fh, indent=2, ensure_ascii=False)
  print(f'>> [save_infer_data] {fp}')


''' Metrics '''

def calc_sim(s1:str, s2:str):
  return SequenceMatcher(None, s1, s2).quick_ratio()

def calc_iou(pts1:Points, pts2:Points) -> float:
  poly1, poly2 = Polygon(pts1), Polygon(pts2)
  if not poly1.is_valid: return 0.0   # poly1 is pred
  inter = poly1.intersection(poly2).area if poly1.intersects(poly2) else 0
  if inter == 0: return 0.0
  union = poly1.area + poly2.area - inter
  return safe_div(inter, union)

def calc_f1(preds:Annots, truths:Annots, iou_thresh:float=0.5, sim_thresh:float=0.5):
  assert len(preds) == len(truths), f'>> sample count mismatch: {len(preds)} != {len(truths)}'

  pred_bbox_cnt = 0
  truth_bbox_cnt = 0
  TP = 0

  for id, truth_bbox_list in (tqdm if len(preds) > 1 else (lambda _: _))(truths.items()):
    pred_bbox_list = preds.get(id, [])
    pred_bbox_cnt  += len(pred_bbox_list)
    truth_bbox_cnt += len(truth_bbox_list)

    pred_matched = set()
    truth_matched = set()
    for truth_idx, truth_bbox in enumerate(truth_bbox_list):
      for pred_idx, pred_bbox in enumerate(pred_bbox_list):
        iou = calc_iou(pred_bbox.bbox, truth_bbox.bbox)
        if iou <= iou_thresh: continue
        sim = calc_sim(pred_bbox.text, truth_bbox.text)
        if sim <= sim_thresh: continue

        if pred_idx not in pred_matched:
          pred_matched .add(pred_idx)
          truth_matched.add(truth_idx)
          TP += 1
          break

  precision = safe_div(TP, pred_bbox_cnt)
  recall    = safe_div(TP, truth_bbox_cnt)
  f_score   = safe_div(2 * precision * recall, precision + recall)
  return f_score, precision, recall

def calc_score(preds:Annots, truths:Annots, infer_time:float, is_real_chip:bool=True) -> float:
  f_score, precision, recall = calc_f1(preds, truths)
  print(f"F-score: {f_score:.5f}, Precision: {precision:.5f}, Recall: {recall:.5f}")
  print(f"Inference time: {infer_time:.5f}")
  print(f"Score: {(min(100, 90 + 40 * f_score - 0.085 * infer_time / (1 if is_real_chip else 160))):.5f}")


if __name__ == '__main__':
  annots = load_annots()
  calc_score(annots, annots, infer_time=500, is_real_chip=False)
