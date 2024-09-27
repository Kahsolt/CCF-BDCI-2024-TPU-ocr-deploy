#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/17 

import json
from pathlib import Path
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union

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
LABEL_FILE = DATA_PATH / 'train_full_labels.json'
PROCESSED_LABEL_FILE = LABEL_FILE.with_suffix('.jsonl')

mean = lambda x: sum(x) / len(x) if len(x) else 0.0
safe_div = lambda x, y: x / y if y > 0 else 0.0


''' Annots '''

# ↓↓↓ 存储时结构
BBox = List[Tuple[int, int]]
Annot = Dict[str, Union[str, BBox]]     # keys: transcription, points
Annots = Dict[str, List[Annot]]         # keys: sample_id
Metrics = Dict[str, Dict[str, float]]   # keys: sample_id; subkeys: Precision, Recall, F1-Score, Runtime

# ↓↓↓ 运行时结构
@dataclass
class InferAnnot:
  text: str
  bbox: BBox
@dataclass
class InferResult:
  annots: List[InferAnnot]
  runtime: float
InferResults = Dict[str, InferResult]

def preprocess_annots(fp:Path=None):
  fp = fp or LABEL_FILE
  assert fp.exists(), '>> You should first download the file "train_full_labels.json"'

  with open(fp, 'r', encoding='utf-8') as fh:
    data = json.load(fh)
  # len(data): 30000
  print('len(data):', len(data))
  #mean(n_bbox) before: 12.753533333333333
  print('mean(n_bbox) before:', mean([len(v) for v in data.values()]))

  items = []
  for k, v in data.items():   # dict -> list
    annot_list = []
    for gt in v:
      if gt['illegibility'] or len(gt['points']) > 4 or gt['transcription'] == "###":
        continue
      annot_list.append({
        'transcription': gt['transcription'],
        'points': gt['points'],
      })
    items.append({
      'id': k,
      'annots': annot_list,
    })
  items.sort(key=lambda e: int(e['id'].split('_')[-1]))
  # mean(n_bbox) after: 8.003933333333332
  print('mean(n_bbox) after:', mean([len(it['annots']) for it in items]))

  save_fp = fp.with_suffix('.jsonl')
  print(f'>> write to cache file {save_fp}')
  with open(save_fp, 'w', encoding='utf-8') as fh:
    for it in items:
      fh.write(json.dumps(it, indent=None, ensure_ascii=False))
      fh.write('\n')

def load_annots(fp:Path=None) -> Annots:
  fp = fp or PROCESSED_LABEL_FILE
  if not fp.is_file():
    preprocess_annots(fp.with_suffix('.json'))

  annots: Annots = {}
  with open(fp, 'r', encoding='utf-8') as fh:
    for line in fh.readlines():
      if not line: continue
      data = json.loads(line.strip())
      annots[data['id']] = data['annots']  # list -> dict
  return annots


''' Results '''

def infer_results_to_metrics(results:InferResults, fp:Path=None) -> Metrics:
  annots = load_annots(fp)
  metrics: Metrics = {}
  for id, annot in annots.items():
    if id not in results: continue
    r = results[id]
    annot_pred = [{
      'transcription': it.text,
      'points': it.bbox,
    } for it in r.annots]
    f_score, precision, recall = calc_f1({id: annot_pred}, {id: annot})
    metrics[id] = {
      'Precision': precision,
      'Recall': recall,
      'F1-Score': f_score,
      'Runtime': r.runtime,
    }
  return metrics

def save_infer_results(results:InferResults, fp:Path=None):
  fp = fp or DEFAULT_SAVE_FILE
  assert isinstance(results, dict) and len(results), '>> infer_results should NOT be empty dict!'
  data = {
    k: [{
      'transcription': e.text,
      'points': e.bbox,
    } for e in v.annots]
    for k, v in results.items()
  }
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False)
  print(f'>> [save_infer_results] {fp}')

def save_metrics(metrics:Metrics, fp:Path=None):
  fp = fp or DEFAULT_SAVE_FILE.with_suffix('.metrics.json')
  assert isinstance(metrics, dict) and len(metrics), '>> metrics should NOT be empty dict!'
  f1 = mean([e['F1-Score'] for e in metrics.values()])
  ts = mean([e['Runtime']  for e in metrics.values()])
  print('mean(f1):',      f1)
  print('mean(runtime):', ts)
  print('Estimated score:',          min(100, 90 + 40 * f1 - 0.085 * ts))
  print('Estimated score (scaled):', min(100, 90 + 40 * f1 - 0.085 * ts / 160))

  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(metrics, fh, indent=2, ensure_ascii=False)
  print(f'>> [save_infer_metrics] {fp}')


''' Metrics '''

def calc_sim(s1:str, s2:str):
  return SequenceMatcher(None, s1, s2).quick_ratio()

def calc_iou(box1:BBox, box2:BBox) -> float:
  poly1, poly2 = Polygon(box1), Polygon(box2)
  if not poly1.is_valid: return 0.0   # poly1 is pred
  inter = poly1.intersection(poly2).area if poly1.intersects(poly2) else 0
  if inter == 0: return 0.0
  union = poly1.area + poly2.area - inter
  return safe_div(inter, union)

def calc_f1(preds:Annots, truths:Annots, iou_thresh:float=0.5, sim_thresh:float=0.5):
  assert len(preds) == len(truths), f'>> sample count mismatch: {len(preds)} != {len(truths)}'

  pred_annot_cnt = 0
  truth_annot_cnt = 0
  TP = 0

  for id, truth_annot_list in (tqdm if len(preds) > 1 else (lambda _: _))(truths.items()):
    pred_annot_list = preds.get(id, [])
    pred_annot_cnt  += len(pred_annot_list)
    truth_annot_cnt += len(truth_annot_list)

    pred_matched = set()
    truth_matched = set()
    for truth_idx, truth_annot in enumerate(truth_annot_list):
      for pred_idx, pred_annot in enumerate(pred_annot_list):
        iou = calc_iou(pred_annot['points'], truth_annot['points'])
        if iou <= iou_thresh: continue
        sim = calc_sim(pred_annot['transcription'], truth_annot['transcription'])
        if sim <= sim_thresh: continue

        if pred_idx not in pred_matched:
          pred_matched .add(pred_idx)
          truth_matched.add(truth_idx)
          TP += 1
          break

  precision = safe_div(TP, pred_annot_cnt)
  recall    = safe_div(TP, truth_annot_cnt)
  f_score   = safe_div(2 * precision * recall, precision + recall)
  return f_score, precision, recall

def calc_score(preds:Annots, truths:Annots, infer_time:float, is_real_chip:bool=True) -> float:
  f_score, precision, recall = calc_f1(preds, truths)
  print(f"F-score: {f_score:.5f}, Precision: {precision:.5f}, Recall: {recall:.5f}")
  print(f"Inference time: {infer_time:.5f}")
  print(f"Score: {(min(100, 90 + 40 * f_score - 0.085 * infer_time / (1 if is_real_chip else 160))):.5f}")


if __name__ == '__main__':
  annots = load_annots()
  calc_score(annots, annots, infer_time=500, is_real_chip=False)    # ~1min50s
