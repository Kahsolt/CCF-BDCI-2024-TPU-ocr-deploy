#!/usr/bin/env python3
# Author: Armit
# Create Time: 周四 2024/10/31 

# 将 results.txt 转换为比赛提交格式的 json 文件

import json
from pathlib import Path
from argparse import ArgumentParser

BASE_PATH = Path(__file__).parent
KEY_FILE = BASE_PATH / 'ppocr_keys_v1.txt'


def run(args):
  print(f'>> load vocab from {KEY_FILE}')
  with open(KEY_FILE, 'r', encoding='utf-8') as fh:
    vocab = fh.read().strip().split('\n')
    vocab.insert(0, None)
    vocab.append(' ')
  print('>> len(vocab):', len(vocab))

  print(f'>> load results from {args.fp}')
  with open(args.fp, 'r', encoding='utf-8') as fh:
    lines = fh.read().strip().split('\n')

  results = {}
  seg = []
  def handle_seg():
    annots = []
    results[seg[0].split('.')[0]] = annots
    for box_ids in seg[1:]:
      box, ids = box_ids.split('|')
      ids = ids.strip().split(' ') if ids.strip() else []
      if not ids and not args.keep_blank: continue
      box = [max(int(e), 0) for e in box.strip().split(' ')]
      ids = [int(e) for e in ids]
      annots.append({
        'transcription': ''.join([vocab[e] for e in ids]),
        'points': list(zip(box[::2], box[1::2])),
      })
    annots.sort(key=lambda e: e['transcription'])
    seg.clear()

  for line in lines:
    if not line:
      handle_seg()
    else:
      seg.append(line)
  if seg: handle_seg()

  fn_out = args.fp.with_suffix('.json')
  print(f'>> save results to {fn_out}')
  print('>> len(results):', len(results))
  with open(fn_out, 'w', encoding='utf-8') as fh:
    json.dump(results, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-F', '--fp', default='results.txt', type=Path, help='path to results.txt file')
  parser.add_argument('--keep_blank', action='store_true')
  args = parser.parse_args()

  run(args)
