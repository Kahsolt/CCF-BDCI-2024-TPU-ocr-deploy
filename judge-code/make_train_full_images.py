#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/26 

# 先下载完整的 train_full_labels.json，再过滤出选定子集
# 下载链接: https://aistudio.baidu.com/datasetdetail/177210

import os
import json
from argparse import ArgumentParser


if __name__ == '__main__':
  parser = ArgumentParser(prog=__file__)
  parser.add_argument('-I', type=str, default='../datasets/train_full_labels.json', help='path of input label json')
  parser.add_argument('-O', type=str, default='../datasets/train_full_images.json', help='path of output label json')
  parser.add_argument('--img_path', type=str, default='../datasets/train_full_images_0', help='input image directory path')
  args = parser.parse_args()

  with open(args.I, 'r', encoding='utf-8') as fh:
    data = json.load(fh)
  print('len(data):', len(data))

  data_filtered = {}
  fns = set(os.path.splitext(e)[0] for e in os.listdir(args.img_path))
  for k, v in data.items():
    if k not in fns: continue
    data_filtered[k] = [e for e in v if not e['illegibility'] and len(e['points']) == 4 and e['transcription'] != "###"]

  print('len(data_filtered):', len(data_filtered))
  with open(args.O, 'w', encoding='utf-8') as fh:
    json.dump(data_filtered, fh, indent=2, ensure_ascii=False)

  print('Done!')
