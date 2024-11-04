#!/usr/bin/env python3
# Author: Armit
# Create Time: 周一 2024/11/04

# 推理结果可视化 (仅空白识别框，需配合 `convert_results.py --keep_blank`)

import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def run(args):
  dp_in = Path(args.input) ; assert dp_in.is_dir()
  with open(args.pred, 'r', encoding='utf-8') as fh:
    annots_dict: dict = json.load(fh)

  annots_dict = {k: [e for e in v if not len(e['transcription'])] for k, v in annots_dict.items()}
  annots_dict = {k: v for k, v in annots_dict.items() if v}
  print('n_img:', len(annots_dict))

  dp_out = Path(args.pred).with_suffix('.blank')
  dp_out.mkdir(exist_ok=True)
  for name, annots in tqdm(annots_dict.items()):
    img = Image.open(dp_in / f'{name}.jpg').convert('RGB')
    draw = ImageDraw.Draw(img)
    for it in annots:
      coords = np.asarray(it['points'], dtype=int).flatten().tolist()
      draw.polygon(coords, outline=(0, 255, 255), width=6)
    img.save(dp_out / f'{name}.jpg')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-I', '--input', default='./datasets/train_full_images_0', help='input image directory path')
  parser.add_argument('-B', '--pred',  default='./results/results.json',         help='pred annots')
  args = parser.parse_args()

  run(args)
