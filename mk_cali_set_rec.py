#!/usr/bin/env python3
# Author: Armit
# Create Time: 周四 2024/10/31 

# 变长模型 rec 在推理时是右填充 255，calib 时不能暴力 resize 倒满

from pathlib import Path
from argparse import ArgumentParser

from PIL import Image
import numpy as np


def run(args):
  dp_out: Path = args.O or args.I.with_name(f'{args.I.name}_{args.H}x{args.W}')
  print(f'>> saving to {dp_out}')
  dp_out.mkdir(exist_ok=True)

  for fp in Path(args.I).iterdir():
    img = Image.open(fp).convert('RGB')
    w, h = img.size
    if h != args.H:
      w /= h / args.H
      w = round(w)
      h = args.H
    w = min(w, args.W)
    img = img.resize((w, h), Image.Resampling.BILINEAR)

    if w < args.W:
      im = np.asarray(img)  # HWC
      im = np.pad(im, [(0, 0), (0, args.W - w), (0, 0)], constant_values=255)
      img = Image.fromarray(im)

    img.save(dp_out / fp.name)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-I', type=Path, default='datasets/cali_set_rec')
  parser.add_argument('-O', type=Path)
  parser.add_argument('-H', default=32,  type=int)
  parser.add_argument('-W', default=320, type=int)
  args = parser.parse_args()

  assert args.I.is_dir()

  run(args)
