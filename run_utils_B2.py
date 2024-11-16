#!/usr/bin/env python3
# Author: Armit
# Create Time: 周六 2024/11/16 

# 神tm改 B榜 数据集了！！这是针对 B 榜数据集的专用小工具！！
# - resize: B2榜数据尺寸很大，我们提前缩小一下 (否则TF上放不下!!)
# - resfix: 重采样后的预测位置还原为原图的坐标位置

import json
from shutil import copy
from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm
from PIL import Image, ImageFilter
import numpy as np
from imagesize import get as get_imagesize

CANVAS_SIZE = 640


def abla_filters():
  import matplotlib.pyplot as plt

  fp = r'datasets\test_images\image_965.jpg'
  img = Image.open(fp).convert('RGB')
  w, h = 640, 640

  pre_filters = {
    'GaussianBlur': ImageFilter.GaussianBlur(radius=5),
    'SMOOTH':       ImageFilter.SMOOTH,
    'SMOOTH_MORE':  ImageFilter.SMOOTH_MORE,
  }
  post_filters = {
    'SHARPEN':           ImageFilter.SHARPEN,
    'EDGE_ENHANCE':      ImageFilter.EDGE_ENHANCE,
    'EDGE_ENHANCE_MORE': ImageFilter.EDGE_ENHANCE_MORE,
  }
  i = 0
  plt.figure(figsize=[10, 10])
  for n1, blur in pre_filters.items():
    for n2, sharp in post_filters.items():
      im = img.filter(blur)
      im = im.resize((w, h), Image.Resampling.LANCZOS)
      im = im.filter(sharp)

      i += 1
      plt.subplot(330+i)
      plt.imshow(im)
      plt.title(f'{n1} + {n2}')
      plt.axis('off')
  plt.tight_layout()
  plt.show()


def run_resize(args):
  dp_out: Path = args.R
  print(f'>> saving to {dp_out}')
  dp_out.mkdir(exist_ok=True)

  if args.use_filter:
    blur  = ImageFilter.SMOOTH
    sharp = ImageFilter.SHARPEN
  fps = list(Path(args.I).iterdir())
  for fp in tqdm(fps):
    fp_out = dp_out / fp.name
    img = Image.open(fp).convert('RGB')

    # 小图不做处理，直接拷贝
    w, h = img.size
    size_max = max(w, h)
    if size_max <= CANVAS_SIZE:
      copy(fp, fp_out)
      continue

    # 大图降采样：短边压倒 CANVAS_SIZE
    det_r = size_max / CANVAS_SIZE
    w = int(w / det_r)
    h = int(h / det_r)

    # 重采样的同时滤波处理，防止图变得太糊或引入锯齿噪声
    if args.use_filter:
      img = img.filter(blur)
      img = img.resize((w, h), Image.Resampling.LANCZOS)
      img = img.filter(sharp)
    else:
      img = img.resize((w, h), Image.Resampling.LANCZOS)

    if CANVAS_SIZE > w or CANVAS_SIZE > h:
      im = np.asarray(img)  # HWC
      im = np.pad(im, [(0, CANVAS_SIZE - h), (0, CANVAS_SIZE - w), (0, 0)], constant_values=0)
      img = Image.fromarray(im)

    img.save(fp_out, quality=100)   # disable jpeg quantization!


def run_fixres(args):
  with open(args.r, 'r', encoding='utf-8') as fh:
    annot_dict: dict = json.load(fh)

  for id, annots in tqdm(annot_dict.items()):
    fp = args.I / f'{id}.jpg'
    size = get_imagesize(fp)
    max_size = max(size)
    if max_size < CANVAS_SIZE: continue
    det_r = max_size / CANVAS_SIZE
    for annot in annots:
      annot['points'] = np.round(np.asarray(annot['points']) * det_r).astype(np.int32).tolist()

  fp_out: Path = args.r.with_stem(f'{args.r.stem}_fix')
  with open(fp_out, 'w', encoding='utf-8') as fh:
    json.dump(annot_dict, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--resize_debug', action='store_true')
  parser.add_argument('--resize', action='store_true')
  parser.add_argument('--fixres', action='store_true')
  parser.add_argument('--use_filter', action='store_true')
  parser.add_argument('-I', type=Path, default='datasets/test_images',         help='original image folder')
  parser.add_argument('-R', type=Path, default='datasets/test_images_640x640', help='resampled image folder')
  parser.add_argument('-r', type=Path, default='results/resB2_640.json',       help='converted infer results.json')
  args = parser.parse_args()

  assert args.I.is_dir()

  if args.resize_debug:
    abla_filters()

  if args.resize:
    run_resize(args)

  if args.fixres:
    assert args.R.is_dir()
    assert args.r.is_file()
    run_fixres(args)
