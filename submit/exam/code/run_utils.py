#!/usr/bin/env python3
# Author: Armit
# Create Time: 周六 2024/11/16 

# 这是一些前后处理小工具，用法参考 README.md :)

import json
from shutil import copy
from pathlib import Path
from argparse import ArgumentParser

from PIL import Image, ImageDraw, ImageFont
try:  # try to compact with SoC
  from tqdm import tqdm
  import numpy as np
  from imagesize import get as get_imagesize
except:
  np = None
  tqdm = lambda _: _

FONT_PATH = "simsun.ttc"
CANVAS_SIZE = 640


def run_resize(args):
  print(f'>> saving to {args.O}')
  args.O.mkdir(exist_ok=True, parents=True)

  fps = list(Path(args.I).iterdir())
  for fp in tqdm(fps):
    img = Image.open(fp).convert('RGB')
    fp_out = args.O / fp.name

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
    img = img.resize((w, h), Image.Resampling.LANCZOS)

    # disable jpeg quantization!
    img.save(fp_out, quality=100)


def run_cvtres(args):
  print(f'>> load raw results from {args.I}')
  with open(args.I, 'r', encoding='utf-8') as fh:
    lines = fh.read().strip().split('\n')

  print(f'>> load vocab from {args.R}')
  with open(args.R, 'r', encoding='utf-8') as fh:
    vocab = fh.read().strip().split('\n')
    vocab.insert(0, None)
    vocab.append(' ')
  print('>> len(vocab):', len(vocab))

  results = {}
  seg = []
  def handle_seg():
    annots = []
    results[seg[0].split('.')[0]] = annots
    for box_ids in seg[1:]:
      box, ids = box_ids.split('|')
      ids = ids.strip().split(' ') if ids.strip() else []
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

  print(f'>> save converted results to {args.O}')
  print('>> len(results):', len(results))
  with open(args.O, 'w', encoding='utf-8') as fh:
    json.dump(results, fh, indent=2, ensure_ascii=False)


def run_fixres(args):
  print(f'>> load converted results from {args.I}')
  with open(args.I, 'r', encoding='utf-8') as fh:
    annots_dict: dict = json.load(fh)

  for id, annots in tqdm(annots_dict.items()):
    size = get_imagesize(args.R / f'{id}.jpg')
    max_size = max(size)
    if max_size < CANVAS_SIZE: continue
    det_r = max_size / CANVAS_SIZE
    for annot in annots:
      annot['points'] = np.round(np.asarray(annot['points']) * det_r).astype(np.int32).tolist()

  print(f'>> save fixed results to {args.O}')
  with open(args.O, 'w', encoding='utf-8') as fh:
    json.dump(annots_dict, fh, indent=2, ensure_ascii=False)


def draw_ocr_box_txt(image:Image.Image, annots:dict) -> Image.Image:
  h, w = image.height, image.width
  img_annot = Image.new('RGB', (w, h), (255, 255, 255))
  cvs = ImageDraw.Draw(img_annot)

  np.random.seed(114514)
  for it in annots:
    box, txt = it['points'], it['transcription']
    color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    coords = np.asarray(box, dtype=int).flatten().tolist()
    cvs.polygon(coords, outline=color)
    box_height = np.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][1])**2)
    box_width  = np.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
    if box_height > 2 * box_width:
      font_size = max(int(box_width * 0.9), 10)
      try:
        font = ImageFont.truetype(FONT_PATH, font_size, encoding='unic')
      except OSError:
        print("无法打开字体文件，使用默认字体。")
        font = ImageFont.load_default()
      cur_y = box[0][1]
      for c in txt:
        try:    # pillow 9.5.0
          char_size = font.getsize(c)
        except AttributeError:  # pillow 10
          char_size = font.getbbox(c)[-2:]
        cvs.text((box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
        cur_y += char_size[1]
    else:
      font_size = max(int(box_height * 0.8), 10)
      try:
        font = ImageFont.truetype(FONT_PATH, font_size, encoding='unic')
      except OSError:
        print("无法打开字体文件，使用默认字体。")
        font = ImageFont.load_default()
      cvs.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)

  img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
  img_show.paste(image,     (w * 0, 0, w * 1, h))
  img_show.paste(img_annot, (w * 1, 0, w * 2, h))
  return img_show

def run_visres(args):
  print(f'>> load converted / fixed results from {args.I}')
  with open(args.I, 'r', encoding='utf-8') as fh:
    annots_dict: dict = json.load(fh)

  print(f'>> saving to folder {args.O}')
  args.O.mkdir(exist_ok=True, parents=True)
  for id, annots in tqdm(annots_dict.items()):
    img = Image.open(args.R / f'{id}.jpg').convert('RGB')
    draw_img = draw_ocr_box_txt(img, annots)
    draw_img.save(args.O / f'{id}.jpg')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--resize', action='store_true')
  parser.add_argument('--cvtres', action='store_true')
  parser.add_argument('--fixres', action='store_true')
  parser.add_argument('--visres', action='store_true')
  parser.add_argument('-I', type=Path, help='input file or folder path')
  parser.add_argument('-O', type=Path, help='output file or folder path')
  parser.add_argument('-R', type=Path, help='reference file or folder path')
  args = parser.parse_args()

  tasks = [
    args.resize,
    args.cvtres,
    args.fixres,
    args.visres,
  ]
  assert sum(tasks) == 1, 'must specify exactly 1 task'

  I: Path = args.I
  O: Path = args.O
  R: Path = args.R

  if args.resize:
    print('>> [Resize]')
    assert I.is_dir(), 'input should be original image folder'
    if O.exists(): print(f'>> overwriting {O}')
    run_resize(args)

  if args.cvtres:
    print('>> [CvtRes]')
    assert I.is_file(), 'input should be raw results.txt'
    if O.exists(): print(f'>> overwriting {O}')
    assert R.is_file(), 'reference should be key file'
    run_cvtres(args)

  if args.fixres:
    print('>> [FixRes]')
    assert I.is_file(), 'input should be converted results.json'
    if O.exists(): print(f'>> overwriting {O}')
    assert R.is_dir(), 'reference should be original image folder'
    run_fixres(args)

  if args.visres:
    print('>> [VisRes]')
    assert I.is_file(), 'reference should be converted / fixed results.json'
    if O.exists(): print(f'>> overwriting {O}')
    assert R.is_dir(), 'input should be original / downsampled image folder'
    run_visres(args)
