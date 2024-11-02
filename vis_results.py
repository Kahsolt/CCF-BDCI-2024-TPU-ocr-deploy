#!/usr/bin/env python3
# Author: Armit
# Create Time: 周六 2024/11/02 

# 推理结果可视化

import math
import json
import random
from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

font_path = "./datasets/fonts/simfang.ttf"


def draw_ocr_box_txt_aside(annots:dict, canvas:ImageDraw.ImageDraw):
  for it in annots:
    box, txt = it['points'], it['transcription']
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    coords = np.asarray(box, dtype=int).flatten().tolist()
    canvas.polygon(coords, outline=color)
    box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][1])**2)
    box_width  = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
    if box_height > 2 * box_width:
      font_size = max(int(box_width * 0.9), 10)
      try:
        font = ImageFont.truetype(font_path, font_size)
      except OSError:
        print("无法打开字体文件，使用默认字体。")
        font = ImageFont.load_default()
      cur_y = box[0][1]
      for c in txt:
        try:    # pillow 9.5.0
          char_size = font.getsize(c)
        except AttributeError:  # pillow 10
          char_size = font.getbbox(c)[-2:]
        canvas.text((box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
        cur_y += char_size[1]
    else:
      font_size = max(int(box_height * 0.8), 10)
      try:
        font = ImageFont.truetype(font_path, font_size)
      except OSError:
        print("无法打开字体文件，使用默认字体。")
        font = ImageFont.load_default()
      canvas.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)


def draw_ocr_box_txt(image:Image.Image, gt:dict, pred:dict):
  h, w = image.height, image.width
  img_A = Image.new('RGB', (w, h), (255, 255, 255))
  img_B = Image.new('RGB', (w, h), (255, 255, 255))

  random.seed(114514)
  draw_A = ImageDraw.Draw(img_A)
  draw_B = ImageDraw.Draw(img_B)
  draw_ocr_box_txt_aside(gt,   draw_A)
  draw_ocr_box_txt_aside(pred, draw_B)

  img_show = Image.new('RGB', (w * 3, h), (255, 255, 255))
  img_show.paste(image, (w * 0, 0, w * 1, h))
  img_show.paste(img_A, (w * 1, 0, w * 2, h))
  img_show.paste(img_B, (w * 2, 0, w * 3, h))
  return np.asarray(img_show)


def run(args):
  dp_in = Path(args.input) ; assert dp_in.is_dir()
  with open(args.gt, 'r', encoding='utf-8') as fh:
    gt_annots: dict = json.load(fh)
  with open(args.pred, 'r', encoding='utf-8') as fh:
    pred_annots: dict = json.load(fh)

  dp_out = Path(args.pred).with_suffix('')
  dp_out.mkdir(exist_ok=True)
  for name, gt in tqdm(gt_annots.items()):
    img = Image.open(dp_in / f'{name}.jpg').convert('RGB')
    pred = pred_annots.get(name, [])
    draw_img = draw_ocr_box_txt(img, gt, pred)
    cv2.imwrite(dp_out / f'{name}.jpg', draw_img[:, :, ::-1])


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-I', '--input', default='./datasets/train_full_images_0',    help='input image directory path')
  parser.add_argument('-A', '--gt',    default='./datasets/train_full_images.json', help='gt annots')
  parser.add_argument('-B', '--pred',  default='./results/results.json',            help='pred annots')
  args = parser.parse_args()

  run(args)
