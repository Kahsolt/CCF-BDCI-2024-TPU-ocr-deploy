#!/usr/bin/env python3
# Author: Armit
# Create Time: 周四 2024/11/14 

# 推理结果可视化 (B榜无标注数据)

from vis_results import *


def draw_ocr_box_txt(image:Image.Image, pred:dict):
  h, w = image.height, image.width
  img_B = Image.new('RGB', (w, h), (255, 255, 255))

  random.seed(114514)
  draw_B = ImageDraw.Draw(img_B)
  draw_ocr_box_txt_aside(pred, draw_B)

  img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
  img_show.paste(image, (w * 0, 0, w * 1, h))
  img_show.paste(img_B, (w * 1, 0, w * 2, h))
  return np.asarray(img_show)


def run(args):
  dp_in = Path(args.input) ; assert dp_in.is_dir()
  with open(args.pred, 'r', encoding='utf-8') as fh:
    pred_annots: dict = json.load(fh)

  dp_out = Path(args.pred).with_suffix('')
  dp_out.mkdir(exist_ok=True)
  for name, pred in tqdm(pred_annots.items()):
    img = Image.open(dp_in / f'{name}.jpg').convert('RGB')
    draw_img = draw_ocr_box_txt(img, pred)
    cv2.imwrite(dp_out / f'{name}.jpg', draw_img[:, :, ::-1])


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-I', '--input', default='./datasets/MSRA_Photo',  help='input image directory path')
  parser.add_argument('-B', '--pred',  default='./results/results.json', help='pred annots')
  args = parser.parse_args()

  run(args)
