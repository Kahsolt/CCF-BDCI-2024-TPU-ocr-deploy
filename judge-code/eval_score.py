import json
from difflib import SequenceMatcher
from argparse import ArgumentParser

from shapely.geometry import Polygon


def string_similar(s1, s2):
    return SequenceMatcher(None, s1, s2).quick_ratio()


def calculate_iou(poly1, poly2):
    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)
    if not polygon1.is_valid or not polygon2.is_valid:
        return 0.0
    if polygon1.intersects(polygon2):
        inter_area = polygon1.intersection(polygon2).area
    else:
        inter_area = 0
    union_area = polygon1.area + polygon2.area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def calculate_f_score(gt_data: dict, result_data: dict, iou_threshold: float = 0.5):
    gt_count = 0
    result_count = 0
    true_positive = 0

    for image_name, gt_list in gt_data.items():
        result_list = result_data.get(image_name, [])
        gt_count += len(gt_list)
        result_count += len(result_list)

        gt_matched = set()
        result_matched = set()
        for gt_idx, gt in enumerate(gt_list):
            if gt['illegibility'] or len(gt['points']) > 4 or gt['transcription'] == "###":
                gt_count -= 1
                continue
            for result_idx, result in enumerate(result_list):
                iou = calculate_iou(gt['points'], result['points'])
                if iou > iou_threshold and string_similar(gt['transcription'], result['transcription']) > 0.5 \
                    and result_idx not in result_matched:
                    gt_matched.add(gt_idx)
                    result_matched.add(result_idx)
                    true_positive += 1
                    break

    precision = true_positive / result_count if result_count > 0 else 0
    recall = true_positive / gt_count if gt_count > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f_score, precision, recall


def read_json_file(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


if __name__ == '__main__':
    parser = ArgumentParser(prog=__file__)
    parser.add_argument('--gt_path', type=str, default='./datasets/train_full_images_0.json', help='path of label json')
    parser.add_argument('--result_json', type=str, default='./results/ppocr_system_results.json', help='path of result json')
    parser.add_argument('--inference_time', type=float, default=0.0, help='inference time')
    opt = parser.parse_args()

    gt_file_path = opt.gt_path
    result_file_path = opt.result_json
    inference_time = float(opt.inference_time)

    gt_data = read_json_file(gt_file_path)
    result_data = read_json_file(result_file_path)

    f_score, precision, recall = calculate_f_score(gt_data, result_data)
    print(f"F-score: {f_score:.5f}, Precision: {precision:.5f}, Recall: {recall:.5f}")
    print(f"Inference time: {inference_time:.5f}")
    print(f"Score: {(min(100, 90+40*f_score-0.085*inference_time/160)):.5f}")
