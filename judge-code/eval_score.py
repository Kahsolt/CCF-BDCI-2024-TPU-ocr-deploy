import json
from difflib import SequenceMatcher
from argparse import ArgumentParser

from tqdm import tqdm
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
        return 0.0
    union_area = polygon1.area + polygon2.area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def calculate_f_score(gt_data: dict, result_data: dict, iou_threshold: float = 0.5, sim_threshold: float = 0.5):
    gt_count = 0
    result_count = 0
    true_positive = 0
    perfect_rec_count = 0
    good_box_list = []  # 有得分的box

    for image_name, gt_list in tqdm(gt_data.items()):
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
                if result_idx in result_matched: continue
                iou = calculate_iou(gt['points'], result['points'])
                if iou <= iou_threshold: continue
                sim = string_similar(gt['transcription'], result['transcription'])
                if sim <= sim_threshold: continue 

                if abs(sim - 1) < 1e-8: perfect_rec_count += 1
                good_box_list.append(result['points'])

                gt_matched.add(gt_idx)
                result_matched.add(result_idx)
                true_positive += 1
                break

    def get_aspect_ratio(rect:Polygon):
        from shapely import get_point, distance
        p0 = get_point(rect.exterior, 0)
        p1 = get_point(rect.exterior, 1)
        p2 = get_point(rect.exterior, 2)
        d0 = distance(p0, p1)
        d1 = distance(p1, p2)
        aspect = min(d0, d1) / max(d0, d1)
        return aspect

    if not 'plot':  # 根据面积/长宽比都无法区分出更容易得分或出错的框，难以在此处优化 :(
        all_box_list = [b['points'] for it in result_data.values() for b in it]
        bad_box_list = [b for b in all_box_list if b not in good_box_list]
        good_box_area_list = [get_aspect_ratio(Polygon(b)) for b in good_box_list]
        bad_box_area_list  = [get_aspect_ratio(Polygon(b)) for b in bad_box_list]
        import matplotlib.pyplot as plt
        plt.scatter(good_box_area_list, [0] * len(good_box_area_list), c='b')
        plt.scatter(bad_box_area_list,  [0] * len(bad_box_area_list),  c='r')
        plt.show()

    # v2_det + mb_rec: 3518 / 9513 = 36.980973404814466
    print(f'>> Perfect recognize {perfect_rec_count} / {result_count} = {perfect_rec_count / result_count:%}')

    precision = true_positive / result_count if result_count > 0 else 0
    recall = true_positive / gt_count if gt_count > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f_score, precision, recall


def read_json_file(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


if __name__ == '__main__':
    parser = ArgumentParser(prog=__file__)
    parser.add_argument('--gt_path', type=str, default='../datasets/train_full_images.json', help='path of label json')
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
    print(f"Score: {(90+40*f_score-0.085*inference_time):.5f}")
