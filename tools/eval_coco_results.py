# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluate COCO-format detection results against the ground truth.

The evaluation follows mmdetection's ``CocoMetric``
(``mmdet/evaluation/metrics/coco_metric.py``): pycocotools ``COCOeval``
with ``maxDets=[100, 300, 1000]`` and IoU thresholds ``0.5:0.05:0.95``,
so the numbers are directly comparable with the official model zoo.

Run it from the mmdetection root so that ``mmdet`` is importable, e.g.::

    PYTHONPATH=. python tools/deployment/eval_coco_results.py \
        --ann-file /home/heqi/dataset/coco/annotations/instances_val2017.json \
        runs/rtmdet/results/rtmdet_tiny_end2end_val200.json

    # compare the ONNX results with a reference json, e.g. produced by
    # mmdet tools/test.py (CocoMetric)
    PYTHONPATH=. python tools/deployment/eval_coco_results.py \
        --ann-file /home/heqi/dataset/coco/annotations/instances_val2017.json \
        runs/rtmdet/results/rtmdet_tiny_end2end_val200.json \
        --ref-json runs/rtmdet/results/mmdet_tiny_val200.json \
        --restrict-to-results
"""
import argparse
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.evaluation import CocoMetric

# mapping of cocoEval.stats, same as mmdet CocoMetric
COCO_METRIC_NAMES = {
    'mAP': 0,
    'mAP_50': 1,
    'mAP_75': 2,
    'mAP_s': 3,
    'mAP_m': 4,
    'mAP_l': 5,
    'AR@100': 6,
    'AR@300': 7,
    'AR@1000': 8,
    'AR_s@1000': 9,
    'AR_m@1000': 10,
    'AR_l@1000': 11
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate COCO detection results with the same protocol '
        'as mmdet CocoMetric')
    parser.add_argument(
        'result_json', help='detection results in COCO result format')
    parser.add_argument(
        '--ann-file', required=True, help='COCO ground truth json')
    parser.add_argument(
        '--ref-json',
        help='optional reference results json to compare against')
    parser.add_argument(
        '--classwise',
        action='store_true',
        help='print a per-category AP table')
    parser.add_argument(
        '--restrict-to-results',
        action='store_true',
        help='only evaluate the images present in the results json, useful '
        'for subset runs; by default all ground truth images are evaluated '
        'like mmdet does')
    return parser.parse_args()


def evaluate(result_json, ann_file, restrict_to_results=False):
    """Run the mmdet CocoMetric bbox evaluation protocol."""
    coco_gt = COCO(ann_file)
    with open(result_json) as f:
        preds = json.load(f)
    if len(preds) == 0:
        raise RuntimeError(f'empty results: {result_json}')
    coco_dt = coco_gt.loadRes(preds)

    # use CocoMetric only to get the official defaults (proposal_nums and
    # iou_thrs), the evaluation below mirrors its code path.
    metric = CocoMetric(ann_file=None, metric='bbox')

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.catIds = sorted(int(c) for c in coco_gt.getCatIds())
    if restrict_to_results:
        coco_eval.params.imgIds = sorted({int(p['image_id']) for p in preds})
    else:
        coco_eval.params.imgIds = sorted(int(i) for i in coco_gt.getImgIds())
    coco_eval.params.maxDets = list(metric.proposal_nums)
    coco_eval.params.iouThrs = metric.iou_thrs

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval, coco_gt, metric


def format_metrics(coco_eval):
    ap = coco_eval.stats
    lines = [
        'bbox_mAP_copypaste: '
        f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} {ap[4]:.3f} '
        f'{ap[5]:.3f}'
    ]
    for name, idx in COCO_METRIC_NAMES.items():
        lines.append(f'{name:<10} {ap[idx]:.3f}')
    return lines


def print_classwise(coco_eval, coco_gt):
    """Per-category AP table, same computation as mmdet CocoMetric."""
    cat_ids = [int(c) for c in coco_eval.params.catIds]
    precisions = coco_eval.eval['precision']
    assert len(cat_ids) == precisions.shape[2]

    table_data = [[
        'category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
    ]]
    for idx, cat_id in enumerate(cat_ids):
        name = coco_gt.loadCats(cat_id)[0]['name']
        row = [name]
        for iou_idx in [None, 0, 5]:
            if iou_idx is None:
                precision = precisions[:, :, idx, 0, -1]
            else:
                precision = precisions[iou_idx, :, idx, 0, -1]
            precision = precision[precision > -1]
            row.append(
                round(float(precision.mean()), 3) if precision.
                size else float('nan'))
        for area_idx in [1, 2, 3]:
            precision = precisions[:, :, idx, area_idx, -1]
            precision = precision[precision > -1]
            row.append(
                round(float(precision.mean()), 3) if precision.
                size else float('nan'))
        table_data.append(row)

    col_widths = [
        max(len(str(row[col])) for row in table_data)
        for col in range(len(table_data[0]))
    ]
    for row in table_data:
        print('  '.join(
            str(cell).ljust(width) for cell, width in zip(row, col_widths)))


def main():
    args = parse_args()
    print(f'== result: {args.result_json}')
    coco_eval, coco_gt, _ = evaluate(args.result_json, args.ann_file,
                                     args.restrict_to_results)
    for line in format_metrics(coco_eval):
        print(line)
    if args.classwise:
        print('== per-category AP ==')
        print_classwise(coco_eval, coco_gt)

    if args.ref_json is not None:
        print(f'== reference: {args.ref_json}')
        ref_eval, _, _ = evaluate(args.ref_json, args.ann_file,
                                  args.restrict_to_results)
        for line in format_metrics(ref_eval):
            print(f'ref {line}')
        print('== comparison (result - reference) ==')
        header = f'{"metric":<10} {"result":>8} {"ref":>8} {"diff":>8}'
        print(header)
        for name, idx in COCO_METRIC_NAMES.items():
            result = coco_eval.stats[idx]
            ref = ref_eval.stats[idx]
            print(f'{name:<10} {result:>8.3f} {ref:>8.3f} '
                  f'{result - ref:>+8.3f}')


if __name__ == '__main__':
    main()
