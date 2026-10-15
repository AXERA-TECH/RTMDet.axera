#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Standalone RTMDet ONNX inference with preprocessing, post-processing and
optional visualization.

Supported models (exported by ``tools/deployment/rtmdet2onnx.py``):

- raw mode: outputs ``cls_score_{i}`` (logits) and ``bbox_pred_{i}``
  (pixel distances) for every feature level. The script performs sigmoid,
  top-k filtering, distance decoding, class-wise NMS and ``max_per_img``.
- end2end mode: outputs ``dets`` (N, 5) and ``labels`` (N,); only a score
  filter and rescaling are applied.

The exported graphs do not contain the data normalization by default, so
this script resizes (keeping the aspect ratio), pads with 114, converts
the image to a float32 BGR tensor and applies ``(img - mean) / std``
before feeding it to the network. The mean/std can be changed with
``--mean``/``--std``; pass ``--no-normalize`` for models exported with
``--normalize`` (normalization baked into the graph).

Examples:
    python tools/deployment/rtmdet_onnx_infer.py \
        runs/rtmdet/rtmdet_tiny.onnx demo/demo.jpg \
        --vis runs/rtmdet/vis/demo_raw.jpg

    python tools/deployment/rtmdet_onnx_infer.py \
        runs/rtmdet/rtmdet_tiny_end2end.onnx demo/demo.jpg \
        --vis runs/rtmdet/vis/demo_end2end.jpg

    # dataset mode: dump COCO-format results that can be compared with the
    # ground truth using tools/deployment/eval_coco_results.py
    python tools/deployment/rtmdet_onnx_infer.py \
        runs/rtmdet/rtmdet_tiny_end2end.onnx \
        --img-dir /home/heqi/dataset/coco/val2017 --limit 200 \
        --output-json runs/rtmdet/results/rtmdet_tiny_end2end_val200.json
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# the COCO category ids are not contiguous, the i-th entry corresponds to
# the i-th COCO class name above
COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
    89, 90
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='RTMDet ONNX inference with pre/post-processing and '
        'optional visualization')
    parser.add_argument(
        '--onnx', 
        default='onnx/rtmdet_tiny.onnx',
        help='ONNX model exported by rtmdet2onnx.py')
    parser.add_argument(
        '--image',
        nargs='?',
        help='input image, optional when --img-dir is given')
    parser.add_argument(
        '--img-dir',
        default='../dataset/val2017',
        help='process all images in this directory (dataset mode), the '
        'image id is parsed from the numeric file name')
    parser.add_argument(
        '--output-json', 
        default='./onnx_results-tiny.json',
        help='dump detections to a COCO-format result json')
    parser.add_argument(
        '--limit',
        type=int,
        default=50,
        help='only process the first N images in dataset mode')
    parser.add_argument(
        '--image-id',
        type=int,
        default=None,
        help='image id used in single image mode when it cannot be '
        'inferred')
    parser.add_argument(
        '--vis-dir',
        default='./onnx_vis',
        help='in dataset mode, save visualizations to this directory')
    parser.add_argument(
        '--vis-limit',
        type=int,
        default=20,
        help='max number of visualizations saved to --vis-dir')
    parser.add_argument(
        '--mode',
        choices=['auto', 'raw', 'end2end'],
        default='auto',
        help='model mode, auto detects it from the output names')
    parser.add_argument(
        '--shape',
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=('HEIGHT', 'WIDTH'),
        help='input size used when the model input is dynamic')
    parser.add_argument(
        '--mean',
        type=float,
        nargs=3,
        default=[103.53, 116.28, 123.675],
        help='BGR mean used for the input normalization, defaults to the '
        'RTMDet COCO setting')
    parser.add_argument(
        '--std',
        type=float,
        nargs=3,
        default=[57.375, 57.12, 58.395],
        help='BGR std used for the input normalization, defaults to the '
        'RTMDet COCO setting')
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='do not normalize the input in this script, use it for models '
        'exported with --normalize (normalization baked into the graph)')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.001,
        help='score threshold applied during post-processing')
    parser.add_argument(
        '--iou-thr', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument(
        '--nms-pre',
        type=int,
        default=30000,
        help='max candidates per level before NMS (raw mode)')
    parser.add_argument(
        '--max-per-img',
        type=int,
        default=300,
        help='max detections kept after NMS (raw mode)')
    parser.add_argument(
        '--min-bbox-size',
        type=float,
        default=0,
        help='minimum box width/height, negative disables the filter '
        '(raw mode)')
    parser.add_argument('--vis', help='path to save the visualization image')
    parser.add_argument(
        '--vis-score-thr',
        type=float,
        default=0.3,
        help='score threshold used for visualization and printing')
    parser.add_argument(
        '--max-print',
        type=int,
        default=100,
        help='max number of detections printed to stdout')
    parser.add_argument(
        '--classes',
        help='optional txt file with one class name per line '
        '(defaults to COCO classes)')
    parser.add_argument(
        '--providers',
        nargs='+',
        default=None,
        help='onnxruntime providers, defaults to CUDA if available '
        'else CPU')
    return parser.parse_args()


def load_class_names(classes_file, num_classes):
    if classes_file is not None:
        with open(classes_file) as f:
            names = [line.strip() for line in f if line.strip()]
    elif num_classes == len(COCO_CLASSES):
        names = list(COCO_CLASSES)
    else:
        names = [str(i) for i in range(num_classes)]
    return names


def create_session(onnx_path, providers):
    if providers is None:
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
    return ort.InferenceSession(onnx_path, providers=providers)


def detect_mode(session, mode):
    if mode != 'auto':
        return mode
    names = [out.name for out in session.get_outputs()]
    return 'end2end' if set(names) >= {'dets', 'labels'} else 'raw'


def get_input_size(session, shape):
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) == 4 and isinstance(input_shape[2], int) \
            and isinstance(input_shape[3], int) and input_shape[2] > 0:
        return int(input_shape[2]), int(input_shape[3])
    assert shape is not None, 'the model input is dynamic, pass --shape'
    return int(shape[0]), int(shape[1])


def preprocess(image, input_hw, mean=None, std=None):
    """Resize keeping ratio, pad with 114, normalize, return BGR NCHW.

    When ``mean``/``std`` are given the sub/div normalization is applied in
    this function (the exported models do not contain those nodes by
    default); otherwise the raw BGR tensor in [0, 255] is returned, which
    is the expected input of models exported with ``--normalize``.
    """
    ori_h, ori_w = image.shape[:2]
    input_h, input_w = input_hw
    scale = min(input_w / ori_w, input_h / ori_h)
    new_w = int(round(ori_w * scale))
    new_h = int(round(ori_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None]
    if mean is not None and std is not None:
        mean = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    scale_factor = (new_w / ori_w, new_h / ori_h)
    return tensor, scale_factor


def _nms_numpy(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (
            boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


def _nms_single_class(boxes, scores, iou_thr):
    """NMS of a single class, prefers fast implementations."""
    try:
        import torch
        import torchvision
        keep = torchvision.ops.nms(
            torch.from_numpy(boxes.astype(np.float32)),
            torch.from_numpy(scores.astype(np.float32)), iou_thr)
        return keep.numpy().astype(np.int64)
    except Exception:
        pass
    if len(boxes) > 1:
        try:
            wh_boxes = np.stack([
                boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0],
                boxes[:, 3] - boxes[:, 1]
            ],
                                axis=1)
            keep = cv2.dnn.NMSBoxes(wh_boxes.tolist(), scores.tolist(), 0.0,
                                    iou_thr)
            return np.array(keep).reshape(-1).astype(np.int64)
        except Exception:
            pass
    return _nms_numpy(boxes, scores, iou_thr)


def nms_classwise(boxes, scores, labels, iou_thr):
    """Class-wise NMS, returns indices sorted by score descending."""
    keep_parts = []
    for cls in np.unique(labels):
        inds = np.where(labels == cls)[0]
        keep_local = _nms_single_class(boxes[inds], scores[inds], iou_thr)
        keep_parts.append(inds[keep_local])
    if not keep_parts:
        return np.zeros(0, dtype=np.int64)
    keep = np.concatenate(keep_parts)
    order = np.argsort(-scores[keep], kind='stable')
    return keep[order]


def postprocess_raw(outputs, output_names, input_hw, score_thr, nms_pre,
                    iou_thr, max_per_img, min_bbox_size):
    """sigmoid + per-level top-k + distance decode + class-wise NMS."""
    input_h, input_w = input_hw
    cls_scores = {}
    bbox_preds = {}
    for name, value in zip(output_names, outputs):
        if name.startswith('cls_score'):
            cls_scores[int(name.rsplit('_', 1)[1])] = value
        elif name.startswith('bbox_pred'):
            bbox_preds[int(name.rsplit('_', 1)[1])] = value

    mlvl_boxes = []
    mlvl_scores = []
    mlvl_labels = []
    for level in sorted(cls_scores):
        cls_score = cls_scores[level][0]  # [C, H, W]
        bbox_pred = bbox_preds[level][0]  # [4, H, W]
        num_classes, height, width = cls_score.shape
        stride = input_h / height

        # [C, H, W] -> [H * W, C], same order as mmdet
        scores = 1.0 / (1.0 + np.exp(-cls_score.reshape(num_classes, -1).T))
        bbox_pred = bbox_pred.reshape(4, -1).T

        flat_scores = scores.reshape(-1)
        num_topk = min(nms_pre, flat_scores.size)
        topk_inds = np.argsort(-flat_scores)[:num_topk]
        topk_scores = flat_scores[topk_inds]
        labels = topk_inds % num_classes
        anchors = topk_inds // num_classes

        points_x = (anchors % width).astype(np.float32) * stride
        points_y = (anchors // width).astype(np.float32) * stride
        dist = bbox_pred[anchors]
        x1 = np.clip(points_x - dist[:, 0], 0, input_w)
        y1 = np.clip(points_y - dist[:, 1], 0, input_h)
        x2 = np.clip(points_x + dist[:, 2], 0, input_w)
        y2 = np.clip(points_y + dist[:, 3], 0, input_h)

        mlvl_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        mlvl_scores.append(topk_scores)
        mlvl_labels.append(labels)

    boxes = np.concatenate(mlvl_boxes)
    scores = np.concatenate(mlvl_scores)
    labels = np.concatenate(mlvl_labels)

    valid = scores > score_thr
    if min_bbox_size >= 0:
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        valid &= (box_w > min_bbox_size) & (box_h > min_bbox_size)
    boxes, scores, labels = boxes[valid], scores[valid], labels[valid]

    # class-wise NMS, equivalent to mmdet's class offset trick
    keep = nms_classwise(boxes, scores, labels, iou_thr)
    keep = keep[:max_per_img]
    return boxes[keep], scores[keep], labels[keep]


def postprocess_end2end(outputs, output_names, score_thr):
    result = dict(zip(output_names, outputs))
    dets = result['dets']
    labels = result['labels']
    if dets.size == 0:
        return (np.zeros(
            (0, 4), dtype=np.float32), np.zeros(
                (0, ), dtype=np.float32), np.zeros((0, ), dtype=np.int64))
    boxes = dets[:, :4].astype(np.float32)
    scores = dets[:, 4].astype(np.float32)
    valid = scores > score_thr
    return boxes[valid], scores[valid], labels[valid].astype(np.int64)


def _class_color(label):
    rng = np.random.RandomState(label)
    return tuple(int(c) for c in rng.randint(64, 256, 3))


def visualize(image, boxes, scores, labels, class_names, score_thr, save_path):
    vis = image.copy()
    for box, score, label in zip(boxes, scores, labels):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = _class_color(int(label))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        name = class_names[int(label)]
        text = f'{name}: {score:.2f}'
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.5, 1)
        cv2.rectangle(vis, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(vis, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    vis_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(vis_dir, exist_ok=True)
    cv2.imwrite(save_path, vis)
    print(f'visualization saved: {save_path}')


def collect_images(args):
    """Collect (image_paths, image_ids) for single/dataset mode.

    In dataset mode (``--img-dir``) the image id is the numeric file stem,
    e.g. ``000000039769.jpg`` -> 39769, which is the COCO image id
    convention. Images are processed in sorted file name order.
    """
    if args.img_dir is not None:
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        paths = sorted(
            p for p in glob.glob(os.path.join(args.img_dir, '*'))
            if p.lower().endswith(exts))
        if args.limit is not None:
            paths = paths[:args.limit]
        if not paths:
            raise ValueError(f'no images found in {args.img_dir}')
        ids = []
        for idx, path in enumerate(paths):
            stem = os.path.splitext(os.path.basename(path))[0]
            ids.append(int(stem) if stem.isdigit() else idx)
        return paths, ids
    if args.image is not None:
        assert os.path.isfile(args.image), \
            f'not a file: {args.image}, use --img-dir for a directory'
        image_id = args.image_id
        if image_id is None:
            stem = os.path.splitext(os.path.basename(args.image))[0]
            image_id = int(stem) if stem.isdigit() else 0
        return [args.image], [image_id]
    raise ValueError('pass an image or --img-dir')


def to_coco_results(image_id, boxes, scores, labels, cat_ids):
    """Convert xyxy boxes to COCO result entries.

    The entry format follows ``CocoMetric.results2json`` of mmdetection:
    ``category_id`` is mapped from the 0-based label through the COCO
    category ids (the COCO ids are not contiguous), ``bbox`` is
    ``[x, y, w, h]``.
    """
    results = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [float(v) for v in box]
        if cat_ids is not None:
            category_id = int(cat_ids[int(label)])
        else:
            category_id = int(label) + 1
        results.append({
            'image_id': int(image_id),
            'bbox': [x1, y1, max(0.0, x2 - x1),
                     max(0.0, y2 - y1)],
            'score': float(score),
            'category_id': category_id
        })
    return results


def main():
    args = parse_args()

    session = create_session(args.onnx, args.providers)
    output_names = [out.name for out in session.get_outputs()]
    mode = detect_mode(session, args.mode)
    input_hw = get_input_size(session, args.shape)
    mean = None if args.no_normalize else args.mean
    std = None if args.no_normalize else args.std

    image_paths, image_ids = collect_images(args)
    single_image = args.img_dir is None
    class_names = load_class_names(args.classes, 80)
    if class_names == COCO_CLASSES:
        cat_ids = COCO_CATEGORY_IDS
    else:
        cat_ids = None

    all_results = []
    vis_saved = 0
    total_time = 0.0
    print(f'model: {args.onnx}')
    for idx, (path, image_id) in enumerate(zip(image_paths, image_ids)):
        image = cv2.imread(path)
        assert image is not None, f'failed to read image: {path}'

        start = time.time()
        tensor, scale_factor = preprocess(image, input_hw, mean, std)
        preprocess_time = time.time() - start

        start = time.time()
        outputs = session.run(None, {session.get_inputs()[0].name: tensor})
        inference_time = time.time() - start

        start = time.time()
        if mode == 'end2end':
            boxes, scores, labels = postprocess_end2end(
                outputs, output_names, args.score_thr)
        else:
            boxes, scores, labels = postprocess_raw(outputs, output_names,
                                                    input_hw, args.score_thr,
                                                    args.nms_pre, args.iou_thr,
                                                    args.max_per_img,
                                                    args.min_bbox_size)
        
        boxes[:, 0::2] /= scale_factor[0]
        boxes[:, 1::2] /= scale_factor[1]
        postprocess_time = time.time() - start
        total_time += preprocess_time + inference_time + postprocess_time

        all_results.extend(
            to_coco_results(image_id, boxes, scores, labels, cat_ids))

        if single_image:
            ori_h, ori_w = image.shape[:2]
            print(f'mode: {mode}, image: {ori_w}x{ori_h}, '
                  f'input: {input_hw[1]}x{input_hw[0]}')
            print(f'image_id: {image_id}, detections: {len(boxes)} '
                  f'(score_thr={args.score_thr})')
            print(f'time: preprocess={preprocess_time * 1000:.1f}ms, '
                  f'inference={inference_time * 1000:.1f}ms, '
                  f'postprocess={postprocess_time * 1000:.1f}ms')
            printed = 0
            for box, score, label in zip(boxes, scores, labels):
                if printed >= args.max_print:
                    break
                if score < args.vis_score_thr:
                    continue
                x1, y1, x2, y2 = box
                print(f'  {class_names[int(label)]:<15} {score:.3f} '
                      f'[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]')
                printed += 1
        elif (idx + 1) % 50 == 0 or idx + 1 == len(image_paths):
            print(f'processed {idx + 1}/{len(image_paths)} images '
                  f'({total_time:.1f}s)')

        if single_image and args.vis is not None:
            visualize(image, boxes, scores, labels, class_names,
                      args.vis_score_thr, args.vis)
        if args.vis_dir is not None and vis_saved < args.vis_limit:
            vis_path = os.path.join(args.vis_dir, os.path.basename(path))
            visualize(image, boxes, scores, labels, class_names,
                      args.vis_score_thr, vis_path)
            vis_saved += 1

    if not single_image and len(image_paths) > 0:
        print(f'{len(image_paths)} images, {total_time:.1f}s total, '
              f'{total_time / len(image_paths) * 1000:.1f}ms/image')

    if args.output_json is not None:
        out_dir = os.path.dirname(os.path.abspath(args.output_json))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(all_results, f)
        print(f'{len(all_results)} detections saved to {args.output_json}')


if __name__ == '__main__':
    main()
