# Copyright (c) OpenMMLab. All rights reserved.
"""Export RTMDet models to ONNX.

By default the normalization (sub/div with the mean/std of the config's
data preprocessor) is removed from the graph, so the exported model takes
an already normalized BGR tensor with shape (N, 3, H, W) as input
(``(img - mean) / std``). This is convenient for deployment runtimes
where the mean/std are configured outside the model. Pass ``--normalize``
to bake the normalization into the graph instead, in that case the model
takes a BGR image tensor in [0, 255] as input.

Two export modes are supported:

1. Raw mode (default): the outputs are the raw classification scores
   (logits) and the decoded bounding box distances of every feature level.
   Post-processing (sigmoid/NMS) is expected to be performed by the
   inference engine.
2. End-to-end mode (``--end2end``): sigmoid, top-k candidate filtering,
   distance decoding, class-wise NMS and ``max_per_img`` are exported into
   the graph. It only supports a fixed input shape and batch size 1. The
   outputs are ``dets`` of shape (N, 5) in (x1, y1, x2, y2, score) format
   and ``labels`` of shape (N,), sorted by score in descending order.
   The boxes are in the coordinate space of the network input (e.g.
   640x640), callers need to rescale them to the original image size.

Examples:
    python tools/deployment/rtmdet2onnx.py \
        configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py \
        runs/checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth \
        runs/rtmdet/rtmdet_tiny.onnx

    python tools/deployment/rtmdet2onnx.py \
        configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py \
        runs/checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth \
        runs/rtmdet/rtmdet_tiny_end2end.onnx --end2end

    python tools/deployment/rtmdet2onnx.py \
        configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py \
        runs/checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth \
        runs/rtmdet/rtmdet_tiny_normalized.onnx --normalize
"""
import argparse

import numpy as np
import torch
import torchvision
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


class BaseRTMDetWrapper(torch.nn.Module):
    """Base wrapper handling the data normalization.

    Args:
        detector (torch.nn.Module): the RTMDet detector.
        normalize_input (bool): whether to bake the normalization
            (bgr_to_rgb/rgb_to_bgr and sub/div with mean/std) into the
            graph. When False the model expects an already normalized
            input tensor. Defaults to True.
    """

    def __init__(self,
                 detector: torch.nn.Module,
                 normalize_input: bool = True) -> None:
        super().__init__()
        self.detector = detector
        self.normalize_input = normalize_input
        data_preprocessor = detector.data_preprocessor
        mean = getattr(data_preprocessor, 'mean', None)
        std = getattr(data_preprocessor, 'std', None)
        if mean is None:
            mean = torch.zeros(1, 3, 1, 1)
        if std is None:
            std = torch.ones(1, 3, 1, 1)
        self.register_buffer('mean', torch.as_tensor(mean).clone().detach())
        self.register_buffer('std', torch.as_tensor(std).clone().detach())
        self.bgr_to_rgb = getattr(data_preprocessor, 'bgr_to_rgb', False)
        self.rgb_to_bgr = getattr(data_preprocessor, 'rgb_to_bgr', False)

    def normalize(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.normalize_input:
            return inputs
        if self.bgr_to_rgb or self.rgb_to_bgr:
            inputs = inputs[:, [2, 1, 0], :, :]
        return (inputs - self.mean) / self.std


class RTMDetONNXWrapper(BaseRTMDetWrapper):
    """Wrap a RTMDet detector for raw ONNX export.

    Returns a flat tuple of the raw head outputs, i.e. the classification
    logits and the decoded bounding box distances of every feature level.
    """

    def __init__(self,
                 detector: torch.nn.Module,
                 normalize_input: bool = True) -> None:
        super().__init__(detector, normalize_input)

    def forward(self, inputs: torch.Tensor) -> tuple:
        inputs = self.normalize(inputs)
        feats = self.detector.extract_feat(inputs)
        cls_scores, bbox_preds = self.detector.bbox_head.forward(feats)
        return tuple(cls_scores) + tuple(bbox_preds)


class RTMDetEnd2EndWrapper(BaseRTMDetWrapper):
    """Wrap a RTMDet detector for end-to-end ONNX export.

    The post-processing of ``BaseDenseHead.predict_by_feat`` is reproduced
    in the graph: sigmoid, global top-k candidate filtering
    (``nms_pre``), distance decoding with clipping to the input size,
    ``min_bbox_size`` filtering, class-wise NMS (boxes are offset by class)
    and ``max_per_img``. Only a fixed input shape and batch size 1 are
    supported.
    """

    def __init__(self,
                 detector: torch.nn.Module,
                 input_shape,
                 normalize_input: bool = True) -> None:
        super().__init__(detector, normalize_input)
        bbox_head = detector.bbox_head
        self.strides = [
            int(stride[0]) for stride in bbox_head.prior_generator.strides
        ]
        test_cfg = detector.test_cfg if detector.test_cfg is not None else {}
        self.score_thr = float(test_cfg.get('score_thr', 0.001) or 0.001)
        self.nms_pre = int(test_cfg.get('nms_pre', 30000) or 30000)
        nms_cfg = test_cfg.get('nms') or {}
        self.iou_thr = float(nms_cfg.get('iou_threshold', 0.65))
        self.max_per_img = int(test_cfg.get('max_per_img', 300) or 300)
        self.min_bbox_size = float(test_cfg.get('min_bbox_size', -1))
        self.input_shape = tuple(input_shape)

    def _decode_level(self, cls_score: torch.Tensor, bbox_pred: torch.Tensor,
                      stride: int):
        """Filter one feature level and decode distances to boxes.

        The flatten order follows mmdet's
        ``cls_score.permute(1, 2, 0).reshape(-1, C)``.
        """
        _, num_classes, _, width = cls_score.shape
        scores = cls_score.permute(0, 2, 3, 1).reshape(-1).sigmoid()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)

        num_topk = min(self.nms_pre, scores.numel())
        scores, topk_inds = scores.topk(num_topk)
        labels = topk_inds % num_classes
        anchors = topk_inds // num_classes

        dtype = bbox_pred.dtype
        # anchor index = h * width + w, which keeps the same order as
        # ``MlvlPointGenerator.single_level_grid_priors``.
        points_x = (anchors % width).to(dtype) * stride
        points_y = (anchors // width).to(dtype) * stride
        dist = bbox_pred[anchors]

        img_h, img_w = self.input_shape
        x1 = (points_x - dist[:, 0]).clamp(0, img_w)
        y1 = (points_y - dist[:, 1]).clamp(0, img_h)
        x2 = (points_x + dist[:, 2]).clamp(0, img_w)
        y2 = (points_y + dist[:, 3]).clamp(0, img_h)
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return boxes, scores, labels

    def forward(self, inputs: torch.Tensor) -> tuple:
        inputs = self.normalize(inputs)
        feats = self.detector.extract_feat(inputs)
        cls_scores, bbox_preds = self.detector.bbox_head.forward(feats)

        mlvl_boxes = []
        mlvl_scores = []
        mlvl_labels = []
        for cls_score, bbox_pred, stride in zip(cls_scores, bbox_preds,
                                                self.strides):
            boxes, scores, labels = self._decode_level(cls_score, bbox_pred,
                                                       stride)
            mlvl_boxes.append(boxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
        boxes = torch.cat(mlvl_boxes)
        scores = torch.cat(mlvl_scores)
        labels = torch.cat(mlvl_labels)

        valid = scores > self.score_thr
        if self.min_bbox_size >= 0:
            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            valid = valid & (box_w > self.min_bbox_size) & (
                box_h > self.min_bbox_size)
        keep_inds = valid.nonzero(as_tuple=False).squeeze(1)
        boxes = boxes[keep_inds]
        scores = scores[keep_inds]
        labels = labels[keep_inds]

        # class-wise NMS via the class offset trick, same as
        # ``mmcv.ops.batched_nms``.
        max_coordinate = boxes.max()
        offsets = labels.to(boxes) * (max_coordinate + 1)
        keep = torchvision.ops.nms(boxes + offsets[:, None], scores,
                                   self.iou_thr)
        keep = keep[:self.max_per_img]

        dets = torch.cat([boxes[keep], scores[keep][:, None]], dim=-1)
        return dets, labels[keep]


def parse_args():
    parser = argparse.ArgumentParser(description='Export RTMDet to ONNX')
    parser.add_argument('config', help='config file of the detector')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('output', help='output ONNX file')
    parser.add_argument(
        '--shape',
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=('HEIGHT', 'WIDTH'),
        help='input image size, defaults to 640 640')
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='export with dynamic input height and width')
    parser.add_argument(
        '--end2end',
        action='store_true',
        help='export post-processing (sigmoid, decode, NMS) into the graph, '
        'only supports fixed shape and batch size 1')
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='bake the input normalization (bgr_to_rgb and sub/div with '
        'mean/std) into the graph; by default it is not exported and the '
        'model expects an already normalized input')
    parser.add_argument(
        '--opset-version', type=int, default=11, help='ONNX opset version')
    parser.add_argument(
        '--device', default='cpu', help='device used for export')
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='do not verify the exported model with onnxruntime')
    return parser.parse_args()


def verify(onnx_file: str, wrapper: torch.nn.Module, dummy: torch.Tensor,
           output_names) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print('onnx/onnxruntime is not installed, skip verification')
        return

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print(f'onnx checker passed: {onnx_file}')

    sess = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
    onnx_outputs = sess.run(None, {'images': dummy.cpu().numpy()})
    device = next(wrapper.parameters()).device
    if device.type != 'cpu':
        wrapper.to('cpu')
    with torch.no_grad():
        torch_outputs = wrapper(dummy.cpu())
    for name, onnx_out, torch_out in zip(output_names, onnx_outputs,
                                         torch_outputs):
        torch_out = torch_out.cpu().numpy()
        abs_diff = np.abs(onnx_out - torch_out)
        rel_diff = abs_diff / np.maximum(np.abs(torch_out), 1e-6)
        print(f'{name}: max abs diff = {abs_diff.max():.6e}, '
              f'max rel diff = {rel_diff.max():.6e}, '
              f'mean rel diff = {rel_diff.mean():.6e}')
        np.testing.assert_allclose(onnx_out, torch_out, rtol=1e-2, atol=1e-3)


def _iou_1_to_n(box, boxes):
    """IoU between a single box and a set of boxes."""
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def verify_end2end(onnx_file: str, wrapper: torch.nn.Module,
                   dummy: torch.Tensor) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print('onnx/onnxruntime is not installed, skip verification')
        return

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print(f'onnx checker passed: {onnx_file}')

    sess = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
    onnx_dets, onnx_labels = sess.run(None, {'images': dummy.cpu().numpy()})

    device = next(wrapper.parameters()).device
    if device.type != 'cpu':
        wrapper.to('cpu')
    with torch.no_grad():
        torch_dets, torch_labels = wrapper(dummy.cpu())
    torch_dets = torch_dets.cpu().numpy()
    torch_labels = torch_labels.cpu().numpy()

    print(f'num detections: onnx = {len(onnx_dets)}, '
          f'torch = {len(torch_dets)}')
    assert len(onnx_dets) == len(torch_dets), 'detection count mismatch'

    # sort by score (descending), then label and coordinates, so that
    # detections with near-equal scores are aligned between the two outputs.
    order_onnx = np.lexsort(
        (onnx_dets[:, 0], onnx_dets[:, 1], onnx_labels, -onnx_dets[:, 4]))
    order_torch = np.lexsort(
        (torch_dets[:, 0], torch_dets[:, 1], torch_labels, -torch_dets[:, 4]))
    onnx_dets = onnx_dets[order_onnx]
    onnx_labels = onnx_labels[order_onnx]
    torch_dets = torch_dets[order_torch]
    torch_labels = torch_labels[order_torch]

    if np.array_equal(onnx_labels, torch_labels):
        abs_diff = np.abs(onnx_dets - torch_dets)
        print(f'dets: max abs diff = {abs_diff.max():.6e}')
        np.testing.assert_allclose(onnx_dets, torch_dets, rtol=1e-3, atol=1e-2)
    else:
        # onnxruntime and torchvision break NMS ties differently, so match
        # detections greedily by score with the same label instead.
        used = np.zeros(len(torch_dets), dtype=bool)
        matched = 0
        ious = []
        box_diffs = []
        for i in np.argsort(-onnx_dets[:, 4]):
            cands = np.where((torch_labels == onnx_labels[i]) & ~used)[0]
            if cands.size == 0:
                continue
            ious_i = _iou_1_to_n(onnx_dets[i, :4], torch_dets[cands, :4])
            j = cands[np.argmax(ious_i)]
            if ious_i.max() > 0.9:
                matched += 1
                used[j] = True
                ious.append(ious_i.max())
                box_diffs.append(
                    np.abs(onnx_dets[i, :4] - torch_dets[j, :4]).max())
        ratio = matched / max(len(onnx_dets), 1)
        print(f'matched detections (same label, IoU > 0.9): {matched}/'
              f'{len(onnx_dets)} ({ratio:.2%})')
        if box_diffs:
            print(f'matched boxes: max abs diff = {max(box_diffs):.6e}, '
                  f'min IoU = {min(ious):.4f}')
        assert ratio >= 0.9, 'detection matching ratio below 90%'


def main():
    args = parse_args()
    register_all_modules()

    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()

    height, width = args.shape
    # a fixed random input is used both for tracing and for the numerical
    # verification; it avoids degenerate zero inputs and score ties that
    # would make the NMS comparison between onnxruntime and torchvision
    # unstable
    generator = torch.Generator().manual_seed(0)
    dummy = torch.rand(
        1, 3, height, width, generator=generator).to(args.device)

    if args.end2end:
        assert not args.dynamic, \
            '--end2end only supports a fixed input shape'
        wrapper = RTMDetEnd2EndWrapper(
            model, (height, width),
            normalize_input=args.normalize).to(args.device).eval()
        output_names = ['dets', 'labels']
        dynamic_axes = None
    else:
        wrapper = RTMDetONNXWrapper(
            model, normalize_input=args.normalize).to(args.device).eval()
        num_levels = len(list(cfg.model.bbox_head.anchor_generator.strides))
        output_names = [f'cls_score_{i}' for i in range(num_levels)
                        ] + [f'bbox_pred_{i}' for i in range(num_levels)]
        dynamic_axes = None
        if args.dynamic:
            dynamic_axes = {'images': {2: 'height', 3: 'width'}}
            for name in output_names:
                dynamic_axes[name] = {2: 'height', 3: 'width'}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            args.output,
            input_names=['images'],
            output_names=output_names,
            opset_version=args.opset_version,
            dynamic_axes=dynamic_axes)
    print(f'exported: {args.output}')

    if not args.no_verify:
        if args.end2end:
            verify_end2end(args.output, wrapper, dummy)
        else:
            verify(args.output, wrapper, dummy, output_names)


if __name__ == '__main__':
    main()
