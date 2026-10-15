import argparse
import os

import cv2
import numpy as np

# mean/std of RTMDet/rtmdet_onnx_infer.py CLI defaults (BGR)
MEAN = [103.53, 116.28, 123.675]
STD = [57.375, 57.12, 58.395]


# copied from self-developed/RTMDet/rtmdet_onnx_infer.py (preprocess)
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
    resized = cv2.resize(
        image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None]
    if mean is not None and std is not None:
        mean = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    scale_factor = (new_w / ori_w, new_h / ori_h)
    return tensor, scale_factor


def main():
    parser = argparse.ArgumentParser(
        description='Convert 1000-jpg to RTMDet preprocessed float32 bins')
    parser.add_argument('--src', default='1000-jpg', help='jpg directory')
    parser.add_argument(
        '--dst', default='rtmdet_1000-bin', help='output bin directory')
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    names = sorted(f for f in os.listdir(args.src) if f.endswith('.jpg'))
    for i, name in enumerate(names, 1):
        image = cv2.imread(os.path.join(args.src, name))
        if image is None:
            raise RuntimeError(f'Failed to read image: {name}')
        tensor, _ = preprocess(image, (640, 640), MEAN, STD)
        out = os.path.join(args.dst, os.path.splitext(name)[0] + '.bin')
        tensor.astype(np.float32).tofile(out)
        if i % 100 == 0 or i == len(names):
            print(f'[{i}/{len(names)}] {out}')

    # numerical self-check of the written bin against preprocess()
    sample = names[0]
    data = np.fromfile(
        os.path.join(args.dst, os.path.splitext(sample)[0] + '.bin'),
        dtype=np.float32).reshape(1, 3, 640, 640)
    expect, _ = preprocess(cv2.imread(os.path.join(args.src, sample)),
                           (640, 640), MEAN, STD)
    np.testing.assert_allclose(data, expect, rtol=1e-5, atol=1e-5)
    print(f'self-check passed: {sample}, shape={data.shape}, '
          f'range=[{data.min():.3f}, {data.max():.3f}]')


if __name__ == '__main__':
    main()
