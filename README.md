# RTMDet COCO 目标检测

将 RTMDet（tiny / s / m / l / x，COCO 80 类）从 ONNX 经 Pulsar2 量化编译为
`compiled.axmodel`，并在 AX650A（NPU3 模式）上完成 COCO val2017 全量
（5000 张）精度验证。tiny / s / m 已评测，AP50 与 AP50-95 与官方一致
（偏差 ≤ 0.004）。

本包（`RTMDet/RTMDet.axera/`）包含量化配置 `config/` 与 ONNX 模型
`onnx/`；导出/评测脚本（`rtmdet2onnx.py`、`rtmdet_onnx_infer.py`、
`rtmdet_axera_infer.py`、`eval_coco_results.py`）与量化产物位于上级
`RTMDet/` 工作区，以下命令均在该工作区执行。

## 1. 模型精度对比

COCO val2017 全量 5000 张，80 类；评测协议与官方训练日志一致
（mmdet `CocoMetric`，`val_evaluator.proposal_nums=(100,1,10)`）。

| 模型 | 精度模式 | AP50-95 (mAP) | AP50 | ΔmAP（Axera − 官方） | 耗时（ms） |
| --- | --- | --- | --- | --- | --- |
| RTMDet-tiny | FP32 | 0.4111 | 0.5785 | / | / |
| RTMDet-tiny | [W16A16](./config/config.json) | 0.4113 | 0.5786 | +0.0023 | 10.129 |
| RTMDet-s | FP32 | 0.445 | 0.619 | / | / |
| RTMDet-s | [W16A16](./config/config.json) | 0.4466 | 0.6184 | +0.0016 | 15.990 |
| RTMDet-m | FP32 | 0.491 | 0.667 | / | / |
| RTMDet-m | [W16A16](./config/config.json) | 0.4941 | 0.6676 | +0.0031 | 40.337 |

性能：AX650N NPU3 模式；单帧耗时可用`ax_run_model -w 10 -r 100 -m <model>.axmodel` 实测。

## 2. 校准集准备

1）数据下载（COCO val2017 官方源，约 1.0G + 241M）：

```bash
mkdir -p ../dataset && cd ../dataset
curl -sL -C - -o val2017.zip \
  "http://images.cocodataset.org/zips/val2017.zip"
curl -sL -C - -o annotations_trainval2017.zip \
  "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
unzip -q val2017.zip && unzip -q annotations_trainval2017.zip
```

解压得到 `../dataset/val2017/`（5000 张）与 `annotations/`；评测用 GT 为
`instances_val2017.json`。
`../yolo_acc_tools/1000-jpg` 为 val2017 按文件名排序的前 1000 张，用于
构建下文的二进制校准集。

2）生成校准集（配置中 `calibration_size=32`）：

- 图像（`config.json` 使用）：32 张 640x640 PNG
  （等比 resize + 114 填充、BGR 0-255、未归一化；mean/std 由板上
  `input_processors` 完成），由 `rtmdet_make_calib.py` 生成并自动归档：

```bash
# 在 RTMDet/ 工作区执行
python rtmdet_make_calib.py   # 默认取 ../yolo_acc_tools/1000-jpg 前 32 张
```

产物：`../yolo_acc_tools/rtmdet_1000-bin(.tar)`、
`../yolo_acc_tools/rtmdet_calib_640x640-img32(.tar)`（tar 条目为
`./<stem>.bin|png`）。

## 3. 模型转换

1）ONNX 导出 `tools/rtmdet2onnx.py`：使用 [MMDetection](https://github.com/open-mmlab/mmdetection) 官方 COCO 权重导出定
shape 640x640；默认 raw 模式（输出 `cls_score_i` / `bbox_pred_i`，后处理
在推理端），`--end2end` 将 sigmoid、解码、NMS 导出进图（输出 `dets` /
`labels`）；默认图内不含归一化（输入为已归一化 BGR），`--normalize` 可将
BGR->RGB 与 mean/std 固化进图。

```bash
python rtmdet2onnx.py <config> <ckpt> RTMDet.axera/onnx/rtmdet_tiny.onnx
python rtmdet2onnx.py <config> <ckpt> RTMDet.axera/onnx/rtmdet_tiny_end2end.onnx \
    --end2end
```

2）Pulsar2 量化编译：

工具链：[Pulsar2-modelScope](https://www.modelscope.cn/models/AXERA-TECH/Pulsar2/files) / [Pulsar2-HF](https://huggingface.co/AXERA-TECH/Pulsar2)，目标平台 **AX650A**，NPU3 模式。

```bash
pulsar2 build --config RTMDet.axera/config/config.json
```

- `config.json`：U16/S16 + Image 校准；板上输入
  U8 NHWC BGR，归一化（mean/std 同 RTMDet）与 resize/填充约定由
  `input_processors` 完成。

## 4. 精度验证

流程：ONNX FP32 评估 → axmodel 评估 → COCO 指标对比。

1）ONNX FP32 评估 `eval_onnx.py`（依赖 onnxruntime）：

```bash
# 抽查：默认前 50 张，输出 onnx_results-tiny.json 与 onnx_vis/（前 20 张可视化）
python eval_onnx.py --onnx RTMDet.axera/onnx/rtmdet_tiny.onnx

# 全量 5000 张 COCO 结果
python eval_onnx.py --onnx RTMDet.axera/onnx/rtmdet_tiny.onnx \
    --limit 5000 --output-json onnx_results-tiny.json
```

2）axmodel 评估 `eval_axera.py`（依赖 [axengine](https://github.com/AXERA-TECH/pyaxengine) 
`AxEngineExecutionProvider`，在 AX650 板上运行；数据集模式输出 COCO 结果
json，前 20 张可视化到 `axera_vis/`）：

```bash
python eval_axera.py                                     # tiny 全量
python eval_axera.py --onnx output_rtmdet_s/compiled.axmodel \
    --output-json output_axera_rtmdet_s.json
```

- 输入张量需与编译时的 `input_processors` 约定一致：Binary 配置为已归一化
  FP32 NCHW；Image（`-preprocess`）配置为 U8 NHWC BGR，mean/std 归一化
  在 axmodel 内部完成。
- 产物：`output_axera_rtmdet_{tiny,s,m}.json`（各 150 万条检测）。

3）COCO 指标 `tools/eval_coco_results.py`（需 mmdet 环境，`mmdet CocoMetric` +
pycocotools）：

```bash
python tools/eval_coco_results.py --ann-file instances_val2017.json \
    output_axera_rtmdet_tiny.json

# 与参考结果对比（如 mmdet 官方 fp32 json）/ 逐类 AP
python tools/eval_coco_results.py --ann-file instances_val2017.json \
    output_axera_rtmdet_tiny.json --ref-json <ref.json> --classwise
```

指标为 pycocotools `COCOeval`（IoU 0.5:0.05:0.95、
`maxDets=[100,300,1000]`），与 mmdet `CocoMetric` 口径一致。

## 5. 参考项目

- **MMDetection**（OpenMMLab）：RTMDet COCO 训练配置与官方权重
  - <https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet>
- **AXERA Pulsar2** v7.0 量化编译工具链 + **axengine** 运行时（AX650A）
  - <https://pulsar2-docs.readthedocs.io/en/latest/pulsar2/introduction.html>
