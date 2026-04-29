# CV Assignment 2 Harness

本仓库对应 `HW2_计算机视觉.pdf`，覆盖三部分：宠物分类、VisDrone 检测与视频多目标跟踪、宠物三分类语义分割。优先使用统一 harness 运行，所有实验输出默认进入 `runs/`，模型权重默认进入 `checkpoints/`。

## 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

当前机器已确认有 `torch 2.6.0+cu126` 与 `torchvision 0.21.0+cu126`；`ultralytics`、`opencv-python`、`wandb` 已可用于本作业环境。

## 统一入口

```bash
PYTHONPATH=src python -m cvhw2 --help
```

## 任务 1：宠物分类

Baseline：ImageNet 预训练 ResNet-18，替换为 37 类输出层，backbone 使用较小学习率，新输出层使用较大学习率。

CPU 冒烟测试：

```bash
PYTHONPATH=src python -m cvhw2 train-classification --config configs/classification_smoke.yaml
```

```bash
PYTHONPATH=src python -m cvhw2 train-classification --config configs/classification_resnet18.yaml
```

随机初始化消融：

```bash
PYTHONPATH=src python -m cvhw2 train-classification --config configs/classification_random.yaml
```

SE 注意力拓展：

```bash
PYTHONPATH=src python -m cvhw2 train-classification --config configs/classification_se_resnet18.yaml
```

生成混淆矩阵和错误样例：

```bash
PYTHONPATH=src python -m cvhw2 analyze-classification \
  --config configs/classification_resnet18.yaml \
  --checkpoint checkpoints/classification/resnet18_imagenet_baseline/best.pt \
  --split test \
  --out-dir assets/classification_resnet18_analysis
```

## 任务 2：VisDrone 检测与视频跟踪

先下载公开 VisDrone2019-DET train/val/test-dev 数据，并将原始标注转换为 YOLO 格式：

```bash
bash scripts/download_visdrone_det.sh
```

如果数据已经手动放在 `data/VisDrone/`，只执行转换：

```bash
PYTHONPATH=src python -m cvhw2 convert-visdrone --source data/VisDrone --out data/visdrone_yolo
```

训练 YOLOv8：

```bash
PYTHONPATH=src python -m cvhw2 train-detection --config configs/detection_yolov8n.yaml
PYTHONPATH=src python -m cvhw2 train-detection --config configs/detection_yolov8s.yaml
```

强单阶段检测器拓展实验，针对 VisDrone 小目标提高输入分辨率：

```bash
PYTHONPATH=src python -m cvhw2 train-detection --config configs/detection_yolo11m_img960.yaml
```

对 10-30 秒视频进行 tracking、稳定 ID 可视化和越线计数：

```bash
PYTHONPATH=src python -m cvhw2 track-video \
  --weights checkpoints/detection/best.pt \
  --video data/videos/test.mp4 \
  --out runs/detection/tracked.mp4 \
  --line 100 400 900 400 \
  --export-frames 120 121 122 123
```

跟踪拓展实验建议：

```bash
PYTHONPATH=src python -m cvhw2 track-video --weights checkpoints/detection/best.pt --video data/videos/test.mp4 --out runs/detection/tracked_bytetrack_c025.mp4 --line 100 400 900 400 --tracker bytetrack.yaml --conf 0.25 --export-frames 120 121 122 123
PYTHONPATH=src python -m cvhw2 track-video --weights checkpoints/detection/best.pt --video data/videos/test.mp4 --out runs/detection/tracked_botsort_c025.mp4 --line 100 400 900 400 --tracker botsort.yaml --conf 0.25 --export-frames 120 121 122 123
PYTHONPATH=src python -m cvhw2 track-video --weights checkpoints/detection/best.pt --video data/videos/test.mp4 --out runs/detection/tracked_bytetrack_c050.mp4 --line 100 400 900 400 --tracker bytetrack.yaml --conf 0.50 --export-frames 120 121 122 123
```

## 任务 3：宠物三分类分割

从零手写 U-Net，不使用预训练权重。三种损失配置：

CPU 冒烟测试：

```bash
PYTHONPATH=src python -m cvhw2 train-segmentation --config configs/segmentation_smoke.yaml
```

```bash
PYTHONPATH=src python -m cvhw2 train-segmentation --config configs/segmentation_ce.yaml
PYTHONPATH=src python -m cvhw2 train-segmentation --config configs/segmentation_dice.yaml
PYTHONPATH=src python -m cvhw2 train-segmentation --config configs/segmentation_combo.yaml
```

## 报告与权重

最终实验报告 PDF 将放在 `reports/final_report.pdf`。训练数据、视频、模型权重和中间运行结果不提交到 GitHub；最佳权重通过网盘链接在报告中提供。

需要准备：

- Github repo 链接。
- 模型权重网盘下载地址。
- 训练/验证 loss、Accuracy/mAP/mIoU 曲线截图。
- 检测跟踪视频关键帧、遮挡连续 3-4 帧、越线计数截图。

多实验结果汇总示例：

```bash
python scripts/summarize_runs.py runs/classification/*/metrics.csv --out assets/classification_summary --metric best_val_accuracy
python scripts/summarize_runs.py runs/segmentation/*/metrics.csv --out assets/segmentation_summary --metric best_val_miou
```
