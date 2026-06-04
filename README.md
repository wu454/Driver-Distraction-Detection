# Driver Distraction Detection — Inference Bundle

Five-ROI v2 Temporal + **Tier-1** 本地部署包：图片 / 摄像头 / 视频推理，Gradio 可视化，以及 State Farm 数据集幻灯片工具。

| 项目 | 说明 |
|------|------|
| 架构 | Five-ROI 多区域 + TemporalFiveModel（5 帧 clip） |
| 部署 | Tier-1 混合门控（5 帧 + 9 帧长窗口） |
| 验证集 | p021–p026，3642 temporal clips |
| **Tier-1 准确率** | **81.80%**（见 `tier1_eval.json`） |

## 类别

| ID | 英文 | 中文 |
|----|------|------|
| c0 | Safe driving | 安全驾驶 |
| c1 | Texting (right) | 右手发短信 |
| c2 | Phone call (right) | 右手打电话 |
| c3 | Texting (left) | 左手发短信 |
| c4 | Phone call (left) | 左手打电话 |
| c5 | Operating radio | 调收音机 |
| c6 | Drinking | 喝水 |
| c7 | Reaching behind | 伸手到后座 |
| c8 | Hair / makeup | 化妆/照镜子 |
| c9 | Talking to passenger | 与后座乘客交谈 |

---

## 快速开始

### 1. 环境

- Python 3.10+
- CUDA 11.8+（推荐，如 RTX 3050）
- PyTorch 2.x

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

GPU 版 PyTorch：https://pytorch.org

### 2. 模型权重

`model_best.pth`（约 155 MB）**不在 Git 中**（见 `.gitignore`）。请放入本目录：

- 从 GitHub **Releases** 下载，或
- 自行从训练产物复制

### 3. Web 界面（推荐演示）

```bash
python driver_distraction_ui.py --device cuda
# http://127.0.0.1:7860
```

Windows：`run_ui.bat`

支持：上传图片、摄像头拍照、视频分析；危险等级 Low / Medium / High；标注视频保存到 `output/`。

### 4. 命令行推理

```bash
# 单张
python predict_five_roi.py --image photo.jpg

# 文件夹
python predict_five_roi.py --dir ./images/ --output results.json
```

### 5. 数据集幻灯片

需与上级目录准备 State Farm 数据：

```
../data/train/              # c0 … c9
../data/driver_imgs_list.csv
```

**无推理（仅拼 MP4）：**

```bash
python slideshow_driver_preview.py --subjects p021,p024 --hold-seconds 1
# 或 run_slideshow_preview.bat
```

**带推理标注：**

```bash
python slideshow_driver_inference.py --subject p021 --device cuda --force-temporal
```

默认 `--sample-mode contiguous`：每类取 img_id **连续段**，利于 5/9 帧时序。

---

## 目录结构

```
inference_bundle/
├── model_best.pth              模型权重（需自行下载，~155 MB）
├── requirements.txt
├── tier1_eval.json             验证集指标
├── .gitignore
│
├── driver_distraction_ui.py    Gradio Web 界面
├── run_ui.bat / run_ui.sh
│
├── predict_five_roi.py         图片 / GIF 推理
├── run_image.sh / run_folder.sh / run_gif.sh
│
├── slideshow_driver_preview.py     无推理幻灯片
├── slideshow_driver_inference.py   带推理幻灯片
├── run_slideshow_preview.bat / .sh
│
├── inference_video.py          视频 CLI（四象限裁剪，未接入 Web UI）
├── inference_prepared_video.py
├── mp4_q2_mirror_prepare.py
│
├── tier1_inference.py          Tier-1 门控
├── inference_gates.py
├── models.py / dataset.py / augmentations.py / roi_config.py
│
├── output/                     界面导出的标注视频（gitignore）
└── logs/
```

---

## 说明

- **Web UI 视频页**用 OpenCV 读帧 + `predict_five_roi` 推理，**不调用** `inference_video.py` 等 CLI 脚本。
- 界面默认 **完整 Tier-1**（与 `--force-temporal` 一致）；勾选「快速推理」会降为单帧，速度更快、精度略低。
- 更细的中文说明见 `使用说明.txt`（本地文档，未纳入 Git）。

---

## 数据与许可

- 数据集：[State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)
- 仅供课程 / 研究；数据使用须遵循 Kaggle 规则。
