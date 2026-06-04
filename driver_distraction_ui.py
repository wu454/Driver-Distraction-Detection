#!/usr/bin/env python3
"""
驾驶员分心检测 — Tier-1 部署可视化界面

用法（本地 RTX 3050，在 inference_bundle 目录内）:
  pip install -r requirements.txt
  pip install gradio
  python driver_distraction_ui.py --device cuda

浏览器打开: http://127.0.0.1:7860

云服务器仅 CPU 时可改用 --device cpu（较慢，不推荐用于演示）
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# 确保从 inference_bundle 目录加载模块
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict_five_roi import (
    DEFAULT_CKPT,
    load_model,
    predict_pil_frame,
    predict_temporal_clip,
)
from tier1_inference import tier1_long_half_window

WEIGHTS_CACHE = str(ROOT / 'logs' / 'tier1_weights_only.pt')


def load_model_ui(ckpt_path: str, device: torch.device):
    """GPU 本地部署走标准 load；仅 --device cpu 时用 mmap 省内存（云服务器备用）。"""
    ckpt_path = ckpt_path if os.path.isfile(ckpt_path) else str(ROOT / 'model_best.pth')

    if device.type == 'cuda':
        return load_model(ckpt_path, device)

    # CPU：可选 mmap 权重缓存（2GB 内存环境）
    if os.path.isfile(WEIGHTS_CACHE):
        from models import TemporalFiveModel, FiveROIModel
        bundle = torch.load(WEIGHTS_CACHE, map_location='cpu', weights_only=True, mmap=True)
        meta = bundle['meta']
        class_names = meta['class_names']
        model_type = meta.get('model_type', 'FiveROIModel')
        if model_type == 'TemporalFiveModel':
            model = TemporalFiveModel(
                len(class_names), pretrained=False,
                temporal=meta.get('temporal_backend', 'transformer'),
            )
        else:
            model = FiveROIModel(len(class_names), pretrained=False)
        model.load_state_dict(bundle['model_state_dict'], strict=False)
        model.class_names = class_names
        model.use_inference_gates = False
        model.eval()
        return model, class_names, meta

    return load_model(ckpt_path, device)

try:
    import gradio as gr
except ImportError:
    print('请先安装 Gradio:  pip install gradio>=4.0.0')
    sys.exit(1)

CLASS_LABELS_ZH = {
    'c0': '安全驾驶',
    'c1': '右手发短信',
    'c2': '右手打电话',
    'c3': '左手发短信',
    'c4': '左手打电话',
    'c5': '调收音机',
    'c6': '喝水',
    'c7': '伸手到后座',
    'c8': '化妆/照镜子',
    'c9': '与后座乘客交谈',
}

CLASS_LABELS_EN = {
    'c0': 'Safe Driving',
    'c1': 'Texting (Right)',
    'c2': 'Phone Call (Right)',
    'c3': 'Texting (Left)',
    'c4': 'Phone Call (Left)',
    'c5': 'Operating Radio',
    'c6': 'Drinking',
    'c7': 'Reaching Behind',
    'c8': 'Hair / Makeup',
    'c9': 'Talking to Passenger',
}

# (英文, 中文, 颜色 hex, 排序权重 越大越危险)
DANGER_INFO = {
    'c0': ('Low', '低', '#22c55e', 0),
    'c5': ('Medium', '中', '#eab308', 2),
    'c6': ('Medium', '中', '#eab308', 2),
    'c9': ('Medium', '中', '#eab308', 2),
    'c8': ('Medium', '中', '#f97316', 3),
    'c7': ('High', '高', '#ef4444', 4),
    'c1': ('High', '高', '#ef4444', 4),
    'c2': ('High', '高', '#ef4444', 4),
    'c3': ('High', '高', '#ef4444', 4),
    'c4': ('High', '高', '#ef4444', 4),
}


def numpy_rgb_to_pil(arr: np.ndarray | None) -> Image.Image | None:
    if arr is None:
        return None
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return Image.fromarray(arr.astype(np.uint8))


def pil_to_numpy_rgb(pil: Image.Image) -> np.ndarray:
    return np.array(pil.convert('RGB'))


def empty_panel_msg(msg: str, *, error: bool = False) -> str:
    color = '#b91c1c' if error else '#64748b'
    return f'<div class="result-panel"><p style="color:{color};margin:0;">{msg}</p></div>'


def format_result(
    pred: dict,
    *,
    elapsed_ms: float | None = None,
    frame_info: str | None = None,
) -> tuple[str, str, str, str, str]:
    """返回 (英文标签, 中文标签, 置信度%, 危险等级展示, HTML 侧栏)."""
    code = pred['pred_class']
    en = CLASS_LABELS_EN.get(code, code)
    zh = CLASS_LABELS_ZH.get(code, code)
    conf = pred['confidence'] * 100
    d_en, d_zh, color, _ = DANGER_INFO.get(code, ('Unknown', '未知', '#94a3b8', 0))

    conf_str = f'{conf:.1f}%'
    danger_str = f'{d_en} / {d_zh}'

    top3_lines = []
    for item in pred.get('top3', [])[:3]:
        c = item['class']
        top3_lines.append(
            f"{CLASS_LABELS_EN.get(c, c)} — {item['prob']*100:.1f}%"
        )
    top3_html = '<br>'.join(top3_lines) if top3_lines else '—'

    meta_lines = []
    if elapsed_ms is not None:
        meta_lines.append(f'推理耗时 {elapsed_ms:.0f} ms')
    if frame_info:
        meta_lines.append(frame_info)
    meta_html = ''
    if meta_lines:
        meta_html = (
            '<div class="result-top3" style="border-top:none;padding-top:0;margin-bottom:10px;">'
            + '<br>'.join(f'<span class="result-label">{line}</span>' for line in meta_lines)
            + '</div>'
        )

    html = f"""
<div class="result-panel">
  {meta_html}
  <div class="result-title" style="color: {color};">{en}</div>
  <div class="result-sub">{zh} ({code})</div>
  <div class="result-block">
    <div class="result-label">置信度 Confidence</div>
    <div class="result-value">{conf_str}</div>
  </div>
  <div class="result-block">
    <div class="result-label">危险等级 Danger Level</div>
    <div class="result-value" style="color: {color};">{danger_str}</div>
  </div>
  <div class="result-top3">
    <div class="result-label">Top-3 候选</div>
    <div class="result-sub">{top3_html}</div>
  </div>
</div>
"""
    return en, zh, conf_str, danger_str, html


class DistractionEngine:
    """加载一次模型，供 Gradio 回调复用。"""

    def __init__(self, ckpt: str, device: str, *, use_amp: bool = False):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        self.device = torch.device(
            device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.use_amp = use_amp and self.device.type == 'cuda'
        ckpt_path = ckpt
        if not os.path.isfile(ckpt_path):
            ckpt_path = str(ROOT / 'model_best.pth')
        self.model, self.class_names, self.ckpt_meta = load_model_ui(ckpt_path, self.device)
        self.is_temporal = self.ckpt_meta.get('model_type') == 'TemporalFiveModel'
        self.tier1 = True
        self.hw5 = 2
        self.hw9 = tier1_long_half_window()
        self._frame_buf: deque[Image.Image] = deque(maxlen=2 * self.hw5 + 1)
        self._lock = threading.Lock()
        amp_note = 'fp16' if self.use_amp else 'fp32'
        print(
            f'[UI] 模型已加载: {ckpt_path}  device={self.device}  '
            f'temporal={self.is_temporal}  tier1=True  amp={amp_note}'
        )
        if self.device.type == 'cpu':
            print(
                '[WARN] 当前为 CPU 推理，会非常慢（单张约 5–30 秒）。'
                '请确认 GPU: python -c "import torch; print(torch.cuda.is_available())"'
            )
        self.warmup()

    def warmup(self):
        dummy = Image.new('RGB', (640, 480), (128, 128, 128))
        t0 = time.perf_counter()
        for _ in range(2):
            self.predict_pil(dummy, use_buffer=False, fast_mode=False)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        print(f'[UI] GPU/模型预热完成 ({(time.perf_counter()-t0)*1000:.0f} ms)')

    def _predict_core(
        self,
        pil: Image.Image,
        *,
        use_buffer: bool,
        fast_mode: bool,
    ) -> dict:
        if fast_mode and not use_buffer:
            return predict_pil_frame(
                self.model, pil, self.device, self.class_names, tier1=self.tier1,
            )
        if self.is_temporal and use_buffer:
            clip5 = self._clip_from_buffer(pil)
            clip9 = None if fast_mode else self._clip_long(clip5)
            return predict_temporal_clip(
                self.model, clip5, self.device, self.class_names,
                is_temporal=True, tier1=self.tier1, clip_frames_long=clip9,
            )
        if self.is_temporal:
            clip5 = [pil] * (2 * self.hw5 + 1)
            clip9 = None if fast_mode else [pil] * (2 * self.hw9 + 1)
            return predict_temporal_clip(
                self.model, clip5, self.device, self.class_names,
                is_temporal=True, tier1=self.tier1, clip_frames_long=clip9,
            )
        return predict_pil_frame(
            self.model, pil, self.device, self.class_names, tier1=self.tier1,
        )

    def predict_pil(self, pil: Image.Image, use_buffer: bool = False, fast_mode: bool = False) -> dict:
        with self._lock:
            with torch.inference_mode():
                if self.use_amp:
                    with torch.autocast('cuda'):
                        return self._predict_core(pil, use_buffer=use_buffer, fast_mode=fast_mode)
                return self._predict_core(pil, use_buffer=use_buffer, fast_mode=fast_mode)

    def _clip_from_buffer(self, center: Image.Image | None = None) -> list[Image.Image]:
        if center is not None:
            self._frame_buf.append(center)
        buf = list(self._frame_buf)
        if not buf:
            return []
        if len(buf) == 1:
            return buf * (2 * self.hw5 + 1)
        cidx = len(buf) - 1
        half = self.hw5
        indices = [max(0, min(len(buf) - 1, cidx + o)) for o in range(-half, half + 1)]
        return [buf[i] for i in indices]

    def _clip_long(self, clip5: list[Image.Image]) -> list[Image.Image]:
        """5 帧 clip → 9 帧（边缘 clamp，与 tier1 推理一致）"""
        c5_center = len(clip5) // 2
        half9 = self.hw9
        return [
            clip5[max(0, min(len(clip5) - 1, c5_center + o))]
            for o in range(-half9, half9 + 1)
        ]

    def reset_buffer(self):
        with self._lock:
            self._frame_buf.clear()


ENGINE: DistractionEngine | None = None
_CJK_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def find_cjk_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size in _CJK_FONT_CACHE:
        return _CJK_FONT_CACHE[size]
    candidates = [
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/msyhbd.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simsun.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
    ]
    for path in candidates:
        if os.path.isfile(path):
            font = ImageFont.truetype(path, size=size)
            _CJK_FONT_CACHE[size] = font
            return font
    font = ImageFont.load_default()
    _CJK_FONT_CACHE[size] = font
    print('[WARN] 未找到中文字体，视频标注中文可能显示异常；Windows 需 C:/Windows/Fonts/msyh.ttc')
    return font


def to_pil(image) -> Image.Image | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    if isinstance(image, np.ndarray):
        return numpy_rgb_to_pil(image)
    if isinstance(image, dict):
        for key in ('path', 'name', 'url'):
            val = image.get(key)
            if isinstance(val, str) and os.path.isfile(val):
                return Image.open(val).convert('RGB')
        nested = image.get('image')
        if nested is not None:
            return to_pil(nested)
    if isinstance(image, str):
        if os.path.isfile(image):
            return Image.open(image).convert('RGB')
        if image.startswith('data:image'):
            import base64
            import io
            b64 = image.split(',', 1)[-1]
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
    return None


def load_image_input(image) -> Image.Image | None:
    return to_pil(image)


def normalize_upload_path(media) -> str | None:
    """兼容 Gradio 4/5 的 Video / File 返回值。"""
    if media is None:
        return None
    if isinstance(media, list):
        for item in media:
            p = normalize_upload_path(item)
            if p:
                return p
        return None
    if isinstance(media, str):
        return media if os.path.isfile(media) else None
    if isinstance(media, (Path,)):
        p = str(media)
        return p if os.path.isfile(p) else None
    if isinstance(media, dict):
        for key in ('video', 'name', 'path', 'orig_name'):
            val = media.get(key)
            if val and os.path.isfile(str(val)):
                return str(val)
    return None


def reencode_for_browser(src_path: str) -> str:
    """转为浏览器可播放的 H.264 MP4；失败则退回原文件。"""
    import shutil
    import subprocess

    dst_path = str(ROOT / '_ui_output_pred_browser.mp4')
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        cmd = [
            ffmpeg, '-y', '-loglevel', 'error',
            '-i', src_path,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            '-an', dst_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            if os.path.isfile(dst_path) and os.path.getsize(dst_path) > 0:
                return dst_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            print(f'[WARN] ffmpeg 转码失败，使用原始输出: {exc}')

    return src_path


def get_engine() -> DistractionEngine:
    global ENGINE
    if ENGINE is None:
        raise RuntimeError('Engine not initialized')
    return ENGINE


def infer_image(image, use_temporal_buffer: bool, fast_mode: bool, *, from_button: bool = False):
    pil = load_image_input(image)
    if pil is None:
        if from_button:
            return (
                '',
                '',
                '',
                empty_panel_msg(
                    '未检测到画面。上传图片后可直接推理；'
                    '使用摄像头时请先在左侧点击 📷 拍照，再点「立即检测」。',
                    error=True,
                ),
            )
        return gr.skip()

    t0 = time.perf_counter()
    pred = get_engine().predict_pil(
        pil,
        use_buffer=use_temporal_buffer and get_engine().is_temporal,
        fast_mode=fast_mode,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    en, zh, conf, danger, html = format_result(pred, elapsed_ms=elapsed_ms)
    return en, conf, danger, html


def infer_image_auto(image, use_temporal_buffer, fast_mode):
    return infer_image(image, use_temporal_buffer, fast_mode, from_button=False)


def infer_image_button(image, use_temporal_buffer, fast_mode):
    return infer_image(image, use_temporal_buffer, fast_mode, from_button=True)


def annotate_frame_bgr(frame_bgr: np.ndarray, pred: dict) -> np.ndarray:
    """在画面底部绘制标注条（PIL 渲染中文，避免 OpenCV 问号乱码）。"""
    h, w = frame_bgr.shape[:2]
    code = pred['pred_class']
    conf = pred['confidence']
    zh = CLASS_LABELS_ZH.get(code, code)
    en = CLASS_LABELS_EN.get(code, code)
    _, _, color_hex, _ = DANGER_INFO.get(code, ('', '', '#ffffff', 0))
    rgb_color = tuple(int(color_hex.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

    bar_h = max(52, int(h * 0.08))
    y1 = h - bar_h

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, 'RGBA')
    draw.rectangle((0, y1, w, h), fill=rgb_color + (225,))

    label = f'{en} | {zh}   {conf * 100:.1f}%'
    font_size = max(18, min(32, int(w / 28)))
    font = find_cjk_font(font_size)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = max(12, (w - tw) // 2)
    ty = y1 + max(8, (bar_h - th) // 2)
    draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0, 200), font=font)
    draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)

    return cv2.cvtColor(np.asarray(img.convert('RGB')), cv2.COLOR_RGB2BGR)


def annotate_frame_rgb(frame_bgr: np.ndarray, pred: dict) -> np.ndarray:
    return cv2.cvtColor(annotate_frame_bgr(frame_bgr.copy(), pred), cv2.COLOR_BGR2RGB)


def save_annotated_video_path() -> str:
    out_dir = ROOT / 'output'
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    return str(out_dir / f'annotated_{stamp}.mp4')


def infer_video(video_media, video_file, fast_mode: bool, video_stride: int):
    video_path = normalize_upload_path(video_media) or normalize_upload_path(video_file)
    if not video_path:
        yield None, '', '', '', empty_panel_msg('请先上传视频文件（推荐下方「选择视频文件」）', error=True), None
        return

    engine = get_engine()
    engine.reset_buffer()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield None, '', '', '', empty_panel_msg(f'无法打开视频: {video_path}', error=True), None
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_path = save_annotated_video_path()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        out_path = out_path.replace('.mp4', '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        yield None, '', '', '', empty_panel_msg('无法创建输出视频（请安装 ffmpeg 或检查 OpenCV）', error=True), None
        return

    stride = max(1, int(video_stride))
    last_pred = None
    idx = 0
    preview_rgb = None
    en = conf = danger = ''
    mode_note = '快速单帧' if fast_mode else '完整 5+9 帧 Tier-1（与命令行一致）'
    html = empty_panel_msg(f'视频分析中…  模式: {mode_note}')

    yield None, '分析中…', '', '', html, None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % stride == 0:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            t0 = time.perf_counter()
            use_buffer = engine.is_temporal and not fast_mode
            last_pred = engine.predict_pil(pil, use_buffer=use_buffer, fast_mode=fast_mode)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            preview_rgb = annotate_frame_rgb(frame, last_pred)
            frame_info = f'帧 {idx + 1}' + (f' / {n}' if n > 0 else '') + f'  ·  间隔 {stride} 帧  ·  {mode_note}'
            en, zh, conf, danger, html = format_result(
                last_pred, elapsed_ms=elapsed_ms, frame_info=frame_info,
            )
            yield preview_rgb, en, conf, danger, html, None

        if last_pred:
            annotate_frame_bgr(frame, last_pred)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()

    if last_pred is None:
        yield None, '', '', '', empty_panel_msg('视频无有效帧', error=True), None
        return

    browser_path = reencode_for_browser(out_path)
    saved_note = f'已保存: {browser_path}'
    en, zh, conf, danger, html = format_result(
        last_pred,
        frame_info=f'完成，共 {idx} 帧  ·  {saved_note}',
    )
    yield preview_rgb, en, conf, danger, html, browser_path


UI_CSS = """
.result-panel {
  background: #ffffff !important;
  color: #0f172a !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 12px;
  padding: 18px 20px;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
.result-title { font-size: 28px; font-weight: 700; margin-bottom: 6px; }
.result-sub { font-size: 17px; color: #334155 !important; margin-bottom: 14px; line-height: 1.5; }
.result-label { font-size: 13px; color: #64748b !important; margin-bottom: 4px; }
.result-value { font-size: 30px; font-weight: 700; color: #0f172a !important; margin-bottom: 14px; }
.result-block { margin-bottom: 8px; }
.result-top3 { border-top: 1px solid #e2e8f0; padding-top: 12px; margin-top: 4px; }
.result-column .gr-textbox label,
.result-column .gr-textbox input,
.result-column .gr-textbox textarea {
  color: #0f172a !important;
  background: #ffffff !important;
}
.result-column .gr-textbox input,
.result-column .gr-textbox textarea {
  font-size: 18px !important;
  font-weight: 600 !important;
  border: 1px solid #cbd5e1 !important;
}
.scene-image img { object-fit: contain !important; background: #1e293b; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title='驾驶员分心检测 · Tier-1',
        theme=gr.themes.Soft(primary_hue='blue'),
        css=UI_CSS,
    ) as demo:
        gr.Markdown(
            '# 驾驶员分心行为检测\n'
            '**Five-ROI v2 Temporal + Tier-1** · 验证集准确率 81.80%'
        )

        with gr.Tabs():
            with gr.TabItem('图片 / 摄像头'):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        gr.Markdown(
                            '### 当前驾驶画面\n'
                            '**摄像头**：先点左侧 📷 拍照定格画面，再点「立即检测」。'
                            '上传图片会在选图后自动推理。'
                        )
                        img_in = gr.Image(
                            label='上传 / 拍照',
                            sources=['upload', 'webcam'],
                            type='numpy',
                            height=420,
                            elem_classes=['scene-image'],
                        )
                        use_buf = gr.Checkbox(
                            label='视频流模式（连续拍照/多帧时勾选，单张图片请关闭）',
                            value=False,
                        )
                        fast_mode = gr.Checkbox(
                            label='快速推理（单帧，精度低于命令行；默认关闭=完整 5+9 帧 Tier-1）',
                            value=False,
                        )
                        btn_img = gr.Button('立即检测', variant='primary')

                    with gr.Column(scale=2, elem_classes=['result-column']):
                        gr.Markdown('### 预测结果')
                        out_en = gr.Textbox(label='Prediction', interactive=False)
                        out_conf = gr.Textbox(label='置信度 Confidence', interactive=False)
                        out_danger = gr.Textbox(label='危险等级 Danger Level', interactive=False)
                        out_panel = gr.HTML(value=empty_panel_msg('上传图片后将自动推理'))

                img_outs = [out_en, out_conf, out_danger, out_panel]
                img_in.change(
                    infer_image_auto,
                    inputs=[img_in, use_buf, fast_mode],
                    outputs=img_outs,
                )
                btn_img.click(
                    infer_image_button,
                    inputs=[img_in, use_buf, fast_mode],
                    outputs=img_outs,
                    show_progress='full',
                )

            with gr.TabItem('视频文件'):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown('### 上传行车视频')
                        gr.Markdown(
                            '推荐用 **选择视频文件** 上传。'
                            '默认使用与 `slideshow_driver_inference.py --force-temporal` 相同的完整 Tier-1 推理。'
                        )
                        vid_file = gr.File(
                            label='选择视频文件（推荐）',
                            file_types=['.mp4', '.avi', '.mov', '.mkv', '.webm'],
                            type='filepath',
                        )
                        vid_in = gr.Video(
                            label='或拖拽到此处（部分格式可能无法预览）',
                            sources=['upload'],
                        )
                        vid_fast = gr.Checkbox(
                            label='快速推理（单帧，精度略降；默认关闭=完整 Tier-1）',
                            value=False,
                        )
                        vid_stride = gr.Slider(
                            minimum=1, maximum=30, value=5, step=1,
                            label='视频推理间隔（每 N 帧更新一次预测，不影响已推理帧的标注）',
                        )
                        btn_vid = gr.Button('分析视频并导出标注', variant='primary')
                        gr.Markdown('### 完整标注视频')
                        vid_out = gr.Video(
                            label='分析完成后在左下角播放（同时保存到 output/ 目录）',
                        )
                    with gr.Column(scale=2, elem_classes=['result-column']):
                        gr.Markdown('### 实时推理')
                        vid_live = gr.Image(
                            label='当前帧（同步刷新）',
                            type='numpy',
                            height=360,
                            interactive=False,
                            elem_classes=['scene-image'],
                        )
                        v_en = gr.Textbox(label='Prediction (当前帧)', interactive=False)
                        v_conf = gr.Textbox(label='置信度', interactive=False)
                        v_danger = gr.Textbox(label='危险等级', interactive=False)
                        v_panel = gr.HTML(value=empty_panel_msg('上传视频后点击「分析视频」，右侧实时刷新'))

                btn_vid.click(
                    infer_video,
                    inputs=[vid_in, vid_file, vid_fast, vid_stride],
                    outputs=[vid_live, v_en, v_conf, v_danger, v_panel, vid_out],
                )

        gr.Markdown(
            '---\n'
            '**精度说明**：界面默认与命令行 `slideshow_driver_inference.py --device cuda --force-temporal` 一致'
            '（5 帧时序 + Tier-1 九帧长窗口）。勾选「快速推理」会改为单帧，结果可能与之前 batch 脚本不同。\n\n'
            '**危险等级**：Low 安全驾驶 · Medium 喝水/调收音机/交谈 · High 手机/伸手/化妆等'
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description='Driver distraction detection UI (Tier-1)')
    parser.add_argument('--ckpt', default=DEFAULT_CKPT)
    parser.add_argument('--device', default='auto', help='auto | cuda | cpu')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true', help='Gradio 公网分享链接')
    args = parser.parse_args()

    global ENGINE
    ENGINE = DistractionEngine(args.ckpt, args.device)

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == '__main__':
    main()
