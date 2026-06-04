#!/usr/bin/env python3
"""
Batch inference on one driver's images from the State Farm dataset,
export MP4 slideshow with Chinese on-screen labels, optional live preview.

Example:
  python slideshow_driver_inference.py --subject p021 --device cuda --force-temporal
  python slideshow_driver_inference.py --subject p021 --hold-seconds 0.2 --output 司机幻灯片.mp4 --show
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import re
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from predict_five_roi import load_model, predict_temporal_clip

DEFAULT_CKPT = './model_best.pth'
from roi_config import WHEEL_ROI
from tier1_inference import frame_indices, tier1_long_half_window

DATA_DIR = './data/train'
MAPPING_CSV = './data/driver_imgs_list.csv'
DEFAULT_OUTPUT = '司机幻灯片.mp4'
PREVIEW_WINDOW = '驾驶员分心检测'

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

# RGB for PIL
COLOR_OK = (80, 220, 80)
COLOR_WRONG = (255, 80, 80)
COLOR_NEUTRAL = (240, 240, 240)
COLOR_SUB = (220, 220, 220)

_CJK_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def parse_img_id(filename: str) -> int:
    m = re.search(r'(\d+)', filename)
    return int(m.group(1)) if m else 0


def resolve_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def check_dataset_paths(data_dir: str, csv_path: str) -> None:
    csv_abs = resolve_path(csv_path)
    data_abs = resolve_path(data_dir)
    if not os.path.isfile(csv_abs):
        raise FileNotFoundError(
            f'找不到映射表 CSV: {csv_path}\n'
            f'  绝对路径: {csv_abs}\n'
            f'请下载 data/driver_imgs_list.csv 并用 --mapping-csv 指定，例如:\n'
            f'  --mapping-csv "../data/driver_imgs_list.csv"'
        )
    if not os.path.isdir(data_abs):
        raise FileNotFoundError(
            f'找不到图片目录: {data_dir}\n'
            f'  绝对路径: {data_abs}\n'
            f'请下载 data/train/ 并用 --data-dir 指定'
        )


def list_subjects(csv_path: str) -> list[str]:
    check_dataset_paths(DATA_DIR, csv_path)
    subjects = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            subjects.add(row['subject'])
    return sorted(subjects)


def load_driver_records(data_dir: str, csv_path: str, subject: str) -> list[dict]:
    check_dataset_paths(data_dir, csv_path)
    records = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['subject'] != subject:
                continue
            img_name = row['img']
            cls = row['classname']
            path = os.path.join(data_dir, cls, img_name)
            if not os.path.isfile(path):
                continue
            records.append({
                'path': path,
                'true_class': cls,
                'img': img_name,
                'img_id': parse_img_id(img_name),
                'subject': subject,
            })
    records.sort(key=lambda r: (r['true_class'], r['img_id']))
    return records


def group_by_class(records: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for r in records:
        groups[r['true_class']].append(r)
    for cls in groups:
        groups[cls].sort(key=lambda r: r['img_id'])
    return dict(groups)


def sample_class_records(
    class_recs: list[dict],
    max_n: int,
    *,
    mode: str = 'contiguous',
    segment_start: str = 'middle',
) -> list[dict]:
    """从一类图片中抽样。

    contiguous（默认）: 取 img_id 排序后连续的一段，5/9 帧时序窗口用的是真实相邻帧，推理更准。
    spread: 在整个类别时间轴上均匀抽 N 张，幻灯片覆盖更全，但时序邻帧跨度大、准确率偏低。
    """
    if max_n <= 0 or len(class_recs) <= max_n:
        return class_recs

    if mode == 'spread':
        idxs = np.linspace(0, len(class_recs) - 1, max_n, dtype=int)
        return [class_recs[int(i)] for i in idxs]

    if mode != 'contiguous':
        raise ValueError(f'未知 sample_mode: {mode}（可选 contiguous / spread）')

    block = max_n
    max_start = len(class_recs) - block
    if segment_start == 'random':
        import random
        start = random.randint(0, max_start) if max_start > 0 else 0
    else:
        start = max_start // 2
    return class_recs[start:start + block]


def build_temporal_pils(class_records: list[dict], center_idx: int, half_window: int) -> list[Image.Image]:
    paths = [r['path'] for r in class_records]
    idxs = frame_indices(center_idx, len(paths), half_window)
    return [Image.open(paths[i]).convert('RGB') for i in idxs]


def class_display_name(code: str) -> str:
    zh = CLASS_LABELS_ZH.get(code, code)
    return f'{code} {zh}'


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
    print('[WARN] 未找到中文字体，标签可能显示为方块；Windows 请确认 C:/Windows/Fonts/msyh.ttc 存在')
    return font


def roi_to_pixels(roi_frac, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi_frac
    return (
        int(round(x1 * w)), int(round(y1 * h)),
        int(round(x2 * w)), int(round(y2 * h)),
    )


def resize_keep_aspect(rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def draw_label_banner(
    frame_bgr: np.ndarray,
    pred_class: str,
    confidence: float,
    true_class: str | None = None,
    show_wheel_roi: bool = False,
) -> np.ndarray:
    """Top-left Chinese classification banner via PIL."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, 'RGBA')

    if show_wheel_roi:
        x1, y1, x2, y2 = roi_to_pixels(WHEEL_ROI, w, h)
        draw.rectangle((x1, y1, x2, y2), outline=(60, 180, 180, 220), width=1)

    pred_text = f'预测: {class_display_name(pred_class)}'
    conf_text = f'置信度: {confidence:.1%}'
    lines = [pred_text, conf_text]
    if true_class is not None:
        ok = pred_class == true_class
        mark = '正确' if ok else '错误'
        lines.append(f'真实: {class_display_name(true_class)}  [{mark}]')
        title_color = COLOR_OK if ok else COLOR_WRONG
    else:
        title_color = COLOR_NEUTRAL

    font_size = max(18, min(32, int(w / 28)))
    font = find_cjk_font(font_size)
    pad = 12
    line_h = font_size + 10
    text_sizes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    max_tw = max(bb[2] - bb[0] for bb in text_sizes)
    box_w = min(w - 16, max_tw + pad * 2)
    box_h = pad * 2 + line_h * len(lines)

    draw.rectangle((8, 8, 8 + box_w, 8 + box_h), fill=(0, 0, 0, 180))
    y = 8 + pad
    for i, line in enumerate(lines):
        color = title_color if i == 0 else COLOR_SUB
        draw.text((8 + pad, y + i * line_h), line, fill=color + (255,), font=font)

    out_rgb = np.asarray(img.convert('RGB'))
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def video_fourcc(output_path: str) -> int:
    ext = os.path.splitext(output_path)[1].lower()
    if ext in ('.mp4', '.m4v', '.mov'):
        return cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter_fourcc(*'XVID')


def open_slideshow_writer(output_path: str, width: int, height: int, fps: float, hold_seconds: float):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fourcc = video_fourcc(output_path)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'无法创建视频文件: {output_path}')
    repeat = max(1, int(round(fps * hold_seconds)))
    return writer, repeat


def default_show_preview() -> bool:
    if platform.system() == 'Windows':
        return True
    return bool(os.environ.get('DISPLAY'))


def cv2_gui_available() -> bool:
    """True when this OpenCV build supports imshow (not headless)."""
    try:
        probe = np.zeros((8, 8, 3), dtype=np.uint8)
        cv2.imshow('__opencv_gui_probe__', probe)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return True
    except cv2.error:
        return False


class TkPreview:
    """Fallback live preview when OpenCV has no HighGUI (common with opencv-python-headless)."""

    def __init__(self, title: str):
        import tkinter as tk
        from PIL import ImageTk

        self._tk = tk
        self._ImageTk = ImageTk
        self.root = tk.Tk()
        self.root.title(title)
        self.quit = False
        self.root.protocol('WM_DELETE_WINDOW', self._request_quit)
        for seq in ('<Escape>', '<q>', '<Q>'):
            self.root.bind(seq, self._request_quit_event)
        self.label = tk.Label(self.root)
        self.label.pack()
        self._photo = None

    def _request_quit(self) -> None:
        self.quit = True

    def _request_quit_event(self, _event) -> None:
        self._request_quit()

    def show_frame(self, frame_bgr: np.ndarray, delay_ms: int) -> bool:
        if self.quit:
            return False
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self._photo = self._ImageTk.PhotoImage(pil)
        self.label.config(image=self._photo)
        deadline = time.time() + delay_ms / 1000.0
        while time.time() < deadline:
            if self.quit:
                return False
            self.root.update()
            time.sleep(0.01)
        return True

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass


class PreviewSession:
    """OpenCV window if available, else Tkinter; otherwise preview off."""

    def __init__(self, title: str, want_show: bool):
        self.enabled = want_show
        self.mode: str | None = None
        self.title = title
        self._tk: TkPreview | None = None
        if not want_show:
            return
        if cv2_gui_available():
            self.mode = 'cv2'
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            print(f'[INFO] 实时预览: OpenCV 窗口「{title}」，按 Q 或 Esc 结束')
            return
        try:
            self._tk = TkPreview(title)
            self.mode = 'tk'
            print(
                f'[INFO] 实时预览: Tkinter 窗口「{title}」（当前 OpenCV 无 GUI，已自动切换）\n'
                f'       若需 OpenCV 窗口: pip uninstall opencv-python-headless -y && pip install opencv-python'
            )
        except Exception as exc:
            self.enabled = False
            print(
                f'[WARN] 无法开启预览 ({exc})，将继续生成视频文件\n'
                f'       也可加 --no-show 跳过预览'
            )

    def show(self, frame_bgr: np.ndarray, delay_ms: int) -> bool:
        if not self.enabled:
            return True
        if self.mode == 'cv2':
            cv2.imshow(self.title, frame_bgr)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                return False
            return True
        if self.mode == 'tk' and self._tk is not None:
            return self._tk.show_frame(frame_bgr, delay_ms)
        return True

    def close(self) -> None:
        if self.mode == 'cv2':
            cv2.destroyAllWindows()
        elif self._tk is not None:
            self._tk.close()


def append_slide(
    writer: cv2.VideoWriter,
    frame_bgr: np.ndarray,
    repeat: int,
    *,
    preview: PreviewSession | None,
    fps: float,
) -> bool:
    """Write repeated frames; optionally preview. Returns False if user quit."""
    delay_ms = max(1, int(round(1000.0 / fps)))
    for _ in range(repeat):
        writer.write(frame_bgr)
        if preview is not None and preview.enabled:
            if not preview.show(frame_bgr, delay_ms):
                print('[INFO] 用户中断预览，已保存已处理部分')
                return False
    return True


def run_inference(args):
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f'找不到模型: {args.ckpt}')
    check_dataset_paths(args.data_dir, args.mapping_csv)

    device = torch.device(
        'cpu' if args.device == 'cpu' or not torch.cuda.is_available() else args.device
    )
    use_single_frame = args.single_frame or (
        device.type == 'cpu' and not args.force_temporal
    )
    if device.type == 'cpu' and not use_single_frame:
        print('[WARN] CPU + 时序推理占用内存较大，建议加 --single-frame')

    want_show = args.show and not args.no_show
    preview = PreviewSession(PREVIEW_WINDOW, want_show)

    model, class_names, ckpt = load_model(args.ckpt, device)
    is_temporal = ckpt.get('model_type') == 'TemporalFiveModel'
    hw5 = 2
    hw9 = tier1_long_half_window()
    use_tier1 = not args.no_tier1

    records = load_driver_records(args.data_dir, args.mapping_csv, args.subject)
    if not records:
        raise ValueError(f'司机 {args.subject} 没有可用图片（检查 data/train 是否完整）')
    if args.max_images > 0:
        records = records[:args.max_images]

    by_class = group_by_class(records)
    sampled_by_class = {
        cls: sample_class_records(
            recs, args.max_per_class,
            mode=args.sample_mode,
            segment_start=args.segment_start,
        )
        for cls, recs in by_class.items()
    }

    class_order = sorted(sampled_by_class.keys())
    work_items: list[tuple[str, int, dict]] = []
    for cls in class_order:
        class_recs = sampled_by_class[cls]
        for center_idx, rec in enumerate(class_recs):
            work_items.append((cls, center_idx, rec))

    total_work = len(work_items)
    log_every = max(1, total_work // 20)
    mode_notes = {
        'contiguous': f'每类 {args.max_per_class} 张（时序连续段，利于 5/9 帧推理）',
        'spread': f'每类 {args.max_per_class} 张（全段均匀抽样）',
    }
    per_class_note = mode_notes.get(args.sample_mode, '') if args.max_per_class > 0 else '每类全部'

    print(f'[OK] 司机={args.subject}  图片数={total_work}  类别数={len(by_class)}  {per_class_note}')
    print(
        f'     时序={is_temporal and not use_single_frame}  tier1={use_tier1}  '
        f'设备={device}  每张停留={args.hold_seconds}s  输出={args.output}'
    )

    writer = None
    repeat = 0
    results: list[dict] = []
    correct = 0
    aborted = False

    for done, (cls, center_idx, rec) in enumerate(work_items, start=1):
        class_recs = sampled_by_class[cls]

        pil_center = Image.open(rec['path']).convert('RGB')
        with torch.inference_mode():
            if use_single_frame or not is_temporal:
                pred = predict_temporal_clip(
                    model, [pil_center], device, class_names,
                    is_temporal=False, tier1=use_tier1,
                )
            else:
                clip5 = build_temporal_pils(class_recs, center_idx, hw5)
                clip9 = build_temporal_pils(class_recs, center_idx, hw9) if use_tier1 else None
                pred = predict_temporal_clip(
                    model, clip5, device, class_names,
                    is_temporal=True, tier1=use_tier1, clip_frames_long=clip9,
                )
                del clip5
                if clip9 is not None:
                    del clip9

        bgr = resize_keep_aspect(
            cv2.cvtColor(np.asarray(pil_center), cv2.COLOR_RGB2BGR),
            args.width, args.height,
        )
        del pil_center
        annotated = draw_label_banner(
            bgr,
            pred['pred_class'],
            pred['confidence'],
            true_class=rec['true_class'],
            show_wheel_roi=args.show_wheel_roi,
        )
        del bgr

        if writer is None:
            h, w = annotated.shape[:2]
            writer, repeat = open_slideshow_writer(args.output, w, h, args.fps, args.hold_seconds)

        if not append_slide(writer, annotated, repeat, preview=preview, fps=args.fps):
            aborted = True
            del annotated
            break
        del annotated

        ok = pred['pred_class'] == rec['true_class']
        correct += int(ok)
        pred_zh = CLASS_LABELS_ZH.get(pred['pred_class'], pred['pred_class'])
        true_zh = CLASS_LABELS_ZH.get(rec['true_class'], rec['true_class'])
        results.append({
            'path': rec['path'],
            'true_class': rec['true_class'],
            'true_label_zh': true_zh,
            'pred_class': pred['pred_class'],
            'pred_label_zh': pred_zh,
            'confidence': pred['confidence'],
            'correct': ok,
        })
        del pred
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        if done % log_every == 0 or done == total_work:
            print(
                f'  [{done}/{total_work}] 准确率={correct/done*100:.1f}%  '
                f'最新: {true_zh} → {pred_zh} ({results[-1]["confidence"]:.1%})'
            )

    if writer is None:
        raise RuntimeError('没有处理任何帧')
    writer.release()
    preview.close()

    processed = len(results)
    acc = correct / max(processed, 1) * 100
    status = '（已提前结束）' if aborted else ''
    print(
        f'[OK] 视频已保存: {args.output}{status}\n'
        f'     幻灯片={processed}  每张={args.hold_seconds}s  准确率={acc:.2f}%'
    )

    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({
                'subject': args.subject,
                'checkpoint': args.ckpt,
                'tier1': use_tier1,
                'images': processed,
                'accuracy': round(acc, 2),
                'output_video': args.output,
                'hold_seconds': args.hold_seconds,
                'aborted': aborted,
                'created_at': datetime.now().isoformat(timespec='seconds'),
                'predictions': results,
            }, f, indent=2, ensure_ascii=False)
        print(f'[OK] JSON: {args.json}')


def main():
    parser = argparse.ArgumentParser(description='司机图片幻灯片推理（中文标注 + MP4 + 实时预览）')
    parser.add_argument('--subject', default='p021', help='司机编号，如 p021（--list-subjects 查看全部）')
    parser.add_argument('--list-subjects', action='store_true', help='列出所有司机编号')
    parser.add_argument('--data-dir', default=DATA_DIR)
    parser.add_argument('--mapping-csv', default=MAPPING_CSV)
    parser.add_argument('--ckpt', default=DEFAULT_CKPT)
    parser.add_argument('--output', default=None, help=f'输出视频（默认 {DEFAULT_OUTPUT}）')
    parser.add_argument('--json', default=None, help='可选：预测结果 JSON')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-tier1', action='store_true')
    parser.add_argument('--hold-seconds', type=float, default=0.2, help='每张图片停留秒数（默认 0.2，更快可用 0.15）')
    parser.add_argument('--fps', type=float, default=25.0)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--max-images', type=int, default=0, help='0=该司机全部图片')
    parser.add_argument('--max-per-class', type=int, default=15, help='每类最多 N 张（默认 15；0=每类全部）')
    parser.add_argument(
        '--sample-mode', choices=('contiguous', 'spread'), default='contiguous',
        help='contiguous=每类取 img_id 连续一段（时序推理更准，默认）；spread=全类均匀抽样',
    )
    parser.add_argument(
        '--segment-start', choices=('middle', 'random'), default='middle',
        help='连续段起点：middle=类别中间；random=随机（仅 contiguous 模式）',
    )
    parser.add_argument('--show-wheel-roi', action='store_true', help='绘制方向盘 ROI 框')
    parser.add_argument('--show', action='store_true', help='实时预览窗口（Windows 默认开启）')
    parser.add_argument('--no-show', action='store_true', help='关闭实时预览（服务器批处理推荐）')
    parser.add_argument(
        '--single-frame', action='store_true',
        help='单帧推理（CPU 默认，省内存）',
    )
    parser.add_argument(
        '--force-temporal', action='store_true',
        help='CPU 上也使用时序 5 帧（需更多内存）',
    )
    args = parser.parse_args()

    if args.list_subjects:
        for s in list_subjects(args.mapping_csv):
            print(s)
        return

    if args.output is None:
        args.output = DEFAULT_OUTPUT
    if args.json is None:
        args.json = f'logs/{args.subject}_slideshow.json'

    if not args.show and not args.no_show:
        args.show = default_show_preview()

    run_inference(args)


if __name__ == '__main__':
    main()
