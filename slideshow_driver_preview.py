#!/usr/bin/env python3
"""
从 State Farm 数据集中为指定司机按类别均匀抽样，生成无推理的 MP4 幻灯片。

依赖: opencv-python, pillow, numpy（无需 PyTorch / 模型权重）

示例:
  python slideshow_driver_preview.py --subjects p021,p024
  python slideshow_driver_preview.py --subject p021 --max-per-class 10 --hold-seconds 1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

CLASS_ORDER = [f'c{i}' for i in range(10)]
PREVIEW_WINDOW = '驾驶员分心数据集幻灯片'
_CJK_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def resolve_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def default_data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in ('data/train', '../data/train'):
        candidate = resolve_path(os.path.join(here, rel))
        if os.path.isdir(candidate):
            return candidate
    return resolve_path(os.path.join(here, '../data/train'))


def default_mapping_csv() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in ('data/driver_imgs_list.csv', '../data/driver_imgs_list.csv'):
        candidate = resolve_path(os.path.join(here, rel))
        if os.path.isfile(candidate):
            return candidate
    return resolve_path(os.path.join(here, '../data/driver_imgs_list.csv'))


def parse_img_id(filename: str) -> int:
    m = re.search(r'(\d+)', filename)
    return int(m.group(1)) if m else 0


def check_dataset_paths(data_dir: str, csv_path: str) -> None:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f'找不到映射表 CSV: {csv_path}\n'
            f'请下载 data/driver_imgs_list.csv 并用 --mapping-csv 指定'
        )
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f'找不到图片目录: {data_dir}\n'
            f'请下载 data/train/ 并用 --data-dir 指定'
        )


def list_subjects(csv_path: str) -> list[str]:
    check_dataset_paths(default_data_dir(), csv_path)
    subjects = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            subjects.add(row['subject'])
    return sorted(subjects)


def load_driver_records(data_dir: str, csv_path: str, subject: str) -> list[dict]:
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
    groups: dict[str, list[dict]] = defaultdict(list)
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
    """contiguous: 时序连续段（默认）；spread: 全类均匀抽样。"""
    if max_n <= 0 or len(class_recs) <= max_n:
        return class_recs

    if mode == 'spread':
        idxs = np.linspace(0, len(class_recs) - 1, max_n, dtype=int)
        return [class_recs[int(i)] for i in idxs]

    if mode != 'contiguous':
        raise ValueError(f'未知 sample_mode: {mode}')

    block = max_n
    max_start = len(class_recs) - block
    if segment_start == 'random':
        import random
        start = random.randint(0, max_start) if max_start > 0 else 0
    else:
        start = max_start // 2
    return class_recs[start:start + block]


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
            font = ImageFont.truetype(path, size)
            _CJK_FONT_CACHE[size] = font
            return font
    font = ImageFont.load_default()
    _CJK_FONT_CACHE[size] = font
    print('[WARN] 未找到中文字体，标签可能显示为方块')
    return font


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


def draw_info_banner(
    frame_bgr: np.ndarray,
    subject: str,
    true_class: str,
    index_in_class: int,
    total_in_class: int,
) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, 'RGBA')
    h, w = frame_bgr.shape[:2]

    lines = [
        f'司机: {subject}',
        f'类别: {class_display_name(true_class)}',
        f'本类: {index_in_class}/{total_in_class}',
    ]
    font_size = max(18, min(32, int(w / 28)))
    font = find_cjk_font(font_size)
    pad = 12
    line_h = font_size + 10
    text_sizes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    max_tw = max(bb[2] - bb[0] for bb in text_sizes)
    box_w = min(w - 16, max_tw + pad * 2)
    box_h = pad * 2 + line_h * len(lines)

    draw.rectangle((8, 8, 8 + box_w, 8 + box_h), fill=(0, 0, 0, 180))
    colors = [(120, 220, 255), (240, 240, 240), (200, 200, 200)]
    y = 8 + pad
    for i, line in enumerate(lines):
        draw.text((8 + pad, y + i * line_h), line, fill=colors[i] + (255,), font=font)

    out_rgb = np.asarray(img.convert('RGB'))
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def draw_title_card(width: int, height: int, title: str, subtitle: str = '') -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (24, 28, 36)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    title_font = find_cjk_font(max(28, min(48, width // 18)))
    sub_font = find_cjk_font(max(18, min(28, width // 28)))

    tb = draw.textbbox((0, 0), title, font=title_font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text(((width - tw) // 2, height // 2 - th - 10), title, fill=(240, 240, 240), font=title_font)
    if subtitle:
        sb = draw.textbbox((0, 0), subtitle, font=sub_font)
        sw = sb[2] - sb[0]
        draw.text(((width - sw) // 2, height // 2 + 16), subtitle, fill=(180, 190, 200), font=sub_font)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def video_fourcc(output_path: str) -> int:
    ext = os.path.splitext(output_path)[1].lower()
    if ext in ('.mp4', '.m4v', '.mov'):
        return cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter_fourcc(*'XVID')


def open_slideshow_writer(output_path: str, width: int, height: int, fps: float, hold_seconds: float):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    writer = cv2.VideoWriter(output_path, video_fourcc(output_path), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'无法创建视频文件: {output_path}')
    repeat = max(1, int(round(fps * hold_seconds)))
    return writer, repeat


def default_show_preview() -> bool:
    if platform.system() == 'Windows':
        return True
    return bool(os.environ.get('DISPLAY'))


def cv2_gui_available() -> bool:
    try:
        probe = np.zeros((8, 8, 3), dtype=np.uint8)
        cv2.imshow('__opencv_gui_probe__', probe)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return True
    except cv2.error:
        return False


class PreviewSession:
    def __init__(self, title: str, want_show: bool):
        self.enabled = want_show and cv2_gui_available()
        self.title = title
        if want_show and self.enabled:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        elif want_show:
            print('[WARN] 当前 OpenCV 无 GUI，已跳过实时预览（视频仍会正常保存）')

    def show(self, frame_bgr: np.ndarray, delay_ms: int) -> bool:
        if not self.enabled:
            return True
        cv2.imshow(self.title, frame_bgr)
        key = cv2.waitKey(delay_ms) & 0xFF
        return key not in (27, ord('q'), ord('Q'))

    def close(self) -> None:
        if self.enabled:
            cv2.destroyAllWindows()


def append_slide(
    writer: cv2.VideoWriter,
    frame_bgr: np.ndarray,
    repeat: int,
    *,
    preview: PreviewSession | None,
    fps: float,
) -> bool:
    delay_ms = max(1, int(round(1000.0 / fps)))
    for _ in range(repeat):
        writer.write(frame_bgr)
        if preview is not None and preview.enabled:
            if not preview.show(frame_bgr, delay_ms):
                return False
    return True


def build_work_items(
    records: list[dict],
    max_per_class: int,
    *,
    sample_mode: str = 'contiguous',
    segment_start: str = 'middle',
) -> list[tuple[str, int, dict, int]]:
    by_class = group_by_class(records)
    items: list[tuple[str, int, dict, int]] = []
    for cls in CLASS_ORDER:
        if cls not in by_class:
            continue
        sampled = sample_class_records(
            by_class[cls], max_per_class,
            mode=sample_mode, segment_start=segment_start,
        )
        total = len(sampled)
        for idx, rec in enumerate(sampled, start=1):
            items.append((cls, idx, rec, total))
    return items


def default_output_path(subject: str, out_dir: str) -> str:
    return os.path.join(out_dir, f'{subject}_数据集幻灯片.mp4')


def run_subject_slideshow(
    subject: str,
    args,
    preview: PreviewSession | None,
) -> dict:
    records = load_driver_records(args.data_dir, args.mapping_csv, subject)
    if not records:
        raise ValueError(f'司机 {subject} 没有可用图片（检查 data/train 与 CSV 是否匹配）')

    work_items = build_work_items(
        records, args.max_per_class,
        sample_mode=args.sample_mode,
        segment_start=args.segment_start,
    )
    if not work_items:
        raise ValueError(f'司机 {subject} 没有可写入幻灯片的图片')

    mode_notes = {
        'contiguous': f'每类 {args.max_per_class} 张（时序连续段）',
        'spread': f'每类 {args.max_per_class} 张（全段均匀抽样）',
    }
    per_class_note = mode_notes.get(args.sample_mode, '') if args.max_per_class > 0 else '每类全部'
    output_path = args.output
    if output_path is None:
        output_path = default_output_path(subject, args.out_dir)
    elif len(args.subjects) == 1 and '{subject}' in output_path:
        output_path = output_path.format(subject=subject)

    print(
        f'[OK] 司机={subject}  帧数={len(work_items)}  {per_class_note}\n'
        f'     每张停留={args.hold_seconds}s  输出={output_path}'
    )

    writer = None
    repeat = 0
    manifest: list[dict] = []
    aborted = False

    if args.title_card:
        title = draw_title_card(args.width, args.height, f'司机 {subject}', 'State Farm 分心行为数据集')
        writer, repeat = open_slideshow_writer(output_path, args.width, args.height, args.fps, args.hold_seconds)
        if not append_slide(writer, title, repeat, preview=preview, fps=args.fps):
            aborted = True

    last_class: str | None = None
    for done, (cls, idx_in_class, rec, total_in_class) in enumerate(work_items, start=1):
        if args.section_title and cls != last_class:
            section = draw_title_card(
                args.width, args.height,
                class_display_name(cls),
                f'司机 {subject}  ·  {idx_in_class}/{total_in_class}',
            )
            if writer is None:
                writer, repeat = open_slideshow_writer(output_path, args.width, args.height, args.fps, args.hold_seconds)
            if not append_slide(writer, section, repeat, preview=preview, fps=args.fps):
                aborted = True
                break
            last_class = cls

        pil = Image.open(rec['path']).convert('RGB')
        bgr = resize_keep_aspect(
            cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR),
            args.width, args.height,
        )
        annotated = draw_info_banner(bgr, subject, rec['true_class'], idx_in_class, total_in_class)

        if writer is None:
            h, w = annotated.shape[:2]
            writer, repeat = open_slideshow_writer(output_path, w, h, args.fps, args.hold_seconds)

        if not append_slide(writer, annotated, repeat, preview=preview, fps=args.fps):
            aborted = True
            break

        manifest.append({
            'subject': subject,
            'path': rec['path'],
            'img': rec['img'],
            'true_class': rec['true_class'],
            'true_label_zh': CLASS_LABELS_ZH.get(rec['true_class'], rec['true_class']),
            'index_in_class': idx_in_class,
            'total_in_class': total_in_class,
        })

        if done % max(1, len(work_items) // 10) == 0 or done == len(work_items):
            print(f'  [{subject}] {done}/{len(work_items)}')

    if writer is None:
        raise RuntimeError(f'司机 {subject} 没有处理任何帧')
    writer.release()

    status = '（已提前结束）' if aborted else ''
    print(f'[OK] 已保存: {output_path}{status}')

    summary = {
        'subject': subject,
        'images': len(manifest),
        'output_video': output_path,
        'hold_seconds': args.hold_seconds,
        'max_per_class': args.max_per_class,
        'aborted': aborted,
        'frames': manifest,
    }

    if args.json:
        json_path = args.json
        if '{subject}' in json_path:
            json_path = json_path.format(subject=subject)
        if not os.path.isabs(json_path) and os.path.dirname(json_path) == '':
            json_path = os.path.join(args.out_dir, json_path)
        os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                **summary,
                'created_at': datetime.now().isoformat(timespec='seconds'),
                'data_dir': args.data_dir,
                'mapping_csv': args.mapping_csv,
            }, f, indent=2, ensure_ascii=False)
        print(f'[OK] JSON: {json_path}')

    return summary


def parse_subjects(raw: str | None, subject: str | None) -> list[str]:
    if raw:
        parts = [s.strip().lower() for s in raw.split(',') if s.strip()]
    elif subject:
        parts = [subject.strip().lower()]
    else:
        parts = ['p021', 'p024']
    normalized = []
    for s in parts:
        if not s.startswith('p'):
            s = f'p{s}'
        normalized.append(s)
    return normalized


def main():
    parser = argparse.ArgumentParser(description='司机数据集幻灯片（无推理，按类抽样拼接 MP4）')
    parser.add_argument('--subjects', default='p021,p024', help='逗号分隔司机编号，默认 p021,p024')
    parser.add_argument('--subject', default=None, help='单个司机（等同 --subjects 只填一个）')
    parser.add_argument('--list-subjects', action='store_true', help='列出 CSV 中全部司机编号')
    parser.add_argument('--data-dir', default=None, help='图片根目录，默认自动找 data/train')
    parser.add_argument('--mapping-csv', default=None, help='driver_imgs_list.csv 路径')
    parser.add_argument('--out-dir', default='.', help='输出目录（默认当前目录）')
    parser.add_argument('--output', default=None, help='输出视频；多司机时可用 {subject} 占位')
    parser.add_argument('--json', default='{subject}_preview_slideshow.json', help='清单 JSON；{subject} 占位')
    parser.add_argument('--max-per-class', type=int, default=15, help='每类最多 N 张（0=全部）')
    parser.add_argument(
        '--sample-mode', choices=('contiguous', 'spread'), default='contiguous',
        help='contiguous=每类连续段（默认）；spread=均匀抽样',
    )
    parser.add_argument(
        '--segment-start', choices=('middle', 'random'), default='middle',
        help='连续段起点（contiguous 模式）',
    )
    parser.add_argument('--hold-seconds', type=float, default=1.0, help='每张停留秒数（默认 1）')
    parser.add_argument('--fps', type=float, default=25.0)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--title-card', action='store_true', help='片头显示司机标题卡')
    parser.add_argument('--section-title', action='store_true', help='每个类别切换时插入标题卡')
    parser.add_argument('--show', action='store_true', help='实时预览（Windows 默认开启）')
    parser.add_argument('--no-show', action='store_true', help='关闭实时预览')
    args = parser.parse_args()

    args.data_dir = resolve_path(args.data_dir or default_data_dir())
    args.mapping_csv = resolve_path(args.mapping_csv or default_mapping_csv())
    args.out_dir = resolve_path(args.out_dir)

    if args.list_subjects:
        for s in list_subjects(args.mapping_csv):
            print(s)
        return

    check_dataset_paths(args.data_dir, args.mapping_csv)
    args.subjects = parse_subjects(args.subjects if args.subject is None else None, args.subject)

    if not args.show and not args.no_show:
        args.show = default_show_preview()
    preview = PreviewSession(PREVIEW_WINDOW, args.show and not args.no_show)

    summaries = []
    for subject in args.subjects:
        if len(args.subjects) > 1:
            args.output = None
        summaries.append(run_subject_slideshow(subject, args, preview))
    preview.close()

    print('\n[完成] 共生成 {} 个视频:'.format(len(summaries)))
    for s in summaries:
        print(f'  - {s["subject"]}: {s["output_video"]} ({s["images"]} 帧)')


if __name__ == '__main__':
    main()
