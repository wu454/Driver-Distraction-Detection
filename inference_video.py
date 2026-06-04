#!/usr/bin/env python3
"""
AVI video inference — five-ROI temporal model + tier1 deploy.

Each frame is split into four quadrants; only the selected quadrant is cropped
and passed to the model (default: quadrant 4 = bottom-right).

Layout (image origin at top-left):
    Q2 (top-left)     | Q1 (top-right)
    ------------------+------------------
    Q3 (bottom-left)  | Q4 (bottom-right)  <- default

Examples:
    python inference_video.py --video test.avi
    python inference_video.py --video test.avi --output test_output_q4.avi --quadrant 4
    python inference_video.py --video test.avi --no-tier1
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image

from predict_five_roi import (
    DEFAULT_CKPT,
    load_model,
    predict_temporal_clip,
)
from roi_config import (
    CONSOLE_ROI,
    FACE_ROI,
    LEFT_HAND_ROI,
    MIRROR_ROI,
    RIGHT_HAND_ROI,
    TEMPORAL_HALF_WINDOW_DEFAULT,
    WHEEL_ROI,
)
from tier1_inference import frame_indices, tier1_long_half_window

# quadrant index -> (name, row, col) on [H, W, C] array
# row 0 = top, row 1 = bottom; col 0 = left, col 1 = right
QUADRANT_LAYOUT = {
    1: ('top-right', 0, 1),
    2: ('top-left', 0, 0),
    3: ('bottom-left', 1, 0),
    4: ('bottom-right', 1, 1),
}

ROI_DRAW = (
    ('face', FACE_ROI, (255, 90, 90)),
    ('left', LEFT_HAND_ROI, (90, 255, 90)),
    ('right', RIGHT_HAND_ROI, (90, 200, 255)),
    ('wheel', WHEEL_ROI, (255, 200, 90)),
    ('mirror', MIRROR_ROI, (255, 90, 255)),
    ('console', CONSOLE_ROI, (200, 200, 90)),
)

SEQ_LEN = 2 * TEMPORAL_HALF_WINDOW_DEFAULT + 1


def parse_args():
    parser = argparse.ArgumentParser(
        description='AVI video inference (five-ROI temporal + tier1, quadrant crop)',
    )
    parser.add_argument('--video', default='test.avi', help='Input AVI/video path')
    parser.add_argument(
        '--output', default='test_output_q4.avi',
        help='Annotated output video (quadrant crop size)',
    )
    parser.add_argument('--ckpt', default=DEFAULT_CKPT, help='Model checkpoint')
    parser.add_argument('--device', default='cuda', help='cuda, cuda:0, or cpu')
    parser.add_argument(
        '--quadrant', type=int, default=4, choices=[1, 2, 3, 4],
        help='Which quadrant to crop and infer (default: 4 = bottom-right)',
    )
    parser.add_argument(
        '--tier1', action='store_true',
        help='Tier-1 hybrid gates + c0 9f blend (default for temporal ckpt)',
    )
    parser.add_argument('--no-tier1', action='store_true', help='Disable tier-1 gates')
    parser.add_argument(
        '--print-every', type=int, default=30,
        help='Print prediction every N frames (0 = every frame)',
    )
    parser.add_argument(
        '--max-frames', type=int, default=0,
        help='Process at most N frames (0 = entire video)',
    )
    parser.add_argument(
        '--json', default=None, metavar='PATH',
        help='Optional JSON log of per-frame predictions',
    )
    return parser.parse_args()


def log(msg: str):
    print(msg, flush=True)


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith('cuda') and not torch.cuda.is_available():
        log('[WARN] CUDA unavailable, using CPU')
        return torch.device('cpu')
    return torch.device(device_str)


def split_quadrants(frame_rgb: np.ndarray) -> dict[int, np.ndarray]:
    """Split one RGB frame into four equal quadrants."""
    h, w = frame_rgb.shape[:2]
    mid_h, mid_w = h // 2, w // 2
    return {
        1: frame_rgb[0:mid_h, mid_w:w].copy(),
        2: frame_rgb[0:mid_h, 0:mid_w].copy(),
        3: frame_rgb[mid_h:h, 0:mid_w].copy(),
        4: frame_rgb[mid_h:h, mid_w:w].copy(),
    }


def extract_quadrant(frame_rgb: np.ndarray, quadrant: int) -> np.ndarray:
    if quadrant not in QUADRANT_LAYOUT:
        raise ValueError(f'quadrant must be 1-4, got {quadrant}')
    return split_quadrants(frame_rgb)[quadrant]


def quadrant_output_size(full_width: int, full_height: int) -> tuple[int, int]:
    return full_width // 2, full_height // 2


def roi_to_pixels(roi_frac, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi_frac
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def draw_five_rois(display: np.ndarray) -> np.ndarray:
    """Draw five-ROI boxes on RGB numpy image."""
    h, w = display.shape[:2]
    for _name, roi, color in ROI_DRAW:
        x1, y1, x2, y2 = roi_to_pixels(roi, w, h)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
    return display


def build_clips(frame_buffer: list[Image.Image], half_window: int) -> list[Image.Image]:
    """Temporal clip from rolling buffer (edge-clamped)."""
    center = len(frame_buffer) - 1
    length = len(frame_buffer)
    idxs = frame_indices(center, length, half_window)
    return [frame_buffer[i] for i in idxs]


def put_label(display: np.ndarray, text: str, y: int = 36, color=(0, 255, 0)):
    cv2.putText(
        display, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, color, 2, cv2.LINE_AA,
    )


def resolve_ckpt(path: str) -> str:
    if os.path.isfile(path):
        return path
    for alt in (
        './backup/five_roi_v2_temporal_tier1/model_best.pth',
        './experiments/exp_five_roi_v2_temporal/model_best.pth',
    ):
        if os.path.isfile(alt):
            return alt
    return path


def main():
    args = parse_args()
    device = resolve_device(args.device)
    ckpt_path = resolve_ckpt(args.ckpt)

    use_tier1 = not args.no_tier1
    q_name = QUADRANT_LAYOUT[args.quadrant][0]
    hw5 = TEMPORAL_HALF_WINDOW_DEFAULT
    hw9 = tier1_long_half_window()

    log(f'[INFO] Device: {device}')
    log(f'[INFO] Checkpoint: {ckpt_path}')
    log(f'[INFO] Inference region: quadrant {args.quadrant} ({q_name})')
    log(f'[INFO] Gates: {"tier1-hybrid" if use_tier1 else "v2"}  temporal={SEQ_LEN}f')

    if not os.path.isfile(args.video):
        raise FileNotFoundError(f'Video not found: {args.video}')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    model, class_names, ckpt = load_model(ckpt_path, device)
    is_temporal = ckpt.get('model_type') == 'TemporalFiveModel'
    if not is_temporal:
        log('[WARN] Checkpoint is not TemporalFiveModel; using single-frame slices as T=1 clips')

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {args.video}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = quadrant_output_size(full_w, full_h)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Failed to create output video: {args.output}')

    target_frames = total_frames if args.max_frames <= 0 else min(args.max_frames, total_frames)
    log(
        f'[INFO] Input: {args.video} | full={full_w}x{full_h} | crop={out_w}x{out_h} | '
        f'fps={fps:.1f} | frames≈{total_frames} | will process={target_frames}'
    )

    frame_buffer: list[Image.Image] = []
    frame_log: list[dict] = []
    frame_idx = 0

    with torch.inference_mode():
        while frame_idx < target_frames:
            ok, bgr = cap.read()
            if not ok:
                break

            full_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = extract_quadrant(full_rgb, args.quadrant)
            display = draw_five_rois(rgb.copy())
            pil = Image.fromarray(rgb)
            frame_buffer.append(pil)

            need = SEQ_LEN if is_temporal else 1
            if len(frame_buffer) < need:
                put_label(
                    display,
                    f'Q{args.quadrant} warming up ({len(frame_buffer)}/{need})',
                    color=(200, 200, 200),
                )
                pred_info = None
            else:
                if is_temporal:
                    clip5 = build_clips(frame_buffer, hw5)
                    clip9 = build_clips(frame_buffer, hw9) if use_tier1 else None
                    result = predict_temporal_clip(
                        model, clip5, device, class_names,
                        is_temporal=True, tier1=use_tier1, clip_frames_long=clip9,
                    )
                else:
                    result = predict_temporal_clip(
                        model, [pil], device, class_names,
                        is_temporal=False, tier1=use_tier1,
                    )

                label = (
                    f"Q{args.quadrant} {result['pred_class']} "
                    f"({result['confidence']:.0%})  "
                    f"M:{result['mirror_score']:.2f} P:{result['phone_score']:.2f}"
                )
                put_label(display, label)
                put_label(
                    display,
                    f"top3: {', '.join(t['class'] for t in result['top3'][:3])}",
                    y=68, color=(180, 255, 180),
                )

                if args.print_every == 0 or frame_idx % args.print_every == 0:
                    log(f'[Frame {frame_idx:05d}] {label}')

                pred_info = {
                    'frame': frame_idx,
                    'quadrant': args.quadrant,
                    'pred_class': result['pred_class'],
                    'confidence': result['confidence'],
                    'mirror_score': result['mirror_score'],
                    'phone_score': result['phone_score'],
                    'top3': result['top3'],
                }
                frame_log.append(pred_info)

            writer.write(cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
            frame_idx += 1

            if frame_idx % 100 == 0 or frame_idx == target_frames:
                pct = 100.0 * frame_idx / max(target_frames, 1)
                log(f'[Progress] {frame_idx}/{target_frames} ({pct:.1f}%)')

    cap.release()
    writer.release()

    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({
                'video': args.video,
                'output': args.output,
                'checkpoint': ckpt_path,
                'quadrant': args.quadrant,
                'tier1': use_tier1,
                'frames_processed': frame_idx,
                'predictions': frame_log,
            }, f, indent=2, ensure_ascii=False)
        log(f'[OK] JSON log: {args.json}')

    log(f'[OK] Saved quadrant-{args.quadrant} video: {args.output} ({frame_idx} frames)')


if __name__ == '__main__':
    main()
