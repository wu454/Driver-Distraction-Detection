#!/usr/bin/env python3
"""
Tier-1 inference on a single-view video (no quadrant crop).

Use after mp4_q2_mirror_prepare.py — each frame is already the driver ROI.

Example:
  python inference_prepared_video.py --video 3MDAD_q2_mirror.mp4 --device cuda
"""
from __future__ import annotations

import argparse
import gc
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image

from predict_five_roi import DEFAULT_CKPT, load_model, predict_temporal_clip
from roi_config import TEMPORAL_HALF_WINDOW_DEFAULT
from tier1_inference import frame_indices, tier1_long_half_window

CLASS_LABELS_EN = {
    'c0': 'safe driving',
    'c1': 'texting - right',
    'c2': 'phone call - right',
    'c3': 'texting - left',
    'c4': 'phone call - left',
    'c5': 'operating radio',
    'c6': 'drinking',
    'c7': 'reaching behind',
    'c8': 'hair and makeup',
    'c9': 'talking to passenger',
}

SEQ_LEN = 2 * TEMPORAL_HALF_WINDOW_DEFAULT + 1


def log(msg: str) -> None:
    print(msg, flush=True)


def class_display(code: str) -> str:
    return f"{code}: {CLASS_LABELS_EN.get(code, code)}"


def draw_banner(bgr: np.ndarray, pred_class: str, confidence: float) -> np.ndarray:
    out = bgr.copy()
    lines = [class_display(pred_class), f'conf {confidence:.1%}']
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, min(1.0, bgr.shape[1] / 900))
    thick = 2
    pad, lh = 12, int(32 * scale)
    max_w = max(cv2.getTextSize(t, font, scale, thick)[0][0] for t in lines)
    box_h = pad * 2 + lh * len(lines)
    overlay = out.copy()
    cv2.rectangle(overlay, (8, 8), (8 + max_w + pad * 2, 8 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
    y0 = 8 + pad + lh - 6
    for i, line in enumerate(lines):
        cv2.putText(out, line, (8 + pad, y0 + i * lh), font, scale, (80, 255, 80), thick, cv2.LINE_AA)
    return out


def build_clips(buf: list[Image.Image], hw: int) -> list[Image.Image]:
    idxs = frame_indices(len(buf) - 1, len(buf), hw)
    return [buf[i] for i in idxs]


def main():
    parser = argparse.ArgumentParser(description='Single-view video inference (tier1)')
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--ckpt', default=DEFAULT_CKPT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-tier1', action='store_true')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--print-every', type=int, default=30)
    parser.add_argument('--json', default=None)
    args = parser.parse_args()

    if args.output is None:
        base, _ = os.path.splitext(os.path.basename(args.video))
        args.output = f'{base}_pred.mp4'

    device = torch.device(
        'cpu' if args.device == 'cpu' or not torch.cuda.is_available() else args.device
    )
    use_tier1 = not args.no_tier1
    hw5, hw9 = 2, tier1_long_half_window()

    model, class_names, ckpt = load_model(args.ckpt, device)
    is_temporal = ckpt.get('model_type') == 'TemporalFiveModel'

    if not os.path.isfile(args.video):
        raise FileNotFoundError(
            f'Video not found: {args.video}\n\n'
            'This file is created by preprocessing — it is NOT included in the bundle.\n'
            'Step 1 (copy your MP4 into this folder, then run):\n'
            '  python mp4_q2_mirror_prepare.py --video "3MDAD (Day).mp4" --output 3MDAD_q2_mirror.mp4\n'
            'Step 2:\n'
            f'  python inference_prepared_video.py --video 3MDAD_q2_mirror.mp4 --device cuda --output 3MDAD_pred.mp4'
        )
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video (codec/path issue): {args.video}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = min(total, args.max_frames) if args.max_frames > 0 else total

    ext = os.path.splitext(args.output)[1].lower()
    fourcc = cv2.VideoWriter_fourcc(*('mp4v' if ext == '.mp4' else 'XVID'))
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot write: {args.output}')

    log(f'[INFO] device={device}  tier1={use_tier1}  temporal={is_temporal}  frames={target}')
    buf: list[Image.Image] = []
    log_rows = []
    idx = 0

    with torch.inference_mode():
        while idx < target:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            buf.append(pil)

            need = SEQ_LEN if is_temporal else 1
            if len(buf) >= need:
                if is_temporal:
                    c5 = build_clips(buf, hw5)
                    c9 = build_clips(buf, hw9) if use_tier1 else None
                    pred = predict_temporal_clip(
                        model, c5, device, class_names,
                        is_temporal=True, tier1=use_tier1, clip_frames_long=c9,
                    )
                else:
                    pred = predict_temporal_clip(
                        model, [pil], device, class_names,
                        is_temporal=False, tier1=use_tier1,
                    )
                disp = draw_banner(bgr, pred['pred_class'], pred['confidence'])
                log_rows.append({
                    'frame': idx, 'pred': pred['pred_class'],
                    'conf': pred['confidence'],
                })
                if args.print_every == 0 or idx % args.print_every == 0:
                    log(f'[Frame {idx:05d}] {pred["pred_class"]} {pred["confidence"]:.1%}')
            else:
                disp = bgr.copy()
                cv2.putText(disp, f'warmup {len(buf)}/{need}', (16, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

            writer.write(disp)
            idx += 1
            if idx % 200 == 0:
                log(f'[Progress] {idx}/{target}')
            gc.collect()

    cap.release()
    writer.release()
    log(f'[OK] {args.output} ({idx} frames)')

    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({'video': args.video, 'output': args.output, 'predictions': log_rows}, f, indent=2)


if __name__ == '__main__':
    main()
