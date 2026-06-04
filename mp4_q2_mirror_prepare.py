#!/usr/bin/env python3
"""
MP4 → quadrant split → Q2 (top-left) horizontal mirror → export video.

No model inference in this script (run on local RTX 3050 later).

Quadrant layout (origin top-left):
    Q2 (top-left)     | Q1 (top-right)
    ------------------+------------------
    Q3 (bottom-left)  | Q4 (bottom-right)

Pipeline:
  1. Read each frame from MP4
  2. Crop quadrant 2 (top-left)
  3. Flip horizontally (mirror) so driver pose matches training camera
  4. Write preprocessed MP4 (+ optional frame folder / grid preview)

Examples:
  python mp4_q2_mirror_prepare.py
  python mp4_q2_mirror_prepare.py --video "3MDAD (Day).mp4" --output 3MDAD_q2_mirror.mp4
  python mp4_q2_mirror_prepare.py --save-frames ./frames_q2_mirror --max-frames 100

Later on local GPU (3050), feed the output to tier1 inference, e.g.:
  python inference_video.py --video 3MDAD_q2_mirror.mp4 --quadrant 1 --device cuda
  (preprocessed clip is already a single view; use full frame as input)

Or extend this file with --infer using predict_five_roi.py + model_best.pth.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import cv2
import numpy as np

# quadrant index -> (name, row, col); row0=top, col0=left
QUADRANT_LAYOUT = {
    1: ('top-right', 0, 1),
    2: ('top-left', 0, 0),
    3: ('bottom-left', 1, 0),
    4: ('bottom-right', 1, 1),
}

DEFAULT_VIDEO = '3MDAD (Day).mp4'
DEFAULT_OUTPUT = '3MDAD_q2_mirror.mp4'
TARGET_QUADRANT = 2


def log(msg: str) -> None:
    print(msg, flush=True)


def split_quadrants(frame_bgr: np.ndarray) -> dict[int, np.ndarray]:
    h, w = frame_bgr.shape[:2]
    mid_h, mid_w = h // 2, w // 2
    return {
        1: frame_bgr[0:mid_h, mid_w:w].copy(),
        2: frame_bgr[0:mid_h, 0:mid_w].copy(),
        3: frame_bgr[mid_h:h, 0:mid_w].copy(),
        4: frame_bgr[mid_h:h, mid_w:w].copy(),
    }


def extract_quadrant(frame_bgr: np.ndarray, quadrant: int) -> np.ndarray:
    if quadrant not in QUADRANT_LAYOUT:
        raise ValueError(f'quadrant must be 1-4, got {quadrant}')
    return split_quadrants(frame_bgr)[quadrant]


def preprocess_q2_mirror(frame_bgr: np.ndarray, quadrant: int = TARGET_QUADRANT) -> np.ndarray:
    """Crop quadrant and apply horizontal flip."""
    crop = extract_quadrant(frame_bgr, quadrant)
    return cv2.flip(crop, 1)


def draw_quadrant_grid(frame_bgr: np.ndarray, highlight: int = TARGET_QUADRANT) -> np.ndarray:
    """Debug preview: 2x2 grid with Q2 outlined."""
    quads = split_quadrants(frame_bgr)
    q2 = cv2.flip(quads[highlight], 1)
    quads[highlight] = q2
    top = np.hstack([quads[2], quads[1]])
    bottom = np.hstack([quads[3], quads[4]])
    grid = np.vstack([top, bottom])
    h, w = grid.shape[:2]
    # highlight box on top-left cell
    cv2.rectangle(grid, (2, 2), (w // 2 - 2, h // 2 - 2), (0, 255, 0), 2)
    cv2.putText(
        grid, f'Q{highlight} mirrored', (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
    )
    return grid


def open_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.mp4', '.m4v'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Failed to create video writer: {path}')
    return writer


def process_video(args: argparse.Namespace) -> dict:
    if not os.path.isfile(args.video):
        raise FileNotFoundError(f'Video not found: {args.video}')

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w, out_h = full_w // 2, full_h // 2
    target = min(total, args.max_frames) if args.max_frames > 0 else total

    log(f'[INFO] Input: {args.video}')
    log(f'[INFO] Full size: {full_w}x{full_h}  fps={fps:.2f}  frames≈{total}')
    log(f'[INFO] Q{args.quadrant} ({QUADRANT_LAYOUT[args.quadrant][0]}) + horizontal flip')
    log(f'[INFO] Output crop: {out_w}x{out_h}  will process {target} frames')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    writer = open_writer(args.output, fps, out_w, out_h)
    preview_writer = None
    if args.preview_grid:
        preview_writer = open_writer(args.preview_grid, fps, full_w, full_h)

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)

    frame_idx = 0
    while frame_idx < target:
        ok, frame = cap.read()
        if not ok:
            break

        processed = preprocess_q2_mirror(frame, quadrant=args.quadrant)
        writer.write(processed)

        if preview_writer is not None:
            preview_writer.write(draw_quadrant_grid(frame, highlight=args.quadrant))

        if args.save_frames:
            out_path = os.path.join(args.save_frames, f'frame_{frame_idx:06d}.jpg')
            cv2.imwrite(out_path, processed)

        frame_idx += 1
        if frame_idx % 200 == 0 or frame_idx == target:
            log(f'[Progress] {frame_idx}/{target} ({100.0 * frame_idx / max(target, 1):.1f}%)')

    cap.release()
    writer.release()
    if preview_writer is not None:
        preview_writer.release()

    meta = {
        'source_video': os.path.abspath(args.video),
        'output_video': os.path.abspath(args.output),
        'quadrant': args.quadrant,
        'quadrant_name': QUADRANT_LAYOUT[args.quadrant][0],
        'horizontal_flip': True,
        'full_resolution': [full_w, full_h],
        'crop_resolution': [out_w, out_h],
        'fps': fps,
        'frames_written': frame_idx,
        'preview_grid': os.path.abspath(args.preview_grid) if args.preview_grid else None,
        'frames_dir': os.path.abspath(args.save_frames) if args.save_frames else None,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'inference_note': (
            'Preprocessed Q2 mirror video is ready. On local GPU, run tier1 inference '
            'on the output file with inference_video.py or predict_five_roi.py + model_best.pth.'
        ),
    }

    if args.meta:
        os.makedirs(os.path.dirname(args.meta) or '.', exist_ok=True)
        with open(args.meta, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        log(f'[OK] Meta: {args.meta}')

    log(f'[OK] Saved: {args.output} ({frame_idx} frames)')
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MP4 quadrant Q2 crop + horizontal mirror (preprocess only, no inference)',
    )
    parser.add_argument('--video', default=DEFAULT_VIDEO, help='Input MP4 path')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Output preprocessed MP4')
    parser.add_argument(
        '--quadrant', type=int, default=TARGET_QUADRANT, choices=[1, 2, 3, 4],
        help='Quadrant to crop (default: 2 = top-left)',
    )
    parser.add_argument('--max-frames', type=int, default=0, help='0 = all frames')
    parser.add_argument('--save-frames', default=None, help='Optional directory to dump JPG frames')
    parser.add_argument(
        '--preview-grid', default=None,
        help='Optional path to save full-frame 2x2 grid preview video',
    )
    parser.add_argument(
        '--meta', default=None,
        help='Optional JSON metadata path (default: <output>.json)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.meta is None:
        base, _ = os.path.splitext(args.output)
        args.meta = f'{base}.json'
    process_video(args)


if __name__ == '__main__':
    main()
