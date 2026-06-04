#!/usr/bin/env python3
"""Deploy five-ROI model on images or GIF (default: output.gif)."""
import argparse
import json
import os
from collections import Counter

import torch
from PIL import Image, ImageDraw, ImageFont

from augmentations import AlbuTransform
from dataset import load_five_roi_pils, load_heuristic_roi_pils
from inference_gates import apply_behavior_gates, combine_gate_scores
from models import FiveROIModel, TemporalFiveModel
from roi_config import ROI_SIZES, TEMPORAL_HALF_WINDOW_DEFAULT
from tier1_inference import (
    apply_tier1_gates,
    frame_indices,
    heuristic_scores_from_pil,
    learned_aux_scores,
    tier1_long_half_window,
    tier1_rescue_logits,
    finalize_tier1_logits,
    tier1_temporal_logits,
)

DEFAULT_GIF = './output.gif'
DEFAULT_CKPT = './model_best.pth'
FALLBACK_CKPT = './checkpoints/five_roi_v2/model_best.pth'
TEMPORAL_HALF_WINDOW = 2


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    class_names = ckpt.get('class_names')
    if class_names is None:
        raise ValueError('Checkpoint missing class_names')
    model_type = ckpt.get('model_type', 'FiveROIModel')
    if model_type == 'TemporalFiveModel':
        backend = ckpt.get('temporal_backend', 'transformer')
        model = TemporalFiveModel(len(class_names), pretrained=False, temporal=backend)
    else:
        model = FiveROIModel(len(class_names), pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.class_names = class_names
    model.use_inference_gates = False
    model.eval()
    model.to(device)
    return model, class_names, ckpt


def preprocess_roi(pil_img, size):
    return AlbuTransform(size, train=False)(pil_img)


def stack_roi_clip(clip_frames, device, opt_v3=False):
    """clip_frames: list of RGB PIL → batched [1,T,C,H,W] tensors."""
    face_t, left_t, right_t, wheel_t, mirror_t = [], [], [], [], []
    for pil in clip_frames:
        fp, lp, rp, wp, mp = load_five_roi_pils(pil, opt_v3=opt_v3)
        face_t.append(preprocess_roi(fp, ROI_SIZES['face']))
        left_t.append(preprocess_roi(lp, ROI_SIZES['left_hand']))
        right_t.append(preprocess_roi(rp, ROI_SIZES['right_hand']))
        wheel_t.append(preprocess_roi(wp, ROI_SIZES['wheel']))
        mirror_t.append(preprocess_roi(mp, ROI_SIZES['mirror']))
    return (
        torch.stack(face_t).unsqueeze(0).to(device),
        torch.stack(left_t).unsqueeze(0).to(device),
        torch.stack(right_t).unsqueeze(0).to(device),
        torch.stack(wheel_t).unsqueeze(0).to(device),
        torch.stack(mirror_t).unsqueeze(0).to(device),
    )


TEMPORAL_HALF_WINDOW = TEMPORAL_HALF_WINDOW_DEFAULT


def resolve_gate_mode(opt_v3: bool, tier1: bool) -> tuple[str, bool]:
    """Return (gate_version, use_v3_heuristics)."""
    if tier1:
        return 'hybrid', True
    if opt_v3:
        return 'v3', True
    return 'v2', False


def predict_temporal_clip(
    model, clip_frames, device, class_names, is_temporal=False,
    opt_v3=False, tier1=False, clip_frames_long=None,
):
    """5-frame clip → prediction; tier1 may blend 9-frame logits for c0 neighborhood."""
    gate_ver, use_v3_roi = resolve_gate_mode(opt_v3, tier1)
    if is_temporal and len(clip_frames) > 1:
        face, left, right, wheel, mirror = stack_roi_clip(clip_frames, device, opt_v3=use_v3_roi)
        center_pil = clip_frames[len(clip_frames) // 2]
        h_scores = heuristic_scores_from_pil(center_pil, opt_v3_heuristics=use_v3_roi)

        roi_long = None
        if tier1 and clip_frames_long is not None and len(clip_frames_long) > 1:
            roi_long = stack_roi_clip(clip_frames_long, device, opt_v3=use_v3_roi)

        with torch.no_grad():
            if tier1:
                logits, aux, _used_long = tier1_temporal_logits(
                    model,
                    (face, left, right, wheel, mirror),
                    class_names,
                    device,
                    roi_long=roi_long,
                    enable_c0_long=clip_frames_long is not None,
                )
            else:
                out = model(face, left, right, wheel, mirror, apply_gates=False, class_names=class_names)
                logits = out[0] if isinstance(out, tuple) else out
                aux = out[1] if isinstance(out, tuple) else {}

        ms_learned, ps_learned = learned_aux_scores(aux)
    else:
        return predict_pil_frame(
            model, clip_frames[len(clip_frames) // 2], device, class_names,
            opt_v3=opt_v3, tier1=tier1,
        )

    if tier1:
        gated = apply_tier1_gates(logits, class_names, h_scores, ms_learned, ps_learned, gate_version=gate_ver)
        gated = finalize_tier1_logits(logits, gated, class_names, h_scores)
    else:
        ms = combine_gate_scores(h_scores['mirror'], ms_learned)
        ps = combine_gate_scores(h_scores['phone'], ps_learned)
        gated = apply_behavior_gates(
            logits.cpu(), class_names, ms, ps,
            console_score=h_scores.get('console'), reach_score=h_scores.get('reach'),
            gate_version=gate_ver,
        )
    probs = torch.softmax(gated, dim=1)[0]
    conf, pred_idx = probs.max(dim=0)
    top3 = probs.topk(min(3, len(class_names)))
    return {
        'pred_class': class_names[pred_idx.item()],
        'confidence': float(conf.item()),
        'mirror_score': float(combine_gate_scores(h_scores['mirror'], ms_learned)),
        'phone_score': float(combine_gate_scores(h_scores['phone'], ps_learned)),
        'top3': [{'class': class_names[i], 'prob': float(probs[i])} for i in top3.indices.tolist()],
        'probs': {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }


def predict_pil_frame(model, pil_img, device, class_names, is_temporal=False, opt_v3=False, tier1=False):
    """Single RGB PIL frame → prediction dict."""
    gate_ver, use_v3_roi = resolve_gate_mode(opt_v3, tier1)
    h_scores = heuristic_scores_from_pil(pil_img, opt_v3_heuristics=use_v3_roi)
    rois = load_heuristic_roi_pils(pil_img, opt_v3=use_v3_roi)

    face = preprocess_roi(rois['face'], ROI_SIZES['face']).unsqueeze(0).to(device)
    left = preprocess_roi(rois['left'], ROI_SIZES['left_hand']).unsqueeze(0).to(device)
    right = preprocess_roi(rois['right'], ROI_SIZES['right_hand']).unsqueeze(0).to(device)
    wheel = preprocess_roi(rois['wheel'], ROI_SIZES['wheel']).unsqueeze(0).to(device)
    mirror = preprocess_roi(rois['mirror'], ROI_SIZES['mirror']).unsqueeze(0).to(device)

    if isinstance(model, TemporalFiveModel):
        face, left, right, wheel, mirror = [t.unsqueeze(1) for t in (face, left, right, wheel, mirror)]

    with torch.no_grad():
        out = model(face, left, right, wheel, mirror, apply_gates=False, class_names=class_names)
        logits = out[0] if isinstance(out, tuple) else out
        aux = out[1] if isinstance(out, tuple) else {}

    ms_learned, ps_learned = learned_aux_scores(aux)
    if tier1:
        gated = apply_tier1_gates(logits, class_names, h_scores, ms_learned, ps_learned, gate_version=gate_ver)
        gated = finalize_tier1_logits(logits, gated, class_names, h_scores)
        ms = combine_gate_scores(h_scores['mirror'], ms_learned)
        ps = combine_gate_scores(h_scores['phone'], ps_learned)
    else:
        ms = combine_gate_scores(h_scores['mirror'], ms_learned)
        ps = combine_gate_scores(h_scores['phone'], ps_learned)
        gated = apply_behavior_gates(
            logits.cpu(), class_names, ms, ps,
            console_score=h_scores.get('console'), reach_score=h_scores.get('reach'),
            gate_version=gate_ver,
        )
    probs = torch.softmax(gated, dim=1)[0]
    conf, pred_idx = probs.max(dim=0)

    top3 = probs.topk(min(3, len(class_names)))
    return {
        'pred_class': class_names[pred_idx.item()],
        'confidence': float(conf.item()),
        'mirror_score': float(ms),
        'phone_score': float(ps),
        'top3': [
            {'class': class_names[i], 'prob': float(probs[i])}
            for i in top3.indices.tolist()
        ],
        'probs': {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }


def iter_gif_frames(gif_path):
    """Yield (frame_index, RGB PIL.Image) from a GIF."""
    with Image.open(gif_path) as im:
        n = getattr(im, 'n_frames', 1)
        for i in range(n):
            im.seek(i)
            frame = im.convert('RGB')
            yield i, frame


def majority_vote(frames_results, class_names):
    """Clip-level label from per-frame argmax votes."""
    votes = Counter(r['pred_class'] for r in frames_results)
    winner, count = votes.most_common(1)[0]
    avg_conf = sum(
        r['probs'].get(winner, 0.0) for r in frames_results
    ) / max(len(frames_results), 1)
    return {
        'pred_class': winner,
        'vote_count': count,
        'total_frames': len(frames_results),
        'avg_prob': avg_conf,
        'vote_breakdown': dict(votes),
    }


def draw_frame_label(pil_img, pred_class, confidence, mirror_s, phone_s):
    """Overlay prediction on frame for saved GIF."""
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    text = f'{pred_class}  {confidence:.0%}  M:{mirror_s:.2f} P:{phone_s:.2f}'
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    x, y = 8, 8
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text((x, y), text, fill=(0, 255, 0), font=font)
    return img


def save_annotated_gif(frames_rgb, frame_results, out_path, duration_ms=200):
    labeled = [
        draw_frame_label(
            fr, r['pred_class'], r['confidence'], r['mirror_score'], r['phone_score'],
        )
        for fr, r in zip(frames_rgb, frame_results)
    ]
    labeled[0].save(
        out_path,
        save_all=True,
        append_images=labeled[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def predict_gif(model, gif_path, device, class_names, save_annotated=None, duration_ms=200):
    frames_rgb = []
    frame_results = []
    for idx, frame in iter_gif_frames(gif_path):
        r = predict_pil_frame(model, frame, device, class_names)
        r['frame'] = idx
        frame_results.append(r)
        frames_rgb.append(frame)

    summary = majority_vote(frame_results, class_names)
    summary['source'] = gif_path
    summary['frames'] = frame_results

    if save_annotated:
        save_annotated_gif(frames_rgb, frame_results, save_annotated, duration_ms=duration_ms)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Five-ROI inference (default: output.gif)',
    )
    parser.add_argument('--ckpt', default=DEFAULT_CKPT)
    parser.add_argument('--gif', default=None, help=f'GIF path (default: {DEFAULT_GIF} if no --image/--dir)')
    parser.add_argument('--image', default=None, help='Single image path')
    parser.add_argument('--dir', default=None, help='Folder of images')
    parser.add_argument('--save-gif', default=None, metavar='PATH',
                        help='Write annotated GIF (e.g. output_pred.gif)')
    parser.add_argument('--gif-duration', type=int, default=200, help='ms per frame in saved GIF')
    parser.add_argument('--output', default=None, help='JSON output path')
    parser.add_argument('--temporal', action='store_true', help='TemporalFiveModel checkpoint')
    parser.add_argument('--tier1', action='store_true',
                        help='Tier-1 deploy: hybrid gates + c0 9f temporal blend (default for temporal ckpt)')
    parser.add_argument('--no-tier1', action='store_true', help='Disable tier-1; use v2 gates only')
    parser.add_argument('--opt-v3', action='store_true', help='Legacy: v3 gates (use --tier1 instead)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = args.ckpt
    if not os.path.isfile(ckpt_path):
        alt = './experiments/exp_five_roi_v2_temporal/model_best.pth'
        if not os.path.isfile(alt):
            alt = './experiments/exp_five_roi_v2/model_best.pth'
        if not os.path.isfile(alt):
            alt = './experiments/exp_five_roi_gated/model_best.pth'
        if not os.path.isfile(alt):
            alt = FALLBACK_CKPT
        if os.path.isfile(alt):
            ckpt_path = alt
        else:
            print(f'[ERROR] Checkpoint not found: {args.ckpt}')
            print('  Train: python train.py --train five_roi')
            return

    model, class_names, ckpt = load_model(ckpt_path, device)
    is_temporal = ckpt.get('model_type') == 'TemporalFiveModel' or args.temporal
    if args.no_tier1:
        use_tier1 = False
    elif args.opt_v3 and not args.tier1:
        use_tier1 = False
    else:
        use_tier1 = args.tier1 or is_temporal
    gate_label = 'tier1-hybrid' if use_tier1 else ('v3' if args.opt_v3 else 'v2')
    print(f'[OK] {ckpt_path}')
    print(f'     classes={len(class_names)}  val_acc={ckpt.get("best_val_acc", "?")}'
          f'  temporal={is_temporal}  gates={gate_label}')

    results = {}

    gif_path = args.gif
    if gif_path is None and args.image is None and args.dir is None:
        gif_path = DEFAULT_GIF if os.path.isfile(DEFAULT_GIF) else None

    if gif_path:
        if not os.path.isfile(gif_path):
            print(f'[ERROR] GIF not found: {gif_path}')
            return
        save_path = args.save_gif
        if save_path is None:
            base = os.path.splitext(os.path.basename(gif_path))[0]
            save_path = f'{base}_pred.gif'
        frames_list = list(iter_gif_frames(gif_path))
        print(
            f'\n[GIF] {gif_path}  frames={len(frames_list)}  '
            f'mode={"5f-temporal" if is_temporal else "single"}  gates={gate_label}'
        )

        frames_rgb = [f for _, f in frames_list]
        frame_results = []
        hw5 = TEMPORAL_HALF_WINDOW
        hw9 = tier1_long_half_window()
        for idx, frame in frames_list:
            if is_temporal:
                clip5 = [
                    frames_rgb[i]
                    for i in frame_indices(idx, len(frames_rgb), hw5)
                ]
                clip9 = None
                if use_tier1:
                    clip9 = [
                        frames_rgb[i]
                        for i in frame_indices(idx, len(frames_rgb), hw9)
                    ]
                r = predict_temporal_clip(
                    model, clip5, device, class_names, is_temporal=True,
                    opt_v3=args.opt_v3, tier1=use_tier1, clip_frames_long=clip9,
                )
            else:
                r = predict_pil_frame(
                    model, frame, device, class_names,
                    opt_v3=args.opt_v3, tier1=use_tier1,
                )
            r['frame'] = idx
            frame_results.append(r)
            print(
                f"  frame {idx}: {r['pred_class']} ({r['confidence']:.1%})  "
                f"mirror={r['mirror_score']:.2f} phone={r['phone_score']:.2f}"
            )

        summary = majority_vote(frame_results, class_names)
        summary['source'] = gif_path
        summary['frames'] = frame_results
        results['gif'] = summary

        print(
            f"\n[CLIP VOTE] {summary['pred_class']} "
            f"({summary['vote_count']}/{summary['total_frames']} frames, "
            f"avg_prob={summary['avg_prob']:.2%})"
        )
        print(f"  breakdown: {summary['vote_breakdown']}")

        if save_path:
            save_annotated_gif(frames_rgb, frame_results, save_path, duration_ms=args.gif_duration)
            print(f'[OK] Annotated GIF: {save_path}')

    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.dir:
        for name in sorted(os.listdir(args.dir)):
            if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(args.dir, name))

    img_results = []
    for p in image_paths:
        img = Image.open(p).convert('RGB')
        r = predict_pil_frame(
            model, img, device, class_names,
            opt_v3=args.opt_v3, tier1=use_tier1,
        )
        r['image'] = p
        img_results.append(r)
        print(
            f"{os.path.basename(p)} → {r['pred_class']} ({r['confidence']:.2%})  "
            f"mirror={r['mirror_score']:.2f} phone={r['phone_score']:.2f}"
        )
    if img_results:
        results['images'] = img_results

    if not results:
        parser.error('Provide --gif, --image, or --dir (default: output.gif)')

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'[OK] JSON: {args.output}')


if __name__ == '__main__':
    main()
