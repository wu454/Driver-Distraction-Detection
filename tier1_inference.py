"""Tier-1 deploy helpers: hybrid gates + adaptive c0 9-frame temporal blend (no retrain)."""
from __future__ import annotations

import torch
from PIL import Image

from dataset import load_heuristic_roi_pils
from inference_gates import (
    apply_behavior_gates,
    apply_call_pose_boost,
    combine_gate_scores,
    heuristic_console_score,
    heuristic_drink_scores,
    heuristic_ear_call_scores,
    heuristic_mirror_score,
    heuristic_phone_score,
    heuristic_reach_score,
)
from roi_config import (
    C0_DISTRACTED_NEIGHBORS,
    LEFT_PHONE_CLASSES,
    PHONE_CLASSES,
    RIGHT_PHONE_CLASSES,
    TIER1_C0_BLEND,
    TIER1_C0_HALF_WINDOW,
    TIER1_C0_TRIGGER_CLASSES,
    TIER1_GATE_VERSION,
    TIER1_RESCUE_ENABLED,
    TIER1_RESCUE_IF_GATED_IN,
    TIER1_RESCUE_RAW_CLASSES,
    SAFE_CLASS,
)


def frame_indices(center: int, length: int, half_window: int) -> list[int]:
    """Temporal window indices with edge clamp (matches dataset behavior)."""
    return [
        max(0, min(length - 1, center + offset))
        for offset in range(-half_window, half_window + 1)
    ]


def c0_long_context_trigger(
    logits: torch.Tensor,
    class_names: list[str],
    logit_margin: float = 0.5,
) -> bool:
    """
    Re-run with 9 frames only when safe-driving is ambiguous:
    - top-1 is c3/c5/c7, or
    - top-1 is c0 but a distracted neighbor is within logit_margin.
    """
    if logits.dim() == 2:
        logits = logits[0]
    k = min(2, len(class_names))
    top = logits.topk(k)
    top1_name = class_names[int(top.indices[0])]
    neighbors = set(C0_DISTRACTED_NEIGHBORS)

    if top1_name in neighbors:
        return True
    if top1_name != SAFE_CLASS:
        return False
    if k < 2:
        return False
    top2_name = class_names[int(top.indices[1])]
    if top2_name not in neighbors:
        return False
    return float(top.values[0] - top.values[1]) < logit_margin


def blend_logits(logits_short: torch.Tensor, logits_long: torch.Tensor, long_weight: float = TIER1_C0_BLEND):
    """Convex blend of two logit vectors (same shape)."""
    w = float(long_weight)
    if logits_short.dim() == 1:
        return (1.0 - w) * logits_short + w * logits_long
    return (1.0 - w) * logits_short + w * logits_long


def heuristic_scores_from_pil(pil_image: Image.Image, opt_v3_heuristics: bool = True) -> dict[str, float]:
    """PIL RGB → mirror/phone/console/reach heuristic scores."""
    rois = load_heuristic_roi_pils(pil_image, opt_v3=opt_v3_heuristics)
    left_call, right_call = heuristic_ear_call_scores(rois['left'], rois['right'])
    left_drink, right_drink = heuristic_drink_scores(rois['left'], rois['right'])
    return {
        'mirror': heuristic_mirror_score(rois['mirror']),
        'phone': heuristic_phone_score(rois['left'], rois['right']),
        'console': heuristic_console_score(rois['console'], rois['wheel']),
        'reach': heuristic_reach_score(rois['left'], rois['right'], rois['face']),
        'left_call': left_call,
        'right_call': right_call,
        'left_drink': left_drink,
        'right_drink': right_drink,
    }


def learned_aux_scores(aux: dict, device=None) -> tuple[float, float]:
    """Extract mirror/phone learned aux probabilities from model aux dict."""
    ms = ps = 0.0
    if aux:
        if 'mirror_logit' in aux:
            t = aux['mirror_logit']
            if device is not None:
                t = t.to(device)
            ms = float(torch.sigmoid(t).reshape(-1)[0].item())
        if 'phone_logit' in aux:
            t = aux['phone_logit']
            if device is not None:
                t = t.to(device)
            ps = float(torch.sigmoid(t).reshape(-1)[0].item())
    return ms, ps


def apply_tier1_gates(
    logits: torch.Tensor,
    class_names: list[str],
    heuristic_scores: dict[str, float],
    learned_mirror: float = 0.0,
    learned_phone: float = 0.0,
    gate_version: str = TIER1_GATE_VERSION,
) -> torch.Tensor:
    """Hybrid tier-1 gates: v2 phone/mirror + v3 console/c0/c7 (no v3 c8 makeup path)."""
    ms = combine_gate_scores(heuristic_scores.get('mirror', 0.0), learned_mirror)
    ps = combine_gate_scores(heuristic_scores.get('phone', 0.0), learned_phone)
    return apply_behavior_gates(
        logits.cpu() if logits.is_cuda else logits,
        class_names,
        ms,
        ps,
        console_score=heuristic_scores.get('console', 0.0),
        reach_score=heuristic_scores.get('reach', 0.0),
        left_call_score=heuristic_scores.get('left_call', 0.0),
        right_call_score=heuristic_scores.get('right_call', 0.0),
        left_drink_score=heuristic_scores.get('left_drink', 0.0),
        right_drink_score=heuristic_scores.get('right_drink', 0.0),
        gate_version=gate_version,
    )


def tier1_rescue_logits(
    raw_logits: torch.Tensor,
    gated_logits: torch.Tensor,
    class_names: list[str],
) -> torch.Tensor:
    """
    If tier1 gates override raw top-1 for phone/c9 into c0/c6/c8, keep raw model output.
    Fixes slideshow cases: c2→c8, c3/c4 penalized to c0, c9→c0.
    """
    if not TIER1_RESCUE_ENABLED:
        return gated_logits
    raw = raw_logits[0] if raw_logits.dim() == 2 else raw_logits
    gated = gated_logits[0] if gated_logits.dim() == 2 else gated_logits
    raw_name = class_names[int(raw.argmax())]
    gated_name = class_names[int(gated.argmax())]
    if raw_name == gated_name:
        return gated_logits
    if raw_name not in TIER1_RESCUE_RAW_CLASSES:
        return gated_logits
    if raw_name in PHONE_CLASSES and gated_name in TIER1_RESCUE_IF_GATED_IN:
        return raw_logits
    if raw_name == 'c6' and gated_name in RIGHT_PHONE_CLASSES:
        return raw_logits
    if raw_name == 'c9' and gated_name == 'c0':
        return raw_logits
    for side_phones in (LEFT_PHONE_CLASSES, RIGHT_PHONE_CLASSES):
        if raw_name in side_phones and gated_name in side_phones and raw_name != gated_name:
            return raw_logits
    return gated_logits


def tier1_post_call_rescue(
    raw_logits: torch.Tensor,
    final_logits: torch.Tensor,
    class_names: list[str],
) -> torch.Tensor:
    """Undo call_pose on drinking: raw c6 misclassified as c2/c1."""
    if not TIER1_RESCUE_ENABLED:
        return final_logits
    raw = raw_logits[0] if raw_logits.dim() == 2 else raw_logits
    final = final_logits[0] if final_logits.dim() == 2 else final_logits
    raw_name = class_names[int(raw.argmax())]
    final_name = class_names[int(final.argmax())]
    if raw_name == 'c6' and final_name in RIGHT_PHONE_CLASSES:
        return raw_logits
    return final_logits


def finalize_tier1_logits(
    raw_logits: torch.Tensor,
    gated_logits: torch.Tensor,
    class_names: list[str],
    heuristic_scores: dict[str, float],
) -> torch.Tensor:
    """Rescue → call_pose (skip if drinking) → c6 post-rescue."""
    out = tier1_rescue_logits(raw_logits, gated_logits, class_names)
    out = apply_call_pose_boost(
        out,
        class_names,
        heuristic_scores.get('left_call', 0.0),
        heuristic_scores.get('right_call', 0.0),
        heuristic_scores.get('left_drink', 0.0),
        heuristic_scores.get('right_drink', 0.0),
    )
    return tier1_post_call_rescue(raw_logits, out, class_names)


@torch.no_grad()
def forward_temporal_logits(model, roi_tensors, class_names, device) -> tuple[torch.Tensor, dict]:
    """Run TemporalFiveModel on batched [B,T,C,H,W] ROI tensors → logits + aux."""
    face, left, right, wheel, mirror = [t.to(device) for t in roi_tensors]
    out = model(face, left, right, wheel, mirror, apply_gates=False, class_names=class_names)
    logits = out[0] if isinstance(out, tuple) else out
    aux = out[1] if isinstance(out, tuple) else {}
    return logits, aux


def tier1_temporal_logits(
    model,
    roi_short,
    class_names,
    device,
    roi_long=None,
    enable_c0_long: bool = True,
    long_weight: float = TIER1_C0_BLEND,
) -> tuple[torch.Tensor, dict, bool]:
    """
    5-frame forward; optionally blend with 9-frame logits when c0 neighborhood is triggered.

    roi_short: (face, left, right, wheel, mirror) each [1, T5, C, H, W]
    roi_long: same shape with T9, required when trigger fires.
    Returns (logits, aux_from_short_pass, used_long_context).
    """
    logits_5, aux = forward_temporal_logits(model, roi_short, class_names, device)
    used_long = False
    if not enable_c0_long or roi_long is None:
        return logits_5, aux, used_long
    if not c0_long_context_trigger(logits_5, class_names):
        return logits_5, aux, used_long
    logits_9, _aux9 = forward_temporal_logits(model, roi_long, class_names, device)
    blended = blend_logits(logits_5, logits_9, long_weight=long_weight)
    return blended, aux, True


def load_dataset_clip_tensors(dataset, idx: int, half_window: int | None = None):
    """
    Build ROI tensors for a clip at arbitrary half_window (reuses dataset transforms).

    Returns (face, left, right, wheel, mirror), center_path, label_idx.
    """
    clip = dataset.clips[idx]
    paths, center, label = clip[0], clip[1], clip[2]
    hw = half_window if half_window is not None else (
        clip[3] if len(clip) >= 4 else dataset.half_window
    )
    frame_ids = frame_indices(center, len(paths), hw)

    face_pils, left_pils, right_pils, wheel_pils, mirror_pils = [], [], [], [], []
    for fi in frame_ids:
        f, l, r, w, m = dataset._load_five_pils(paths[fi])
        face_pils.append(f)
        left_pils.append(l)
        right_pils.append(r)
        wheel_pils.append(w)
        mirror_pils.append(m)

    def stack_clip(transform, pils):
        if hasattr(transform, 'apply_clip'):
            return torch.stack(transform.apply_clip(pils))
        return torch.stack([transform(p) for p in pils])

    face = stack_clip(dataset.transform_face, face_pils).unsqueeze(0)
    left = stack_clip(dataset.transform_left, left_pils).unsqueeze(0)
    right = stack_clip(dataset.transform_right, right_pils).unsqueeze(0)
    wheel = stack_clip(dataset.transform_wheel, wheel_pils).unsqueeze(0)
    mirror = stack_clip(dataset.transform_mirror, mirror_pils).unsqueeze(0)
    return (face, left, right, wheel, mirror), paths[center], label


def tier1_long_half_window() -> int:
    return TIER1_C0_HALF_WINDOW
