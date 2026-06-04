"""Mirror / phone / console / reach heuristics and logit gating for fixed-camera deployment."""
import numpy as np
import torch

from roi_config import (
    C0_DISTRACTED_NEIGHBORS,
    CALL_POSE_THRESH,
    CONSOLE_SCORE_THRESH,
    DRINK_POSE_THRESH,
    GATE_C0_BOOST,
    GATE_C0_CONTEXT_BOOST,
    GATE_C0_SUPPRESS_DISTRACTED,
    GATE_C5_BOOST,
    GATE_C5_NO_CONSOLE,
    GATE_C5_SUPPRESS_BACKSEAT,
    GATE_C8_BOOST,
    GATE_C8_NO_MIRROR,
    GATE_C8_SUPPRESS_RADIO,
    GATE_C8_SUPPRESS_REACH,
    GATE_PHONE_NO_DEVICE,
    LEFT_PHONE_CLASSES,
    MAKEUP_CLASS,
    MIRROR_SCORE_THRESH,
    MIRROR_SCORE_THRESH_V3,
    PHONE_CLASSES,
    PHONE_SCORE_THRESH,
    RADIO_CLASS,
    REACH_BEHIND_CLASS,
    REACH_SCORE_THRESH,
    RIGHT_PHONE_CLASSES,
    SAFE_CLASS,
    TIER1_C8_MIRROR_SUPPRESS_THRESH,
)


def heuristic_mirror_score(pil_mirror):
    """Vanity mirror open: higher edge + contrast in upper-right crop."""
    gray = np.array(pil_mirror.convert('L').resize((96, 64)), dtype=np.float32) / 255.0
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge = float(gx + gy)
    contrast = float(gray.std())
    bright = float((gray > 0.55).mean())
    upper = gray[: gray.shape[0] // 2, :]
    upper_edge = float(np.abs(np.diff(upper, axis=1)).mean() + np.abs(np.diff(upper, axis=0)).mean())
    score = (
        0.35 * min(edge / 0.12, 1.0)
        + 0.30 * min(contrast / 0.18, 1.0)
        + 0.20 * bright
        + 0.15 * min(upper_edge / 0.14, 1.0)
    )
    return float(np.clip(score, 0.0, 1.0))


def heuristic_phone_score(pil_left, pil_right):
    """Proxy: saturated rectangular blobs in hand regions (device-like)."""
    scores = []
    for pil in (pil_left, pil_right):
        arr = np.array(pil.convert('RGB').resize((112, 112)), dtype=np.float32) / 255.0
        sat = arr.max(axis=2) - arr.min(axis=2)
        dark = (arr.mean(axis=2) < 0.35).mean()
        edge = np.abs(np.diff(arr.mean(axis=2), axis=1)).mean()
        s = 0.4 * min(float(sat.mean()) / 0.15, 1.0) + 0.35 * min(float(edge) / 0.08, 1.0) + 0.25 * float(dark)
        scores.append(s)
    return float(np.clip(max(scores), 0.0, 1.0))


def heuristic_console_score(pil_console, pil_wheel=None):
    """Center-stack / radio (c5): localized contrast + knobs/slider edges in console band."""
    gray = np.array(pil_console.convert('L').resize((112, 96)), dtype=np.float32) / 255.0
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge = float(gx + gy)
    contrast = float(gray.std())
    center_mass = gray[gray.shape[0] // 3: 2 * gray.shape[0] // 3, :]
    center_edge = float(np.abs(np.diff(center_mass, axis=1)).mean())
    score = (
        0.40 * min(edge / 0.14, 1.0)
        + 0.30 * min(contrast / 0.16, 1.0)
        + 0.30 * min(center_edge / 0.10, 1.0)
    )
    if pil_wheel is not None:
        wheel = np.array(pil_wheel.convert('L').resize((64, 64)), dtype=np.float32) / 255.0
        wheel_edge = float(np.abs(np.diff(wheel, axis=1)).mean())
        score = 0.85 * score + 0.15 * min(wheel_edge / 0.10, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def heuristic_ear_call_score(pil_hand) -> float:
    """Phone-at-ear: hand mass in upper half of hand ROI (c2/c4 cue)."""
    arr = np.array(pil_hand.convert('L').resize((112, 112)), dtype=np.float32) / 255.0
    mass = arr > 0.36
    if not mass.any():
        return 0.0
    ys = np.nonzero(mass)[0]
    cy = float(ys.mean()) / 112.0
    upper_mass = float(mass[:56, :].mean())
    edge = float(np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean())
    high_hand = max(0.0, 0.48 - cy) / 0.22
    score = (
        0.45 * min(high_hand, 1.0)
        + 0.35 * min(upper_mass / 0.12, 1.0)
        + 0.20 * min(edge / 0.11, 1.0)
    )
    return float(np.clip(score, 0.0, 1.0))


def heuristic_ear_call_scores(pil_left, pil_right) -> tuple[float, float]:
    return heuristic_ear_call_score(pil_left), heuristic_ear_call_score(pil_right)


def heuristic_drink_score(pil_hand) -> float:
    """Drinking (c6): hand/cup at mouth — center of hand ROI, not ear-side."""
    arr = np.array(pil_hand.convert('L').resize((112, 112)), dtype=np.float32) / 255.0
    mass = arr > 0.34
    if not mass.any():
        return 0.0
    ys, xs = np.nonzero(mass)
    cy = float(ys.mean()) / 112.0
    cx = float(xs.mean()) / 112.0
    center_mass = float(mass[40:72, 38:74].mean())
    mouth_zone = max(0.0, 1.0 - abs(cx - 0.50) / 0.16) * max(0.0, 1.0 - abs(cy - 0.42) / 0.14)
    low_hand = max(0.0, (cy - 0.38) / 0.22)
    score = 0.55 * mouth_zone + 0.30 * center_mass / 0.15 + 0.15 * min(low_hand, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def heuristic_drink_scores(pil_left, pil_right) -> tuple[float, float]:
    return heuristic_drink_score(pil_left), heuristic_drink_score(pil_right)


def heuristic_reach_score(pil_left, pil_right, pil_face):
    """
    Reach-behind (c7): hand away from wheel, elevated toward seat-back / off-center.
    """
    scores = []
    face_arr = np.array(pil_face.convert('L').resize((96, 96)), dtype=np.float32) / 255.0
    face_cy = float((face_arr > 0.35).nonzero()[0].mean()) / 96.0 if (face_arr > 0.35).any() else 0.35

    for pil in (pil_left, pil_right):
        arr = np.array(pil.convert('L').resize((112, 112)), dtype=np.float32) / 255.0
        mass = arr > 0.40
        if not mass.any():
            scores.append(0.0)
            continue
        ys, xs = np.nonzero(mass)
        cy = float(ys.mean()) / 112.0
        cx = float(xs.mean()) / 112.0
        edge = float(np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean())
        high_hand = max(0.0, 0.55 - cy)
        off_center = abs(cx - 0.5)
        away_from_face = max(0.0, cy - face_cy)
        s = (
            0.35 * min(high_hand / 0.20, 1.0)
            + 0.25 * min(off_center / 0.25, 1.0)
            + 0.20 * min(away_from_face / 0.15, 1.0)
            + 0.20 * min(edge / 0.12, 1.0)
        )
        scores.append(s)
    return float(np.clip(max(scores), 0.0, 1.0))


def class_name_to_indices(class_names):
    return {n: i for i, n in enumerate(class_names)}


def apply_behavior_gates(
    logits,
    class_names,
    mirror_score,
    phone_score,
    left_hand_score=None,
    right_hand_score=None,
    mirror_thresh=MIRROR_SCORE_THRESH,
    phone_thresh=PHONE_SCORE_THRESH,
    console_score=None,
    reach_score=None,
    left_call_score=None,
    right_call_score=None,
    left_drink_score=None,
    right_drink_score=None,
    gate_version='v2',
):
    """
    Logit gates (fixed camera):
    v2 — mirror suppresses c8; phone suppresses c1–c4; optional handedness nudge.
    v3 — adds console/reach cues for c0/c5/c8 disambiguation (no retrain required).
    hybrid (tier-1) — v2 phone/mirror base + v3 console/c0/c7; skips v3 c8 makeup path.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    out = logits.clone()
    idx = class_name_to_indices(class_names)
    batch = out.size(0)

    def _scalar_score(score, index, default=0.0):
        if score is None:
            return default
        if isinstance(score, torch.Tensor):
            s = score.reshape(-1)
            return float(s[index].item() if s.numel() > 1 else s.item())
        if isinstance(score, (list, tuple, np.ndarray)):
            return float(score[index])
        return float(score)

    for b in range(batch):
        row = out[b]
        ms = _scalar_score(mirror_score, b)
        ps = _scalar_score(phone_score, b)
        cs = _scalar_score(console_score, b, default=0.0)
        rs = _scalar_score(reach_score, b, default=0.0)

        c8_mirror_thresh = (
            TIER1_C8_MIRROR_SUPPRESS_THRESH
            if gate_version == 'hybrid'
            else mirror_thresh
        )
        if ms < c8_mirror_thresh and MAKEUP_CLASS in idx:
            row[idx[MAKEUP_CLASS]] -= GATE_C8_NO_MIRROR

        lc = _scalar_score(left_call_score, b, default=0.0)
        rc = _scalar_score(right_call_score, b, default=0.0)
        ld = _scalar_score(left_drink_score, b, default=0.0)
        rd = _scalar_score(right_drink_score, b, default=0.0)
        drinking = max(ld, rd) >= DRINK_POSE_THRESH
        ear_call_active = (
            (lc >= CALL_POSE_THRESH and lc >= rc + 0.04)
            or (rc >= CALL_POSE_THRESH and rc >= lc + 0.04)
        ) and not drinking

        if ps < phone_thresh and not ear_call_active:
            for name in PHONE_CLASSES:
                if name in idx:
                    row[idx[name]] -= GATE_PHONE_NO_DEVICE
            if SAFE_CLASS in idx:
                row[idx[SAFE_CLASS]] += GATE_C0_BOOST

        if ps >= phone_thresh and left_hand_score is not None and right_hand_score is not None:
            ls = float(left_hand_score[b] if hasattr(left_hand_score, '__len__') else left_hand_score)
            rs_hand = float(right_hand_score[b] if hasattr(right_hand_score, '__len__') else right_hand_score)
            if ls > rs_hand + 0.08:
                for name in RIGHT_PHONE_CLASSES:
                    if name in idx:
                        row[idx[name]] -= 1.0
            elif rs_hand > ls + 0.08:
                for name in LEFT_PHONE_CLASSES:
                    if name in idx:
                        row[idx[name]] -= 1.0

        if gate_version not in ('v3', 'hybrid'):
            continue

        mirror_v3 = MIRROR_SCORE_THRESH_V3
        makeup_active = ms >= mirror_v3
        console_active = cs >= CONSOLE_SCORE_THRESH

        calm = (
            ms < mirror_v3
            and ps < phone_thresh
            and cs < CONSOLE_SCORE_THRESH
            and rs < REACH_SCORE_THRESH
        )
        if calm and SAFE_CLASS in idx:
            row[idx[SAFE_CLASS]] += GATE_C0_CONTEXT_BOOST
            for name in C0_DISTRACTED_NEIGHBORS:
                if name in idx:
                    row[idx[name]] -= GATE_C0_SUPPRESS_DISTRACTED

        # c5 console path — skip when mirror/makeup is active (avoids c8→c5)
        if console_active and not makeup_active:
            if RADIO_CLASS in idx:
                row[idx[RADIO_CLASS]] += GATE_C5_BOOST
            if 'c9' in idx:
                row[idx['c9']] -= GATE_C5_SUPPRESS_BACKSEAT
        elif RADIO_CLASS in idx and not makeup_active:
            row[idx[RADIO_CLASS]] -= GATE_C5_NO_CONSOLE

        if gate_version == 'v3':
            if makeup_active:
                if MAKEUP_CLASS in idx:
                    row[idx[MAKEUP_CLASS]] += GATE_C8_BOOST
                if REACH_BEHIND_CLASS in idx:
                    row[idx[REACH_BEHIND_CLASS]] -= GATE_C8_SUPPRESS_REACH
                if RADIO_CLASS in idx and GATE_C8_SUPPRESS_RADIO > 0:
                    row[idx[RADIO_CLASS]] -= GATE_C8_SUPPRESS_RADIO
            elif (
                REACH_BEHIND_CLASS in idx
                and rs >= REACH_SCORE_THRESH
                and not console_active
            ):
                row[idx[REACH_BEHIND_CLASS]] += 0.8
        elif (
            gate_version == 'hybrid'
            and REACH_BEHIND_CLASS in idx
            and rs >= REACH_SCORE_THRESH
            and not console_active
            and not makeup_active
        ):
            row[idx[REACH_BEHIND_CLASS]] += 0.8

    if squeeze:
        return out.squeeze(0)
    return out


def combine_gate_scores(heuristic_score, learned_prob, weight_heuristic=0.4):
    """Blend PIL heuristic with trained aux head probability."""
    h = float(heuristic_score)
    p = float(learned_prob)
    return float(np.clip(weight_heuristic * h + (1.0 - weight_heuristic) * p, 0.0, 1.0))


def apply_call_pose_boost(
    logits: torch.Tensor,
    class_names: list[str],
    left_call: float,
    right_call: float,
    left_drink: float = 0.0,
    right_drink: float = 0.0,
) -> torch.Tensor:
    """Boost c2/c4 on ear-call; skip entirely when drinking (c6 protection). Never penalizes c6."""
    from roi_config import (
        CALL_POSE_THRESH,
        DRINK_POSE_THRESH,
        GATE_CALL_POSE_BOOST,
        GATE_CALL_SUPPRESS_C0,
        GATE_CALL_SUPPRESS_C8,
        GATE_CALL_SUPPRESS_SIDE_TEXT,
    )

    if max(left_drink, right_drink) >= DRINK_POSE_THRESH:
        return logits

    squeeze = logits.dim() == 1
    out = logits.unsqueeze(0).clone() if squeeze else logits.clone()
    idx = class_name_to_indices(class_names)
    row = out[0]

    if right_call >= CALL_POSE_THRESH and right_call >= left_call + 0.04:
        if 'c2' in idx:
            row[idx['c2']] += GATE_CALL_POSE_BOOST
        if 'c1' in idx:
            row[idx['c1']] -= GATE_CALL_SUPPRESS_SIDE_TEXT
        if MAKEUP_CLASS in idx:
            row[idx[MAKEUP_CLASS]] -= GATE_CALL_SUPPRESS_C8
        if SAFE_CLASS in idx:
            row[idx[SAFE_CLASS]] -= GATE_CALL_SUPPRESS_C0

    if left_call >= CALL_POSE_THRESH and left_call >= right_call + 0.04:
        if 'c4' in idx:
            row[idx['c4']] += GATE_CALL_POSE_BOOST
        if 'c3' in idx:
            row[idx['c3']] -= GATE_CALL_SUPPRESS_SIDE_TEXT
        if MAKEUP_CLASS in idx:
            row[idx[MAKEUP_CLASS]] -= GATE_CALL_SUPPRESS_C8
        if SAFE_CLASS in idx:
            row[idx[SAFE_CLASS]] -= GATE_CALL_SUPPRESS_C0

    return out.squeeze(0) if squeeze else out
