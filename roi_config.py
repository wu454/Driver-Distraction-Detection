"""Fixed-camera ROI fractions (x1, y1, x2, y2) and class groupings for gating."""

# Legacy triple-ROI (backward compatible)
PERSON_ROI = (0.06, 0.02, 0.56, 0.70)
HAND_ROI = (0.38, 0.42, 0.78, 0.88)
WHEEL_ROI = (0.45, 0.28, 0.92, 0.92)

# Five-ROI layout: fixed mount, vanity mirror upper-right
FACE_ROI = PERSON_ROI
LEFT_HAND_ROI = (0.08, 0.38, 0.44, 0.90)
RIGHT_HAND_ROI = (0.48, 0.38, 0.84, 0.90)
MIRROR_ROI = (0.68, 0.02, 0.98, 0.32)
CONSOLE_ROI = (0.30, 0.48, 0.72, 0.92)

# v3: tighter crops for radio (c5) and vanity mirror (c8) — use with --opt-v3
CONSOLE_ROI_V3 = (0.28, 0.46, 0.74, 0.90)
MIRROR_ROI_V3 = (0.66, 0.00, 0.99, 0.34)

ROI_SIZES = {
    'face': 256,
    'left_hand': 224,
    'right_hand': 224,
    'wheel': 288,
    'mirror': 160,
    'console': 192,
}

SAFE_CLASS = 'c0'
RADIO_CLASS = 'c5'
REACH_BEHIND_CLASS = 'c7'
MAKEUP_CLASS = 'c8'
PHONE_CLASSES = frozenset({'c1', 'c2', 'c3', 'c4'})
LEFT_PHONE_CLASSES = frozenset({'c3', 'c4'})
RIGHT_PHONE_CLASSES = frozenset({'c1', 'c2'})
C0_DISTRACTED_NEIGHBORS = frozenset({'c3', 'c5', 'c7'})

# Horizontal-flip label swap (driver camera, left/right in image)
FLIP_SWAP = {
    'c1': 'c3',
    'c3': 'c1',
    'c2': 'c4',
    'c4': 'c2',
}

# Inference gate strengths (logit subtract) — v1 was too aggressive on val
GATE_C8_NO_MIRROR = 2.5
GATE_PHONE_NO_DEVICE = 2.0
GATE_C0_BOOST = 1.0
MIRROR_SCORE_THRESH = 0.42
PHONE_SCORE_THRESH = 0.45

# CE class weights (boost hard / deployment-critical classes)
CLASS_LOSS_WEIGHTS = {
    'c0': 1.25,
    'c3': 1.15,
    'c4': 1.15,
    'c8': 1.10,
}

# ---------------------------------------------------------------------------
# v3 optimization (--opt-v3): c0 temporal + c5/c8 ROI & gates
# Enable at train time:  python train.py --train five_roi --five-temporal --opt-v3
# Enable at deploy:      python predict_five_roi.py --opt-v3
# ---------------------------------------------------------------------------
OPT_V3 = False

TEMPORAL_HALF_WINDOW_DEFAULT = 2
CLASS_TEMPORAL_HALF_WINDOW = {
    'c0': 4,   # 9 frames — stable safe-driving context
    'c5': 3,   # 7 frames — radio knob motion
    'c8': 3,   # 7 frames — mirror grooming motion
}

# hybrid=c0 long sliding (9f interior) + class windows for c5/c8; others 5f sliding
CLIP_SAMPLING_V3 = 'hybrid'
CLIP_SUBJECT_CENTER_MIN_FRAMES = 5
CLIP_SKIP_EDGE_PADDING = True

CLASS_SAMPLER_BOOST = {
    'c0': 1.50,
    'c5': 1.40,
    'c8': 1.40,
}

CLASS_LOSS_WEIGHTS_V3 = {
    'c0': 1.45,
    'c3': 1.05,
    'c4': 1.10,
    'c5': 1.35,
    'c8': 1.35,
}

# v3 inference gates (deploy without retrain)
CONSOLE_SCORE_THRESH = 0.40
REACH_SCORE_THRESH = 0.38
MIRROR_SCORE_THRESH_V3 = 0.38
GATE_C0_CONTEXT_BOOST = 1.5
GATE_C0_SUPPRESS_DISTRACTED = 1.2
GATE_C5_NO_CONSOLE = 2.0
GATE_C5_BOOST = 1.2
GATE_C8_BOOST = 1.5
GATE_C8_SUPPRESS_RADIO = 0.0
GATE_C8_SUPPRESS_REACH = 0.6
GATE_C5_SUPPRESS_BACKSEAT = 1.0

# Tier-1 deploy: hybrid gates + c0 long temporal context (no retrain)
TIER1_GATE_VERSION = 'hybrid'
TIER1_C0_HALF_WINDOW = 4          # 9 frames for safe-driving neighborhood
TIER1_C0_BLEND = 0.5              # weight on 9f logits when blending with 5f
TIER1_C0_TRIGGER_CLASSES = frozenset({'c0', 'c3', 'c5', 'c7'})
TIER1_C8_MIRROR_SUPPRESS_THRESH = 0.35  # softer c8 suppress than v2 (0.42)

# When tier1 gates flip phone/c9 away from model raw top-1, trust raw logits instead.
TIER1_RESCUE_ENABLED = True
TIER1_RESCUE_RAW_CLASSES = frozenset({'c1', 'c2', 'c3', 'c4', 'c9'})
TIER1_RESCUE_IF_GATED_IN = frozenset({'c0', 'c6', 'c8'})

# Ear-call pose (c2/c4)
CALL_POSE_THRESH = 0.30
GATE_CALL_POSE_BOOST = 1.6
GATE_CALL_SUPPRESS_C8 = 1.8
GATE_CALL_SUPPRESS_SIDE_TEXT = 0.6
GATE_CALL_SUPPRESS_C0 = 0.9

# c6 保护：喝水姿态检测，阻止贴耳 boost 误伤
DRINK_POSE_THRESH = 0.34


def active_mirror_roi(opt_v3=False):
    return MIRROR_ROI_V3 if opt_v3 else MIRROR_ROI


def active_console_roi(opt_v3=False):
    return CONSOLE_ROI_V3 if opt_v3 else CONSOLE_ROI


def temporal_half_window_for_class(class_name, default=None, opt_v3=False):
    """Per-class temporal half-window; opt_v3 enables CLASS_TEMPORAL_HALF_WINDOW."""
    if default is None:
        default = TEMPORAL_HALF_WINDOW_DEFAULT
    if opt_v3:
        return CLASS_TEMPORAL_HALF_WINDOW.get(class_name, TEMPORAL_HALF_WINDOW_DEFAULT)
    return default


def class_loss_weights(opt_v3=False):
    return CLASS_LOSS_WEIGHTS_V3 if opt_v3 else CLASS_LOSS_WEIGHTS


def clip_sampling_mode(opt_v3=False):
    return CLIP_SAMPLING_V3 if opt_v3 else 'sliding'
