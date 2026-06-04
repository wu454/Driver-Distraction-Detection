#!/usr/bin/env bash
# Step1: Q2 crop + mirror  |  Step2: tier1 inference on 3050
set -euo pipefail
cd "$(dirname "$0")"
IN="${1:?usage: run_mp4_pipeline.sh input.mp4 [output_pred.mp4]}"
MID="${IN%.*}_q2_mirror.mp4"
OUT="${2:-${IN%.*}_pred.mp4}"
python mp4_q2_mirror_prepare.py --video "$IN" --output "$MID"
python inference_prepared_video.py --video "$MID" --output "$OUT" --device cuda --json "logs/$(basename "$OUT").json"
echo "[OK] $OUT"
