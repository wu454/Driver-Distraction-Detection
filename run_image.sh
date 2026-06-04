#!/usr/bin/env bash
cd "$(dirname "$0")"
python predict_five_roi.py --ckpt ./model_best.pth --image "${1:?usage: run_image.sh path.jpg}" "${@:2}"
