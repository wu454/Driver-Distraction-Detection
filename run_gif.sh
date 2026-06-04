#!/usr/bin/env bash
cd "$(dirname "$0")"
python predict_five_roi.py --ckpt ./model_best.pth --gif "${1:-output.gif}" --save-gif "${2:-output_pred.gif}" "${@:3}"
