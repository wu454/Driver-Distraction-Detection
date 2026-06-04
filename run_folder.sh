#!/usr/bin/env bash
cd "$(dirname "$0")"
python predict_five_roi.py --ckpt ./model_best.pth --dir "${1:?usage: run_folder.sh ./images/}" "${@:2}"
