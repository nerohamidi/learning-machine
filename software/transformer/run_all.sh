#!/bin/bash
# Full transformer pipeline (conda env ml). Logs in ../result/vit/.
set -e
PY=~/miniconda3/envs/ml/bin/python
cd "$(dirname "$0")"
$PY -W ignore train_vit.py --stage teacher  --epochs ${TEACHER_EPOCHS:-120}
$PY -W ignore train_vit.py --stage transfer --epochs 5
$PY -W ignore train_vit.py --stage finetune --epochs 30
$PY -W ignore train_vit.py --stage qat      --epochs 30
$PY -W ignore train_vit.py --stage eval
