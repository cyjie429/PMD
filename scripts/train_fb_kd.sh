#!/usr/bin/env bash

set -x
set -e

TASK="FB15k237"
# model_path="./checkpoint/cosine/fb15k237/model_last.mdl"
# model_path="./checkpoint/psd/12_9z0d2/fb15k237/model_best.mdl"
model_path="./checkpoint/psd/9z0d2_6z0d1/fb15k237/model_best.mdl"
OUTPUT_DIR="./checkpoint/psd/6z0d1_3z0d0/fb15k237/"
DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

CUDA_VISIBLE_DEVICES=2,3 python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 3e-5 \
--use-link-graph \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "$DATA_DIR/valid.txt.json" \
--task ${TASK} \
--batch-size 512 \
--print-freq 100 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--finetune-t \
--pre-batch 0 \
--epochs 6 \
--workers 4 \
--eval-model-path "${model_path}" \
--max-to-keep 5 "$@"
