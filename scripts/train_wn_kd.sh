#!/usr/bin/env bash

set -x
set -e

TASK="WN18RR"
model_path="./checkpoint/newkd/12_6_3/model_last.mdl"
# model_path="./checkpoint/9*0d05_bestmodel/wn18rr/model_last.mdl"
# model_path="./checkpoint/9*0d05_bestmodel/wn18rr/model_last.mdl"
OUTPUT_DIR="./checkpoint/newkd/6_3_5/"
DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

CUDA_VISIBLE_DEVICES=4 python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 5e-5 \
--use-link-graph \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--task ${TASK} \
--batch-size 512 \
--print-freq 100 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--pre-batch 0 \
--finetune-t \
--epochs 50 \
--workers 4 \
--eval-model-path "${model_path}" \
--teacher-bert-num-hidden-layers 6 \
--student-bert-num-hidden-layers 3 \
--max-to-keep 6 "$@"
