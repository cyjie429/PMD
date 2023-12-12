#!/usr/bin/env bash

set -x
set -e

TASK="wiki5m_ind"
model_path="./checkpoint/kd/wikidata5m/9_6/model_last.mdl"
OUTPUT_DIR="./checkpoint/kd/wikidata5m/6_3_1/"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TASK=$1
    shift
fi

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

CUDA_VISIBLE_DEVICES=4,5 python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 3e-5 \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--task "${TASK}" \
--batch-size 512 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--finetune-t \
--pre-batch 0 \
--epochs 1 \
--workers 3 \
--eval-model-path "${model_path}" \
--teacher-bert-num-hidden-layers 6 \
--student-bert-num-hidden-layers 3 \
--max-to-keep 10 "$@"
