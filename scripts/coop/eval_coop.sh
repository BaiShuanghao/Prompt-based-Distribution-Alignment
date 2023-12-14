#!/bin/bash

# custom config
DATA= # your directory

DATASET=$1
CFG=$2  # config file
TRAINER=$3
BACKBONE=$4 # backbone name
NCTX=$5  # number of context tokens
DOMAINS=$6
GPU=$7

CTP=end     # class token position (end or middle)
CSC=False   # class-specific context (False or True)
INIT=None

DIR=output/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/csc${CSC}_ctp${CTP}/${DOMAINS}_nctx${NCTX}

python train.py \
    --gpu ${GPU} \
    --backbone ${BACKBONE} \
    --domains ${DOMAINS} \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${DIR} \
    --eval-only \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    TRAINER.COOP.CTX_INIT ${INIT}

