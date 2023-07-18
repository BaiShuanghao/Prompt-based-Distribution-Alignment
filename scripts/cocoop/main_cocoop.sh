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

DIR=output/cocoop/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/csc${CSC}_ctp${CTP}/${DOMAINS}_nctx${NCTX}

if [ -d "$DIR" ]; then
  echo "Results are available in ${DIR}, so skip this job"
else
  echo "Run this job and save the output to ${DIR}"
  
  python train.py \
    --gpu ${GPU} \
    --backbone ${BACKBONE} \
    --domains ${DOMAINS} \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.COCOOP.N_CTX ${NCTX} \
    TRAINER.COCOOP.CSC ${CSC} \
    TRAINER.COCOOP.CLASS_TOKEN_POSITION ${CTP} \
    TRAINER.COCOOP.CTX_INIT ${INIT} 

fi
