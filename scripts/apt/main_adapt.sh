#!/bin/bash

# custom config
DATA= # your directory

DATASET=$1
CFG=$2  # config file
TRAINER=$3
BACKBONE=$4 # backbone name
NTOK=$5
DOMAINS=$6
GPU=$7

LOCATION=middle
DEEPLAYER=None
TP=True

# CoOp
# TDEEP=False
# VP=False
# VDEEP=False
# SHARE=False

# MaPLe
TDEEP=True
VP=True
VDEEP=True
SHARE=True

DIR=output/APT/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/tdeep${TDEEP}_vdeep${VDEEP}_${LOCATION}/${DOMAINS}_ntok${NTOK}

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
    --config-file configs/trainers/APT/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.APT.TP ${TP}\
    TRAINER.APT.T_DEEP ${TDEEP} \
    TRAINER.APT.N_CTX ${NTOK} \
    TRAINER.APT.VP ${VP} \
    TRAINER.APT.V_DEEP ${VDEEP}\
    TRAINER.APT.NUM_TOKENS ${NTOK} \
    TRAINER.APT.LOCATION ${LOCATION} \
    TRAINER.APT.DEEP_LAYERS ${DEEPLAYER} \
    TRAINER.APT.DEEP_SHARED ${SHARE} 
    
fi