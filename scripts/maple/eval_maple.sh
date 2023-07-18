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
DEEP=True
DEEPLAYER=None

DIR=output/maple/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/deep${DEEP}_${LOCATION}/${DOMAINS}_ntok${NTOK}

python train.py \
    --gpu ${GPU} \
    --backbone ${BACKBONE} \
    --domains ${DOMAINS} \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${DIR} \
    --eval-only \
    TRAINER.MAPLE.NUM_TOKENS ${NTOK} \
    TRAINER.MAPLE.N_CTX ${NTOK} \
    TRAINER.MAPLE.T_DEEP ${DEEP} \
    TRAINER.MAPLE.V_DEEP ${DEEP} \
    TRAINER.MAPLE.LOCATION ${LOCATION} \
    TRAINER.MAPLE.DEEP_LAYERS ${DEEPLAYER} \
    DATASET.NUM_SHOTS ${SHOTS} 

