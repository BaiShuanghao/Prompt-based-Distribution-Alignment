#!/bin/bash

# custom config
DATA= # ********** your directory ***********

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

# bash scripts/pda/eval_pda.sh officehome b32_ep10_officehome PDA ViT-B/16 2 a-c 0
DIR=output/PDA/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/tdeep${TDEEP}_vdeep${VDEEP}_${LOCATION}/${DOMAINS}_ntok${NTOK}

python train.py \
  --gpu ${GPU} \
  --backbone ${BACKBONE} \
  --domains ${DOMAINS} \
  --root ${DATA} \
  --trainer ${TRAINER} \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  --config-file configs/trainers/PDA/${CFG}.yaml \
  --output-dir ${DIR} \
  --model-dir ${DIR} \
  --eval-only \
  TRAINER.PDA.TP ${TP}\
  TRAINER.PDA.T_DEEP ${TDEEP} \
  TRAINER.PDA.N_CTX ${NTOK} \
  TRAINER.PDA.VP ${VP} \
  TRAINER.PDA.V_DEEP ${VDEEP}\
  TRAINER.PDA.NUM_TOKENS ${NTOK} \
  TRAINER.PDA.LOCATION ${LOCATION} \
  TRAINER.PDA.DEEP_LAYERS ${DEEPLAYER} \
  TRAINER.PDA.DEEP_SHARED ${SHARE} 
  