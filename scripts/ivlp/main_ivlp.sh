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

DIR=output/ivlp/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/deep${DEEP}_${LOCATION}/${DOMAINS}_ntok${NTOK}_1
  
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
    --config-file configs/trainers/IVLP/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.IVLP.NUM_TOKENS ${NTOK} \
    TRAINER.IVLP.N_CTX ${NTOK} \
    TRAINER.IVLP.T_DEEP ${DEEP} \
    TRAINER.IVLP.V_DEEP ${DEEP} \
    TRAINER.IVLP.LOCATION ${LOCATION} \
    TRAINER.IVLP.DEEP_LAYERS ${DEEPLAYER} \
    DATASET.NUM_SHOTS ${SHOTS} 

fi
