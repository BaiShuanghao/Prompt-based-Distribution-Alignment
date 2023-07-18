#!/bin/bash

# custom config
DATA= # your directory

DATASET=$1 # name of the dataset
CFG=$2  # config file
TRAINER=$3
BACKBONE=$4 # backbone name
T=$5 # temperature
TAU=$6 # pseudo label threshold
U=$7 # coefficient for loss_u
DOMAINS=$8 
GPU=$9

DIR=output/dapl/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/${DOMAINS}_${T}_${TAU}_${U}

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
      --config-file configs/trainers/DAPL/${CFG}.yaml \
      --output-dir ${DIR} \
      TRAINER.DAPL.T ${T} \
      TRAINER.DAPL.TAU ${TAU} \
      TRAINER.DAPL.U ${U} 
      
fi
