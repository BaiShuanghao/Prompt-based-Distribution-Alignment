#!/bin/bash

# custom config
DATA=  # your directory

DATASET=$1 # name of the dataset
CFG=$2  # config file
TRAINER=$3
BACKBONE=$4 # backbone name
DOMAINS=$5
GPU=$6

DIR=output/clip/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/${DOMAINS}

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
    --config-file configs/trainers/CLIP/${CFG}.yaml \
    --output-dir ${DIR} 
fi
