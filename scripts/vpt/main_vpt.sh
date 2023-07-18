#!/bin/bash

# custom config
DATA= # your directory

DATASET=$1
CFG=$2  # config file
TRAINER=$3
CONV=$4   # VPT*: True  VPT: False
BACKBONE=$5 # backbone name
NTOK=$6
DOMAINS=$7
GPU=$8

TYPE=random
LOCATION=middle
DEEP=False      # VPT-shallow: False; VPT-deep: True

if [[ $CONV == "False" ]]; then
  if [[ $DEEP == "False" ]]; then
    DIR=output/vpt/${TRAINER}_Shallow/${DATASET}/${CFG}/${BACKBONE//\//}/deep${DEEP}_${LOCATION}/${DOMAINS}_ntok${NTOK}
  else
    DIR=output/vpt/${TRAINER}_Deep/${DATASET}/${CFG}/${BACKBONE//\//}/deep${DEEP}_${LOCATION}/${DOMAINS}_ntok${NTOK}
  fi
else
  DIR=output/vpt/${TRAINER}_Conv/${DATASET}/${CFG}/${BACKBONE//\//}/${TYPE}/${DOMAINS}_ntok${NTOK}
fi

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
  --config-file configs/trainers/VPT/${CFG}.yaml \
  --output-dir ${DIR} \
  TRAINER.VPT.NUM_TOKENS ${NTOK} \
  TRAINER.VPT.V_DEEP ${DEEP} \
  TRAINER.VPT.LOCATION ${LOCATION} \
  TRAINER.VPT.ENABLE_CONV ${CONV} \
  TRAINER.VPT.TYPE ${TYPE} 
  
fi
