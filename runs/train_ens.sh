#!/bin/bash

# parameters
DATA=$HOME/.datasets
NAMEDATASET='ferp'
PROJECT='../out/tripletruns'
PATHMODELSCONF='../modelsconf.json'
EPOCHS=2000
BATCHSIZETRAIN=120
BATCHSIZETEST=120
LEARNING_RATE=0.1
MOMENTUM=0.9
PRINT_FREQ=100
WORKERS=10
RESUME='chk000000.pth.tar'
GPU=0
ARCH='expert'
LOSS='cross'
OPT='sgd'
SCHEDULER='step'
SNAPSHOT=5
NUMCLASS=8
NUMCHANNELS=3
IMAGESIZE=64
TRAINSIZE=100000
VALSIZE=10000
EXP_NAME='expert_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_001'

rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
rm -rf $PROJECT/$EXP_NAME/
mkdir $PROJECT
mkdir $PROJECT/$EXP_NAME

## execute --parallel \
python ../train_ens.py \
$DATA \
--project=$PROJECT \
--name=$EXP_NAME \
--pathmodelsconf=$PATHMODELSCONF \
--epochs=$EPOCHS \
--batch-size-train=$BATCHSIZETRAIN \
--batch-size-test=$BATCHSIZETEST \
--learning-rate=$LEARNING_RATE \
--momentum=$MOMENTUM \
--print-freq=$PRINT_FREQ \
--workers=$WORKERS \
--resume=$RESUME \
--gpu=$GPU \
--loss=$LOSS \
--opt=$OPT \
--snapshot=$SNAPSHOT \
--scheduler=$SCHEDULER \
--arch=$ARCH \
--size-train=$TRAINSIZE \
--size-val=$VALSIZE \
--num-classes=$NUMCLASS \
--name-dataset=$NAMEDATASET \
--channels=$NUMCHANNELS \
--image-size=$IMAGESIZE \
--finetuning \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \
