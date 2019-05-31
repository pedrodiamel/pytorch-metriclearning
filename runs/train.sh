#!/bin/bash

# parameters
DATA=$HOME/.datasets
NAMEDATASET='ferp'
PROJECT='../out/tripletruns'
EPOCHS=150
BATCHSIZETRAIN=60
BATCHSIZETEST=60
LEARNING_RATE=0.0001
MOMENTUM=0.5
WEIGHT_DECAY=0.0005
PRINT_FREQ=100
WORKERS=10
RESUME='chk000000xxx.pth.tar'
GPU=0
ARCH='preactresembnet18'
LOSS='hinge'
OPT='adam'
SCHEDULER='fixed'
SNAPSHOT=10
NUMCHANNELS=3
NUMCLASS=8
MARGIN=1
TRAINSIZE=100000
VALSIZE=5000
DIM=64
IMAGESIZE=32
EXP_NAME='triplet_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_emb'$DIM'_imsize'$IMAGESIZE'_000'


rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
rm -rf $PROJECT/$EXP_NAME/
mkdir $PROJECT    
mkdir $PROJECT/$EXP_NAME  


## execute
CUDA_VISIBLE_DEVICES=2,3  python ../train.py \
$DATA \
--project=$PROJECT \
--name=$EXP_NAME \
--epochs=$EPOCHS \
--batch-size-train=$BATCHSIZETRAIN \
--batch-size-test=$BATCHSIZETEST \
--image-size=$IMAGESIZE \
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
--margin=$MARGIN \
--dim=$DIM \
--num-classes=$NUMCLASS \
--triplet-size-train=$TRAINSIZE \
--triplet-size-val=$VALSIZE \
--name-dataset=$NAMEDATASET \
--channels=$NUMCHANNELS \
--finetuning \
--parallel \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \

