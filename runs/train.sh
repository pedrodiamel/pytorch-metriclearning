#!/bin/bash

# parameters
DATA=$HOME/.datasets
NAMEDATASET='ferp'
PROJECT='../out/tripletruns'
EPOCHS=60
BATCHSIZETRAIN=60
BATCHSIZETEST=60
LEARNING_RATE=0.0001
MOMENTUM=0.5
WEIGHT_DECAY=0.0005
PRINT_FREQ=100
WORKERS=60
RESUME='chk000000xxx.pth.tar'
GPU=0
ARCH='resnetemb18'
LOSS='hinge'
OPT='adam'
SCHEDULER='step'
SNAPSHOT=5
NUMCHANNELS=3
NUMCLASS=8
MARGIN=1
TRAINSIZE=100000
VALSIZE=10000
DIM=32
IMAGESIZE=224
EXP_NAME='triplet_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_emb'$DIM'_001'


rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
rm -rf $PROJECT/$EXP_NAME/
mkdir $PROJECT    
mkdir $PROJECT/$EXP_NAME  


## execute
python ../train.py \
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

