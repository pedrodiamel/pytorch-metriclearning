#!/bin/bash

# parameters
DATA=$HOME/.datasets
NAMEDATASET='cifar10'
PROJECT='../out/tripletruns'
EPOCHS=60
BATCHSIZE=60
LEARNING_RATE=0.0001
MOMENTUM=0.5
PRINT_FREQ=100
WORKERS=60
RESUME='chk000000xxx.pth.tar'
GPU=0
ARCH='embresnet18'
LOSS='hinge'
OPT='adam'
SCHEDULER='fixed'
SNAPSHOT=5
NUMCHANNELS=3
MARGIN=1
TRAINSIZE=100000
VALSIZE=10000
EMB=32
EXP_NAME='exp_triplet_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_emb'$EMB'_001'


rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
rm -rf $PROJECT/$EXP_NAME/
mkdir $PROJECT    
mkdir $PROJECT/$EXP_NAME  


## execute
python ../classification_tripletloss_train.py \
$DATA \
--project=$PROJECT \
--name=$EXP_NAME \
--epochs=$EPOCHS \
--batch-size=$BATCHSIZE \
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
--emb-dim=$EMB \
--triplet-size-train=$TRAINSIZE \
--triplet-size-val=$VALSIZE \
--name-dataset=$NAMEDATASET \
--channels=$NUMCHANNELS \
--finetuning \
--parallel \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \
