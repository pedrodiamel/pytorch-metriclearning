#!/bin/bash


PATHDATASET='~/.datasets/'
PROJECT='../out/tripletruns'
PROJECTNAME='triplet_cvggemb13_hinge_adam_ferp_emb64_imsize64_000'
PATHNAMEOUT='../out'
FILENAME='result.txt'
PATHMODEL='models'
NAMEMODEL='model_best.pth.tar' #'model_best.pth.tar' #'chk000565.pth.tar'
MODEL=$PROJECT/$PROJECTNAME/$PATHMODEL/$NAMEMODEL  

python ../eval_fer.py \
--project=$PROJECT \
--projectname=$PROJECTNAME \
--pathdataset=$PATHDATASET \
--pathnameout=$PATHNAMEOUT \
--filename=$FILENAME \
--model=$MODEL \


