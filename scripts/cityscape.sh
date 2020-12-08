#!/usr/bin/env bash
set -x
#DATAPATH="."
DATAPATH="/data/yyx/data/"
   
python -u main.py  --dataset cityscape \
    --maxdisp 192 \
    --datapath $DATAPATH --trainlist ./filenames/cityscape_train_list.txt --testlist ./filenames/cityscape_test_list.txt \
    --epochs 21 --lrepochs "10,12,14,16:2" \
    --model gwcnet-gc --logdir ./checkpoints/sceneflow/gwcnet-gc-color\
    --loadckpt ./checkpoints/sceneflow/gwcnet-gc-color/checkpoint_000019.ckpt