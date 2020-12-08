#!/usr/bin/env bash
set -x
#DATAPATH="."
DATAPATH="/data/yyx/data/ETH3D/TRAIN/"
   
python -u main_eth3d.py  --dataset eth3d \
    --maxdisp 384 \
    --datapath $DATAPATH --trainlist ./filenames/eth3d_train.txt --testlist ./filenames/eth3d_train.txt \
    --epochs 21 --lrepochs "10,12,14,16:2" \
    --model gwcnet-gc --logdir ./checkpoints/sceneflow/gwcnet-gc-color\
    --loadckpt ./checkpoints/sceneflow/gwcnet-gc-color/checkpoint_000006.ckpt
    