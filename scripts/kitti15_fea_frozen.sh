#!/usr/bin/env bash
set -x
DATAPATH="/data/yyx/data/kitti/2015/data_scene_flow"
python -u main_fea_frozen.py --dataset kitti \
    --datapath $DATAPATH \
    --trainlist ./filenames/origin/kitti15_train.txt --testlist ./filenames/origin/kitti15_val.txt \
    --epochs 300 --lrepochs "200:10" \
    --model gwcnet-gc --logdir ./checkpoints/kitti/opt_frozen_ft --loadckpt ./checkpoints/kitti/opt_frozen_ft/checkpoint_000020.ckpt \
    --test_batch_size 1
