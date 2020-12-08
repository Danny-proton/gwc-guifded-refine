#!/usr/bin/env bash
set -x

DATAPATH="/data/yyx/data/kitti/2015/data_scene_flow"
#python save_disp.py --datapath $DATAPATH --testlist ./filenames/kitti15_test.txt --model gwcnet-g --loadckpt ./checkpoints/kitti15/gwcnet-g/best.ckpt
CUDA_VISIBLE_DEVICES=3 python save_disp.py --datapath $DATAPATH --testlist ./filenames/kitti15_test.txt --model gwcnet-g --loadckpt ./checkpoints/sceneflow/gwcnet-gc-color/checkpoint_000016.ckpt