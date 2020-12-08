#!/usr/bin/env bash
set -x
#DATAPATH="/data/yyx/data/middlebury/data"
DATAPATH="/data/yyx/data/sceneflow"
#DATAPATH="/data/yyx/data/kitti/2015/data_scene_flow/"


# CUDA_VISIBLE_DEVICES=7 python -u main.py  --dataset kitti \
#     --datapath $DATAPATH --trainlist ./filenames/sceneflow_train_finalpass.txt --testlist ./filenames/origin/kitti15_val.txt \
#     --epochs 100 --lrepochs "10,12,14,16:2" \
#     --model gwcnet-gc --logdir ./checkpoints/sceneflow/gwcnet-gc-extra \
#     --loadckpt ./checkpoints/sceneflow/gwcnet-gc-color/checkpoint_000019.ckpt 
    
# python -u main.py  --dataset sceneflow \
#     --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test_50.txt \
#     --epochs 30 --lrepochs "10,12,14,16,20:2" \
#     --model gwcnet-gc --logdir ./checkpoints/sceneflow/gwcnet-gc-color \
#     --loadckpt ./checkpoints/sceneflow/gwcnet-gc-color/checkpoint_000020.ckpt

# python -u main.py  --dataset kitti \
#     --datapath $DATAPATH --trainlist ./filenames/origin/kitti15_train.txt --testlist ./filenames/origin/kitti15_200.txt \
#     --epochs 100 --lrepochs "10,12,14,16,20:2" \
#     --model gwcnet-gc --logdir ./checkpoints/sceneflow/gwcnet-gc-color \
#     --loadckpt ./checkpoints/sceneflow/gwcnet-gc-color/checkpoint_000000.ckpt

# python -u main.py  --dataset middlebury \
#     --maxdisp 256 \
#     --datapath $DATAPATH --trainlist ./filenames/middleburry_train.txt --testlist ./filenames/middleburry_train.txt \
#     --epochs 100 --lrepochs "10,12,14,16,20:2" \
#     --model gwcnet-gc --logdir ./checkpoints/sceneflow/gwcnet-gc-color \
#     --loadckpt ./checkpoints/sceneflow/gwcnet-gc-color/checkpoint_000019.ckpt

CUDA_VISIBLE_DEVICES=3 python -u main_mono_depth.py  --dataset sceneflow \
    --batch_size 1 \
    --train \
    --train_mono \
    --start_epoch 5 \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train_finalpass.txt --testlist ./filenames/sceneflow_test_50.txt \
    --epochs 50 --lrepochs "10,12,14,16,20:2" \
    --model gwcnet-gc --logdir ./checkpoints/mono_sceneflow/ \
    --loadckpt ./checkpoints/sceneflow/gwcnet-gc-color/checkpoint_000019.ckpt \
    --mono_checkpoint_path checkpoints/mono_sceneflow/mono_depth_checkpoint_000005.ckpt \
    --mono_encoder densenet161_bts \
    --mono_model_name bts_eigen_v2_pytorch_densenet161 \
    --test_batch_size 1 \
    --mono_max_depth 80 \
    --do_kb_crop \
    --maxdisp 192 \
    --bf 283.5
