#!/usr/bin/env bash
set -x
DATAPATH="/data/yyx/data/kitti/2015/data_scene_flow"
CUDA_VISIBLE_DEVICES=3 python -u main.py --dataset kitti \
    --datapath $DATAPATH \
    --trainlist ./filenames/origin/kitti15_train.txt \
    --testlist ./filenames/origin/kitti15_val.txt \
    --epochs 1000 --lrepochs "400,600,800:10" \
    --model gwcnet-gc \
    --mono_encoder densenet161_bts \
    --logdir ./checkpoints/kitti/ \
    --mono_model_name bts_eigen_v2_pytorch_densenet161 \
    --loadckpt ./checkpoints/kitti/checkpoint_000060.ckpt \
    --test_batch_size 1 \
    --mono_max_depth 80 \
    --make_occ_mask \
    --do_kb_crop
    #--loadckpt ./checkpoints/kitti/only_update_bn_ft_lr4/checkpoint_000032.ckpt \
    
