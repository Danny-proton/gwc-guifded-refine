from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss ,silog_loss
from utils import *
from torch.utils.data import DataLoader
from math import isnan
import torchvision
import gc
import cv2 as cv
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname

import matplotlib.pyplot as plt
from pylab import *
import matplotlib
from models.bts import BtsModel
matplotlib.use('pdf')


cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--model', default='gwcnet-gc', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--bf', type=float, default=720*0.54, help='baseline*focal length')
parser.add_argument('--start_from_zero_epoch', action='store_true', help='start_from_zero_epoch')

parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--adam_eps', type=float, help='epsilon in mono Adam optimizer', default=1e-6)
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
#parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--train', action='store_true', help='train the model')
parser.add_argument('--resume', default=False, help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

parser.add_argument('--mono_model_name', type=str, help='model name', default='bts_nyu_v2')
parser.add_argument('--mono_encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet161_bts')
parser.add_argument('--mono_weight_decay',type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--mono_input_height', type=int, help='input height', default=480)
parser.add_argument('--mono_input_width', type=int, help='input width', default=640)
parser.add_argument('--mono_max_depth', type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--mono_checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--r2l', help='if set, predict disp from right img to left img', action='store_true')
parser.add_argument('--make_occ_mask', help='if set, make occ mask', action='store_true')
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--occlude', type=float,   help='occ rate in result', default=0.73)
parser.add_argument('--mask', type=float,   help='occ rate in result', default=0.2)

parser.add_argument('--variance_focus',type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# test_gt_path='/data/yyx/GwcNet-master/checkpoints/kitti/ft_from0/kitti_test_gt/'
# test_pred1_path='/data/yyx/GwcNet-master/checkpoints/kitti/ft_from0/kitti_test_pred/'
# os.makedirs(test_gt_path exist_ok=True)
# os.makedirs(test_pred1_path, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)
# logger = SummaryWriter()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args)
mono_model = BtsModel(params=args)
model = nn.DataParallel(model)
mono_model = nn.DataParallel(mono_model)
model.cuda()
mono_model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# mono_optimizer = optim.Adam([{'params': mono_model.module.encoder.parameters(), 'mono_weight_decay': args.mono_weight_decay},
#                                 {'params': mono_model.module.decoder.parameters(), 'mono_weight_decay': 0}],
#                                 lr=args.lr, eps=args.adam_eps)
mono_optimizer = optim.Adam(mono_model.parameters(), lr=args.lr, betas=(0.9, 0.999))

silog_criterion = silog_loss(variance_focus=args.variance_focus)

#先跑测试，再写loss函数，存储节点方式



# for index,(name,value) in enumerate(model.named_parameters()):
#     #value.requires_grad = (index < last)
#     #value.requires_grad = False
#     print(index+1, name," : ",value.requires_grad)

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    start_epoch = state_dict['epoch'] + 1
if args.mono_checkpoint_path:
    print("loading mono model {}".format(args.mono_checkpoint_path))
    mono_checkpoint = torch.load(args.mono_checkpoint_path)
    mono_model.load_state_dict(mono_checkpoint['model'])
print("start at epoch {}".format(start_epoch))

if args.start_from_zero_epoch:
    start_epoch = 0
print("start_epoch",start_epoch)
def train():
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)
        print("current rate is ",args.lr)
        print(args.maxdisp)
        #avg_train_scalars = AverageMeter()
        all_loss=0.0
        all_mono_loss=0.0
        
        #print(all_loss)
        # training
        if args.train:
            nannum=0
            mono_nannum=0
            for batch_idx, sample in enumerate(TrainImgLoader):
                #print(len(sample))
                #print("!!!!!!!!!!!!!")
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                start_time = time.time()
                do_summary = global_step % args.summary_freq == 0
                loss,mono_loss, scalar_outputs, image_outputs = train_sample(sample,compute_metrics=do_summary)
                #loss_all=loss_all+loss
                if math.isnan(loss):
                    print(batch_idx,'loss is nan')
                    nannum=nannum+1
                else:
                    all_loss=loss+all_loss
                if math.isnan(mono_loss):
                    print(batch_idx,'mono loss is nan')
                    mono_nannum=mono_nannum+1
                else:
                    all_mono_loss=mono_loss+all_mono_loss
                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    save_images(logger, 'train', image_outputs, global_step)

                #avg_train_scalars.update(loss)
                #print("!!!",avg_train_scalars)
                del scalar_outputs, image_outputs
                if (batch_idx%100==0)or(batch_idx==len(TrainImgLoader)-1):
                    print('Epoch {}/{}, Iter {}/{},loss = {:.3f},mono_loss = {:.3f}, time = {:.3f},loss avg={:.3f},mono loss avg={:.3f}'.format(epoch_idx, args.epochs,
                                                                                        batch_idx,
                                                                                        len(TrainImgLoader), loss,mono_loss,
                                                                                        time.time() - start_time,all_loss/(batch_idx-nannum+1),all_mono_loss/(batch_idx-mono_nannum+1)))

                    
            #loss_all=loss_all.avg()
            #avg_train_scalars=avg_train_scalars.mean()
            #print("avg_train_loss",all_loss/(len(TestImgLoader)-nannum))
                    #print("losses",loss_all/(batch_idx+1))
            #print("loss averange=",loss_all/12537.0)
            #saving checkpoints

            if (epoch_idx + 1) % args.save_freq == 0:
                # checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                # torch.save(checkpoint_data, "{}/checkpoint_with_mono_{:0>6}.ckpt".format(args.logdir, epoch_idx))
                checkpoint_data = {'epoch': epoch_idx, 'model': mono_model.state_dict(), 'optimizer': mono_optimizer.state_dict()}
                torch.save(checkpoint_data, "{}/mono_depth_checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
            gc.collect()
            print("avg_train_loss",all_loss/(len(TrainImgLoader)-nannum))

        #testing
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            # print("test model")
            #print("sample",sample)
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, mono_loss, scalar_outputs, image_outputs = test_sample(batch_idx,sample, compute_metrics=do_summary)

            if do_summary:
                # print("is it here? in do summary?",len(image_outputs))
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            if (batch_idx%100==0)or(batch_idx==len(TestImgLoader)-1):
                print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f},mono_loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(TestImgLoader), loss, mono_loss ,
                                                                                     time.time() - start_time))
            #print(avg_test_scalars["EPE"])
        avg_test_scalars = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        gc.collect()

def lr_consistency_map(disp_left ,disp_right):
    lr_cons=[]
    # for i in range(len(disp_left)):
    #     lr_cons.append(torch.abs(disp_left[i] - disp_right[i]))
    lr_con_unabs=disp_left - disp_right
    lr_con=torch.abs(lr_con_unabs)
    lr_cons.append(lr_con)
    return lr_cons

# train one sample
def train_sample(sample, compute_metrics=False):
    # model.eval()
    model.train()
    mono_model.train()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    optimizer.zero_grad()
    mono_optimizer.zero_grad()

    disp_ests = model(imgL, imgR)
    disp_ests_final= disp_ests[-1]

    #mono
    mono_all_depth_ests =mono_model(imgL, imgR)
    depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, mono_final_depth =mono_all_depth_ests
    mono_final_depth=torch.squeeze(mono_final_depth,dim=0)
    mono_disp_est =args.bf / mono_final_depth
    if args.make_occ_mask:
        if args.r2l==False:
            args.r2l=True
            disp_ests_r2l = model(imgR, imgL)
            args.r2l=False
        else:
            args.r2l=False
            disp_ests_r2l = model(imgR, imgL)
            args.r2l=True
        disp_ests_final_r2l=disp_ests_r2l[-1]
        lr_consistency=lr_consistency_map(disp_ests_final,disp_ests_final_r2l)
        mask_occ=(lr_consistency[-1][0, :, :]>args.occlude)
        mask_occ_re=(lr_consistency[-1][0, :, :]<=args.occlude)
        disp_ests_unocc=disp_ests_final[mask_occ_re]
        mono_disp_est_occ=mono_disp_est[mask_occ]
        disp_est_occ_refine=disp_ests_unocc+mono_disp_est_occ
        disp_ests[-1]=disp_est_occ_refine

    #print(len(disp_ests))
    #print(disp_ests[0].shape)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    #print("##########",mask.shape)
    loss = model_loss(disp_ests, disp_gt, mask)
    #loss = model_loss(disp_ests, disp_gt, mask,imgL, imgR)
    depth_gt=args.bf /disp_gt
    if args.dataset == "sceneflow":
        mono_mask = depth_gt > 0.1
    else :
        mono_mask = depth_gt > 1.0
    mono_loss = silog_criterion.forward(mono_final_depth, depth_gt, mono_mask)
    # print("mono loss",mono_loss)
    scalar_outputs = {"loss": loss}
    scalar_outputs = {"mono loss": mono_loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            # print(len(disp_ests))
            image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
            image_outputs["mono errormap"] = [disp_error_image_func()(mono_disp_est, disp_gt)]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["mono EPE"] = [EPE_metric(mono_disp_est, disp_gt, mask)]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["mono D1"] = [D1_metric(mono_disp_est, disp_gt, mask)]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    # loss.backward()
    mono_loss.backward()
    optimizer.step()
    mono_optimizer.step()
    return tensor2float(loss), tensor2float(mono_loss), tensor2float(scalar_outputs), image_outputs

# test one sample
@make_nograd_func
def test_sample(batch_idx,sample, compute_metrics=True):
    model.eval()
    mono_model.eval()
    #model.train()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    disp_ests,confidence,index = model(imgL, imgR)
    disp_ests_final= disp_ests[-1]
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    # print(confidence.max(),confidence.min(),confidence.mean())
    mono_depth =mono_model(imgL, imgR)
    mono_final_depth =mono_depth[-1]
    mono_final_depth=torch.squeeze(mono_final_depth,dim=0)
    mono_disp_est =args.bf / mono_final_depth
    mono_loss = silog_criterion.forward(mono_final_depth, disp_gt, mask)
    if args.make_occ_mask:
        # print("occ mask")
        if args.r2l==False:
            args.r2l=True
            disp_ests_r2l,_,_ = model(imgR, imgL)
            args.r2l=False
        else:
            args.r2l=False
            disp_ests_r2l,_,_ = model(imgR, imgL)
            args.r2l=True
        disp_ests_final_r2l=disp_ests_r2l[-1]
        # print("disp type--------------",type(disp_ests_final),disp_ests_final.shape,disp_ests_final.dtype,type(disp_ests_final_r2l),disp_ests_final_r2l.shape,disp_ests_final_r2l.dtype)
        a=disp_ests_final.cuda()-disp_ests_final_r2l.cuda()
        
        #一致性遮挡并取阈值
        lr_consistency=lr_consistency_map(disp_ests_final.cuda(),disp_ests_final_r2l.cuda())
        mask_occ=(lr_consistency[-1]>args.occlude).float()
        mask_occ_re=(lr_consistency[-1]<=args.occlude).float()
        
        #0-1遮罩的生成
        mono=torch.abs(mono_disp_est-disp_gt)
        bio=torch.abs(disp_ests_final-disp_gt)
        mbb=(bio>mono).float()#m比b好
        bbm=(bio<mono).float()

        sum_mbb=torch.sum(mbb)
        sum_bbm=torch.sum(bbm)
        print(sum_mbb+sum_bbm)
        print("GT",sum_mbb/(sum_mbb+sum_bbm),"---")
        print()
        print()
        print()
        #还原浮点遮罩
        relu=nn.ReLU()
        mbb_float=relu(bio-mono)
        bbm_float=relu(mono-bio)

        #建模的伪更优点
        bio_sub_mono=disp_ests_final-mono_disp_est
        mono_sub_bio=mono_disp_est-disp_ests_final
        k=bio_sub_mono
        bbm_pred_unrelu=k.pow(2)+4*confidence*k
        bbm_pred=relu(bbm_pred_unrelu)

        mbb_pred=relu(-bbm_pred_unrelu)
        
        print("bbm_pred",bbm_pred.mean())
        a=torch.sum((bbm_pred>0).float())
        b=torch.sum((mbb_pred>0).float())
        print(a+b,b/(a+b))
        print()
        print(a,"------",b,"------")

        #伪遮罩取阈值
        mask_mbb_pred=(mbb_pred>args.mask).float()
        mask_bbm_pred=(bbm_pred>args.mask).float()
        
        #作用遮罩

        # disp_ests_unocc=disp_ests_final*mask_occ
        # mono_disp_ests_occ=mono_disp_ests*mask_occ_re
        disp_ests_unocc=disp_ests_final*bbm
        mono_disp_est_occ=mono_disp_est*mbb
        # print(disp_ests_unocc.shape,mono_disp_ests_occ.shape)
        disp_est_occ_refine=disp_ests_unocc+mono_disp_est_occ
        disp_est_origin=disp_ests[-1]
        disp_ests[-1]=disp_est_occ_refine



        


    #disp_ests = model(imgL, imgR)
    # print("left_fea",left_fea[0].shape)
    # print("right_fea",right_fea[0].shape)
    # print("cost",cost.shape)
    # disp_gt= F.upsample(torch.unsqueeze(disp_gt,dim=0),[disp_ests[0].size()[1], disp_ests[0].size()[2]], mode='bilinear')
    # disp_gt=torch.squeeze(disp_gt,dim=0)
    # print(disp_gt.size())
    

    #print(mask.shape)
    #select_fea_scale=left_fea[0]
    #svisualize_feature_map(select_fea_scale)
    #disp_ests[0][mask==0]=0
    
    #fea store
    # iml_2=F.upsample(imgL, [select_fea_scale.size()[2], select_fea_scale.size()[3]], mode='bilinear')
    # left_feature_path='/data/yyx/GwcNet-master/checkpoints/sceneflow_monkaa/gwcnet-gc/left_feature/'
    # _, H, W = select_fea_scale[0].shape
    # im_fea = torch.zeros(( H, W))
    # for f in range(select_fea_scale.size()[1]):
    #     im_fea = np.array(select_fea_scale[0][f,:, :].cpu()*255, dtype=np.uint16)
    #     cv.imwrite(join(left_feature_path, "left_feature-%d.png" % (f+1)),im_fea)
    # #im = np.array(iml_2[0,:,:,:].permute(1,2,0).cpu()*255, dtype=np.uint8)
    # im = np.array(iml_2[0,:,:,:].permute(1,2,0).cpu()*255, dtype=np.uint8)
    # cv.imwrite(join(left_feature_path, "itercolor-%d.jpg" % batch_idx),im)



    #driving test
    # test_gt_path='/data/yyx/GwcNet-master/checkpoints/sceneflow_monkaa/gwcnet-gc-gray/sceneflow_test_gt/'
    # test_pred1_path='/data/yyx/GwcNet-master/checkpoints/sceneflow_monkaa/gwcnet-gc-gray/sceneflow_test_pred/'
        
    # _, H, W = disp_ests[0].shape
    # gt=torch.zeros((H,W))
    # im_pred1 = torch.zeros(( H, W))

    # gt = np.array(disp_gt[0,:, :].cpu()*2, dtype=np.uint16)
    # cv.imwrite(join(test_gt_path, "sceneflow-gt-%d.png" % batch_idx),gt)
    # im_pred1 = np.array(disp_ests[3][0,:, :].cpu()*2, dtype=np.uint16)
    # cv.imwrite(join(test_pred1_path, "sceneflow-%d.png" % batch_idx),im_pred1)

    #kitti test
    # test_gt_path='/data/yyx/GwcNet-master/checkpoints/kitti/pri_192/kitti_test_gt/'
    # test_pred1_path='/data/yyx/GwcNet-master/checkpoints/kitti/pri_192/kitti_test_pred/'
    
    # # # os.makedirs(test_gt_path exist_ok=True)
    # # # os.makedirs(test_pred1_path, exist_ok=True)
    # _, H, W = disp_ests[0].shape
    # gt=torch.zeros((H,W))
    # im_pred1 = torch.zeros(( H, W))

    # gt = np.array(disp_gt[0,:, :].cpu()*256 +1, dtype=np.uint16)
    # cv.imwrite(join(test_gt_path,  "kitti-gt-%d.png" % batch_idx),gt)
    # im_pred1 = np.array(disp_ests[0][0,:, :].cpu()*256, dtype=np.uint16)
    # cv.imwrite(join(test_pred1_path,  "kitti-pred-%d.png" % batch_idx),im_pred1)
    

    #disp_ests[-1]=index.float()
    
    loss = model_loss(disp_ests, disp_gt,mask)
    #loss = model_loss(disp_ests, disp_gt,mask,imgL, imgR)
    #print(disp_ests[-1].dtype,index.dtype)
    if args.make_occ_mask:
        image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR, "mono_better_than_bio": mbb, "mask_occ": mask_occ,"confidence":confidence,"bbm_float":bbm_float,"mbb_float":mbb_float,"bbm_pred":bbm_pred,"mbb_pred":mbb_pred}
        image_outputs["errormap_origin"] = [disp_error_image_func()(disp_est_origin, disp_gt)]
    else:
        image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    # print(disp_ests[-1].shape,mono_disp_est.shape)
    scalar_outputs = {"loss": loss}
    image_outputs["errormap"] = [disp_error_image_func()(disp_ests[-1], disp_gt)]
    scalar_outputs["EPE"] = [EPE_metric(disp_ests[-1], disp_gt, mask)]
    image_outputs["mono errormap"] = [disp_error_image_func()(mono_disp_est, disp_gt)]
    scalar_outputs["mono EPE"] = [EPE_metric(mono_disp_est, disp_gt, mask)]
    scalar_outputs["mono D1"] = [D1_metric(mono_disp_est, disp_gt, mask)]
    scalar_outputs["D1"] = [D1_metric(disp_ests[-1], disp_gt, mask)]
    scalar_outputs["D3"] = [D3_metric(disp_ests[-1], disp_gt, mask)]
    scalar_outputs["Thres1"] = [Thres_metric(disp_ests[-1], disp_gt, mask, 1.0)]
    scalar_outputs["Thres2"] = [Thres_metric(disp_ests[-1], disp_gt, mask, 2.0)]
    scalar_outputs["Thres3"] = [Thres_metric(disp_ests[-1], disp_gt, mask, 3.0)]
    # scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    # scalar_outputs["D3"] = [D3_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    # scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    # scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    # scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    # scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    # scalar_outputs = {"loss": loss}
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    # scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    # scalar_outputs["D3"] = [D3_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    # scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    # scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    # scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    # scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(mono_loss), tensor2float(scalar_outputs), image_outputs

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
 
 
def visualize_feature_map(img_batch):
    feature_map = img_batch.cpu()
    feature_map=feature_map.data.numpy()
    print("fea-heat",feature_map.shape)
 
    feature_map_combination = []
    plt.figure()
 
    num_pic = feature_map.shape[1]
    print("num,pic",num_pic)
    row, col = get_row_col(num_pic)
    print(row,col)
 
    for i in range(0, num_pic):
        feature_map_split = feature_map[0,i, :, :]
        #print(feature_map_split.shape)
        feature_map_combination.append(feature_map_split)
        #plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        axis('off')
        filename='fea_map/kitti/0/'+str(i+1)+'.png'
        plt.savefig(filename)
    #plt.show()
 
    # 各个特征图按1：1 叠加
    feature_map_sum=feature_map_split[0]
    for i in range(len(feature_map_combination)-1):
        feature_map_sum = feature_map_combination[i+1]+feature_map_sum
    print("!!!!",feature_map_sum.shape)
    plt.imshow(feature_map_sum)
    plt.savefig('fea_map/kitti/0/'+'feature_map_sum.png')

def set_misc(model):
    if args.bn_no_track_stats:
        print("Disabling tracking running stats in batch norm layers")
        model.apply(bn_init_as_tf)

    if args.fix_first_conv_blocks:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', 'base_model.layer1.1', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'denseblock1.denselayer2', 'norm']
        print("Fixing first two conv blocks")
    elif args.fix_first_conv_block:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'norm']
        print("Fixing first conv block")
    else:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', '.bn']
        else:
            fixing_layers = ['conv0', 'norm']
        print("Fixing first conv layer")

    for name, child in model.named_children():
        if not 'encoder' in name:
            continue
        for name2, parameters in child.named_parameters():
            # print(name, name2)
            if any(x in name2 for x in fixing_layers):
                parameters.requires_grad = False





if __name__ == '__main__':
    train()
