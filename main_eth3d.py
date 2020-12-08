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
from models import __models__, model_loss
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
matplotlib.use('pdf')


cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--model', default='gwcnet-gc', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=384, help='maximum disparity')

parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
#parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--resume', default=False, help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

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

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

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
print("start at epoch {}".format(start_epoch))


def train():
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)
        print("current rate is ",args.lr)
        print(args.maxdisp)
        #avg_train_scalars = AverageMeter()
        all_loss=0.0
        #print(all_loss)
        # training
        # nannum=0
        # for batch_idx, sample in enumerate(TrainImgLoader):
        #     #print(len(sample))
        #     #print("!!!!!!!!!!!!!")
        #     global_step = len(TrainImgLoader) * epoch_idx + batch_idx
        #     start_time = time.time()
        #     do_summary = global_step % args.summary_freq == 0
        #     loss, scalar_outputs, image_outputs = train_sample(sample,compute_metrics=do_summary)
        #     #loss_all=loss_all+loss
        #     if math.isnan(loss):
        #         print(batch_idx,'loss is nan')
        #         nannum=nannum+1
        #     else:
        #         all_loss=loss+all_loss
        #     if do_summary:
        #         save_scalars(logger, 'train', scalar_outputs, global_step)
        #         save_images(logger, 'train', image_outputs, global_step)

        #     #avg_train_scalars.update(loss)
        #     #print("!!!",avg_train_scalars)
        #     del scalar_outputs, image_outputs
        #     if (batch_idx%100==0)or(batch_idx==len(TrainImgLoader)-1):
        #         print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f},losses averange={:.3f}'.format(epoch_idx, args.epochs,
        #                                                                                batch_idx,
        #                                                                                len(TrainImgLoader), loss,
        #                                                                                time.time() - start_time,all_loss/(batch_idx-nannum+1)))

                
        # #loss_all=loss_all.avg()
        # #avg_train_scalars=avg_train_scalars.mean()
        # #print("avg_train_loss",all_loss/(len(TestImgLoader)-nannum))
        #         #print("losses",loss_all/(batch_idx+1))
        # #print("loss averange=",loss_all/12537.0)
        # #saving checkpoints
        # if (epoch_idx + 1) % args.save_freq == 0:
        #     checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        #     torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        # gc.collect()
        # print("avg_train_loss",all_loss/(len(TrainImgLoader)-nannum))

        # testing
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            #print("sample",sample)
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(batch_idx,sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            if (batch_idx%100==0)or(batch_idx==len(TestImgLoader)-1):
                print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time))
            #print(avg_test_scalars["EPE"])
        avg_test_scalars = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        gc.collect()


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    optimizer.zero_grad()

    disp_ests = model(imgL, imgR)

    #print(len(disp_ests))
    #print(disp_ests[0].shape)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    #print("##########",mask.shape)
    loss = model_loss(disp_ests, disp_gt, mask)
    #loss = model_loss(disp_ests, disp_gt, mask,imgL, imgR)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

# test one sample
@make_nograd_func
def test_sample(batch_idx,sample, compute_metrics=True):
    model.eval()
    #model.train()

    imgL, imgR, disp_gt ,mask= sample['left'], sample['right'], sample['disparity'] ,sample["mask"]
    imgL = imgL.cuda()
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    mask=mask.cuda()

    disp_gt = disp_gt.cuda()
    disp_ests = model(imgL, imgR)
    #disp_ests = model(imgL, imgR)
    # print("left_fea",left_fea[0].shape)
    # print("right_fea",right_fea[0].shape)
    # print("cost",cost.shape)
    #mask = (disp_gt < args.maxdisp) & (disp_gt > 0)

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
    


    loss = model_loss(disp_ests, disp_gt,mask)
    #loss = model_loss(disp_ests, disp_gt,mask,imgL, imgR)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    image_outputs["errormap"] = [disp_error_image_func()(disp_ests[-1], disp_gt)]
    scalar_outputs["EPE"] = [EPE_metric(disp_ests[-1], disp_gt, mask)]
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

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

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

if __name__ == '__main__':
    train()
