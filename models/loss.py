import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import math
import torch.nn as nn

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        eps = 1e-7
        # depth_est=torch.masked_select(depth_est,mask)
        # depth_gt=torch.masked_select(depth_gt,mask)
        # print(depth_est.shape,depth_gt.shape)
        d = torch.log(depth_est[mask]+eps) - torch.log(depth_gt[mask]+eps)
        # print("我老婆叫新垣结衣")
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

def model_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    #print(len(disp_ests))
    #print(disp_ests[0].shape)
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        # disp_gt= F.upsample(torch.unsqueeze(disp_gt,dim=0),[disp_est.size()[1], disp_est.size()[2]], mode='bilinear')
        # disp_gt=torch.squeeze(disp_gt)
        # error_abs=torch.abs(disp_est-disp_gt)
        # mask_confidence=error_abs<=1.0
        # print("len",len(confidence[mask_confidence]))
        # print("confidence",confidence[0][100][:])
        # print("index",index[0][100][:])
        # print("disparity",disp_est[0][100][:])
        #print("gt",disp_gt[0][24000:24240])
        
        #confidence_hw=confidence.view(-1)
        #confidence_hw=np.array(confidence_hw)
        #print(len(confidence_hw))
        #plt.xlabel('confidence')
        #plt.ylabel('Frequency')
        #plt.xlim(xmax=1, xmin=0)
        #plt.hist(x=confidence_hw, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.9)
        #plt.imshow(n)
        #filename='fea_map/kitti/confidence/'+str(batch_id+1)+'.png'
        #filename='fea_map/0.png'
        #plt.savefig(filename)
       # plt.close()
        # print(type(disp_gt))
        
        
        
        # disp_est=torch.tensor(disp_est).float()
        # out=torch.masked_select(disp_est,mask)
        # gt=torch.masked_select(disp_gt,mask)
        # # print(type(out),type(gt))
        # all_losses.append(weight * F.smooth_l1_loss(out,gt, size_average=True))
        
        
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)


'''
import torch
import torch.nn as nn
import numpy as np
import pdb

def model_loss(disp_ests, disp_gt, mask,left,right):
    # left=color_to_gray(left)
    # right=color_to_gray(right)
    #print(left.shape)
    weights = [5.0, 5.0, 7.0, 10.0]
    all_losses = []
    left=torch.squeeze(color_to_gray(left),dim=1)
    for disp_est, weight in zip(disp_ests, weights):
        #disp_est=torch.unsqueeze(disp_est,dim=1)
        #print("####",disp_est.shape,mask.shape)
        rebuild_left=torch.squeeze(color_to_gray(generate_image_left(right,torch.unsqueeze(disp_est,dim=1))),dim=1)
        
        #print(rebuild_left.shape)
        #all_losses.append(weight * F.smooth_l1_loss(rebuild_left[mask], left[mask], size_average=True))
        lr_error=torch.abs(rebuild_left-left)
        lr_error = torch.where(lr_error > 0.6, torch.full_like(lr_error, 0), lr_error) 
        #lr_error = torch.where(disp_gt < 0, torch.full_like(lr_error, 0), lr_error)
        lr_loss=weight * torch.mean(lr_error)
        #print(SSIM(disp_est,left).shape)
        #ssim_loss=torch.mean(SSIM(disp_est,left))
        all_losses.append(lr_loss)
    return sum(all_losses)

def generate_image_left(img, disp):
        # print img.shape, disp.shape
        return bilinear_sampler_1d_h(img, -disp)

def generate_image_right(img, disp):
        return bilinear_sampler_1d_h(img, disp)
        
def color_to_gray(img):
    img = img.permute(0, 2, 3, 1)
    img_gray = 0.299*img[:,:,:,0]+0.587*img[:,:,:,1]+0.114*img[:,:,:,2]
    img_gray = img_gray.unsqueeze(3).permute(0, 3, 1, 2)
    # im = np.array(imgL[0,:,:,:].cpu().permute(1,2,0)*255, dtype=np.uint8)        
    return img_gray

def SSIM(x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, 3, 1)
        mu_y = F.avg_pool2d(y,3, 1)

        sigma_x  = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
        sigma_y  = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y , 3, 1) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)


def bilinear_sampler_1d_h(input_images, x_offset, wrap_mode='border', name='bilinear_sampler', **kwargs):
    def repeat(x, n_repeats):
        rep  = torch.unsqueeze(x,1).repeat([1,n_repeats])
        return torch.reshape(rep, [-1])

    def interpolate(im, x, y):
        # handle both texture border types
        edge_size = 0
        if wrap_mode == 'border':
                edge_size = 1
                im = nn.functional.pad(im, (0,0,1,1,1,1,0,0), mode='constant')
                x = x + edge_size
                y = y + edge_size
        elif wrap_mode == 'edge':
            edge_size = 0
        else:
            return None
            
        x = torch.clamp(x, 0.0,  width_f - 1 + 2 * edge_size)
        
        x0_f = torch.floor(x)
        y0_f = torch.floor(y)
        x1_f = x0_f + 1

        x0 = x0_f.int()
        y0 = y0_f.int()
        x1 = torch.min(x1_f,  torch.tensor(width_f - 1 + 2 * edge_size).cuda()).int()

        dim2 = (width + 2 * edge_size)
        dim1 = (width + 2 * edge_size) * (height + 2 * edge_size)
        
        base = repeat(torch.arange(0, num_batch) * dim1, height * width).cuda().int()
        base_y0 = base + y0 * dim2
        idx_l = base_y0 + x0
        idx_r = base_y0 + x1

        im_flat = im.reshape((-1, num_channels))

        idx_l = idx_l.repeat(num_channels, 1).t().long()
        idx_r = idx_r.repeat(num_channels, 1).t().long()

        pix_l = im_flat.gather(0, idx_l)
        pix_r = im_flat.gather(0, idx_r)
        
        weight_l = torch.unsqueeze(x1_f - x, 1)
        weight_r = torch.unsqueeze(x - x0_f, 1)

        return weight_l * pix_l + weight_r * pix_r

    def transform(input_images, x_offset):
        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        x_t, y_t = np.meshgrid(torch.linspace(0.0,   width_f - 1.0,  width).numpy(),
                                torch.linspace(0.0 , height_f - 1.0 , height).numpy())
      
        x_t_flat = torch.reshape(torch.from_numpy(x_t), (1, -1)).cuda().float()
        y_t_flat = torch.reshape(torch.from_numpy(y_t), (1, -1)).cuda().float()
       
        x_t_flat = x_t_flat.repeat((num_batch, 1))
        y_t_flat = y_t_flat.repeat((num_batch, 1))
      
        x_t_flat = torch.reshape(x_t_flat, [-1])
        y_t_flat = torch.reshape(y_t_flat, [-1])
        # print('###################################',x_t_flat.type())
        x_t_flat = x_t_flat + torch.reshape(x_offset, [-1])
        

        input_transformed = interpolate(input_images, x_t_flat, y_t_flat)

        output = torch.reshape(
            input_transformed, (num_batch, height, width, num_channels))
        output = output.permute(0, 3, 1, 2)
        return output
    #print("!!!",input_images.shape)
    input_images = input_images.permute(0, 2, 3, 1)
    x_offset = x_offset.permute(0, 2, 3, 1)
    num_batch    = input_images.shape[0]
    height       = input_images.shape[1]
    width        = input_images.shape[2]
    num_channels = input_images.shape[3]

    height_f = float(height)
    width_f  = float(width)
    
    wrap_mode = wrap_mode

    output = transform(input_images, x_offset)
    return output
'''