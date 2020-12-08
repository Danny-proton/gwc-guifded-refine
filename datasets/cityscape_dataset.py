'''
import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from . import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    data = Image.open(path).convert('RGB')
    # print(data.size)
    im_h = (data.size[-2] // 64) * 64
    im_w = (data.size[-1] // 64) * 64
    data = data.crop((0, 0, im_h, im_w))
    # print(data.size)
    return data


def disparity_loader(path):
    data = Image.open(path)
    im_h = (data.size[-2] // 64) * 64
    im_w = (data.size[-1] // 64) * 64
    data = data.crop((0, 0, im_h, im_w))
    return data


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training,normalize, datapath,loader=default_loader,
                 dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.normalize = normalize
        self.datapath=datapath

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(os.path.join(self.datapath, left))
        right_img = self.loader(os.path.join(self.datapath, right))
        dataL = self.dploader(os.path.join(self.datapath, disp_L))

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = (np.ascontiguousarray(dataL, dtype=np.float32) -1)/ 256
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False, normalize=self.normalize)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL
        else:
            # w, h = left_img.size

            # left_img = left_img.crop((w-1232, h-368, w, h))
            # right_img = right_img.crop((w-1232, h-368, w, h))
            # w1, h1 = left_img.size

            # dataL = dataL.crop((w-1232, h-368, w, h))
            # dataL = (np.ascontiguousarray(dataL, dtype=np.float32)-1) / 256
            #
            # processed = preprocess.get_transform(augment=False, normalize=self.normalize)
            # left_img = processed(left_img)
            # right_img = processed(right_img)
            #
            # return left_img, right_img, dataL

            w, h = left_img.size
            th, tw = 512, 1024

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = (np.ascontiguousarray(dataL, dtype=np.float32) - 1) / 256
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False, normalize=self.normalize)
            left_img = processed(left_img)
            right_img = processed(right_img)
            return left_img, right_img, dataL
    def __len__(self):
        return len(self.left)
'''

import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread


class CityscapeDatset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        #data, scale = pfm_imread(filename)
        data = Image.open(filename)
        data = (np.ascontiguousarray(data, dtype=np.float32)-1)/256
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        #print(self.datapath)
        #print(self.left_filenames[index])
        #print(os.path.join(self.datapath, self.left_filenames[index]))
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            w, h = left_img.size
            crop_w, crop_h = 1024, 512
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "top_pad": 0,
                    "right_pad": 0}
