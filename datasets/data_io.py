import numpy as np
import re
import torchvision.transforms as transforms


def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    #print("!!!!!!!!!!!!!!!!!")
    # mean = [0.375, 0.397, 0.385]
    # std = [0.302, 0.314, 0.325]
    # mean = [0.399]
    # std = [0.1871]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

class Gray(object):

    def __call__(self, tensor):
        # TODO: make efficient
        R = tensor[:, 0, :, :]
        G = tensor[:, 1, :, :]
        B = tensor[:, 2, :, :]
        tensor[0]=0.299*R+0.587*G+0.114*B
        tensor = tensor[0]
        tensor = tensor.view(tensor.size()[0],tensor.size()[2],tensor.size()[3])
        return tensor

# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


# read an .pfm file into numpy array, used to load SceneFlow disparity files
def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
