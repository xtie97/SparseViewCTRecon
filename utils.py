import numpy as np
import math 
import operator 
import time 
import os 
from collections import OrderedDict
import json
import torch 
import torch.nn.functional as TF

def mkdir(path):
    """build a new folder 
    """
    folder = os.path.exists(path)
    if not folder:                   
        os.makedirs(path)            
        print('---  new folder  ---')
        print('---  OK  ---')
        
def read_binary_dataset(file_name, dim):   # please pay attentions to the order of data!\n"
    f = open(file_name, 'rb')
    data = np.fromfile(f, dtype=np.float32, count=-1)
    f.close()
    data = np.reshape(data, (-1, dim[0], dim[1]), order='C') # -1: slice 
    return data

def write_binary_dataset(filename, data):
    f=open(filename, "wb")
    f.write(bytearray(data))
    f.close() 
  
# convert to NoneDict, which return None for missing key.
class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def parse(args):
    opt_path = args.config
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    return json.loads(json_str, object_pairs_hook=OrderedDict)

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def gradient_loss(gen_frames, gt_frames, alpha=1):
    gen_dx, gen_dy = gradient(gen_frames)
    gt_dx, gt_dy = gradient(gt_frames)
    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)

    # condense into one tensor and avg
    return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)

# gradient
def gradient(x):
    # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
    # x: (b,c,h,w), float32 or float64
    # dx, dy: (b,c,h,w)
    # gradient step = 1
    left = x
    right = TF.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = TF.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
    dx, dy = right - left, bottom - top 
    # dx will always have zeros in the last column, right-left
    # dy will always have zeros in the last row,    bottom-top
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy 
         
