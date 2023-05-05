import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from DataGenerator import DataGenerator
import pandas as pd 
from utils import dict_to_nonedict, parse, set_requires_grad, write_binary_dataset
from monai.networks import nets
import pandas as pd 

#########################################################################
def test(param):
    data_dir = param["data_dir"]
    df_test = param["df_test"] 
    df_test = pd.read_csv(df_test) 
    filenames = df_test[param['potential']].tolist()
    ckpt_folder = param["checkpoints"] 
    os.makedirs(ckpt_folder, exist_ok=True)
    model_name = os.path.join(ckpt_folder, 'model_{}.pth.tar'.format(param["savename"]))
    savefig_name = os.path.join(ckpt_folder, 'Loss_{}.png'.format(param["savename"]))
    loss_monitor_csv = os.path.join(ckpt_folder, 'Loss_{}.csv'.format(param["savename"]))
    cudnn.benchmark = True
    
    # initialize and load the model
    model = nets.AttentionUnet(spatial_dims=2, 
				in_channels=1, 
				out_channels=1,
				channels=[32, 64, 128, 256, 512, 512],
				strides=[2, 2, 2, 2, 2],
				dropout=0.0).cuda()

    if os.path.isfile(model_name):
        checkpoint = torch.load(model_name)
        print("=> Loading trained generator model")
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
    else:
        raise Exception("=> Fail to fine the checkpoint")
  	
    val_dataset = DataGenerator(data_dir, df_test, param['potential'], param['num_views'], transform=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1,
                             shuffle=False, num_workers=6, pin_memory=True, drop_last=False)
                     
    results_sep_folder = param["results_separate"] 
    os.makedirs(results_sep_folder, exist_ok=True)
    epochVal(model, val_loader, filenames, results_sep_folder)
    print('----------------------------------------------------------------------')

#-------------------------------------------------------------------------------- 
def epochVal(model, dataLoader, filenames, results_sep_folder): 
    model.eval()
    set_requires_grad(model, requires_grad=False)

    weight_factor = 1000 
    nimg = 1000 
    results_out = os.path.join(results_sep_folder, 'output')
    os.makedirs(results_out, exist_ok=True)
    
    count = 0 
    with torch.no_grad():
        for inp, target in tqdm (dataLoader):
            out = model(Variable(inp).cuda())
            sparse = inp.numpy().squeeze() * weight_factor
            dense = target.numpy().squeeze() * weight_factor
            output = out.cpu().detach().numpy().squeeze() * weight_factor
              
            filename = filenames[count].replace('.raw', '_rcn.raw') 
            write_binary_dataset(os.path.join(results_out, filename), output) 

            count += 1
    
    torch.cuda.empty_cache()


#-------------------------------------------------------------------------------- 
def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad
#--------------------------------------------------------------------------------         
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/hyperparam.json',
                        help='JSON file for configuration')
    return parser.parse_args()
    
if __name__ == '__main__':
    args = get_args()
    opt = parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = dict_to_nonedict(opt)
    test(param = opt)

    
