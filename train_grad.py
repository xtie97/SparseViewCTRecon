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
from utils import dict_to_nonedict, parse, set_requires_grad, write_binary_dataset, gradient_loss
from monai.networks import nets

#########################################################################
def train(param):
    data_dir = param["data_dir"]
    df_train = param["df_train"]
    df_val = param["df_val"] 
    df_train = pd.read_csv(df_train)
    df_val = pd.read_csv(df_val)
    
    ckpt_folder = param["checkpoints"] 
    os.makedirs(ckpt_folder, exist_ok=True)
    intermediate_results = param["intermediate"]
    os.makedirs(intermediate_results, exist_ok=True)
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
        print("=> Train the Generator from scratch")
   
    if os.path.isfile(loss_monitor_csv):
        print("=> Loading loss monitor csv")
        loss_param = pd.read_csv(loss_monitor_csv)
        losstrain_list = loss_param.train_loss.tolist()
        lossval_list = loss_param.val_loss.tolist()
        epochID_list = loss_param.epoch.tolist()
        loss_min = min(lossval_list)
        epoch_init = epochID_list[-1] + 1
    else:
        loss_min = 1e5
        losstrain_list = []
        lossval_list = [] 
        epochID_list = []
        epoch_init = 1 
    
    # Build the traininig and validation Dataloader
    # the default training/validation data is 140 kVp images and 123 view angles 
    train_dataset = DataGenerator(data_dir, df_train, 'high', 123, transform=True) 
    train_loader = DataLoader(dataset=train_dataset, batch_size=param["batch_size"],
                             shuffle=True, num_workers=6, pin_memory=True, drop_last=False)
    
    val_dataset = DataGenerator(data_dir, df_val, 'high', 123, transform=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=param["batch_size"],
                             shuffle=False, num_workers=6, pin_memory=True, drop_last=False)
                                            
    # Define the optimizer
    optimizer_G = optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters())}], lr=param["lrate"], betas=(0.9, 0.999), weight_decay=1e-5)  
    scheduler = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.2, patience=10, min_lr=1e-6) 
    # Define the LOSS 
    loss_l2 = nn.MSELoss(reduction='mean') 
    save_epoch = 0
    nonsave_epoch = 0
    for epochID in range(param["nepoch"]): 
        epochID = epochID + epoch_init
        # Training mode
        loss_gen = epochTrain(model, train_loader, optimizer_G, loss_l2)
        loss_gen_val = epochVal(model, val_loader, optimizer_G, loss_l2, results_folder=intermediate_results)
        scheduler.step(loss_gen_val)
        
        losstrain_list.append(loss_gen.cpu().ddetach().numpy())
        lossval_list.append(loss_gen_val.cpu().detach().numpy())
        epochID_list.append(epochID)
       
        if loss_gen_val < loss_min:
            loss_min = loss_gen_val
            torch.save({'epoch': epochID, 'state_dict': model.state_dict()}, model_name)
            print ('Epoch [' + str(epochID) + '] [save]  Training loss = ' + str(loss_gen))
            print ('Epoch [' + str(epochID) + '] [save]  Validation loss = ' + str(loss_gen_val))
            save_epoch = epochID     
        else:

            print ('Epoch [' + str(epochID) + '] [----]  Training loss = ' + str(loss_gen))
            print ('Epoch [' + str(epochID) + '] [----]  Validation loss = ' + str(loss_gen_val))
            nonsave_epoch = epochID 
            
        fig = plt.subplots(1, 1, figsize=(5, 5))
        plt.plot(losstrain_list, label='Training Loss')
        plt.plot(lossval_list, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc=0)
        plt.savefig(savefig_name, dpi=300, bbox_inches='tight')
        plt.close()
        loss_monitor = pd.DataFrame({'epoch': epochID_list, 'train_loss': losstrain_list, 'val_loss': lossval_list})
        loss_monitor.to_csv(loss_monitor_csv,index=False)
        print('----------------------------------------------------------------------')
        if nonsave_epoch >= (save_epoch + 20):
            break

#-------------------------------------------------------------------------------- 
def epochTrain (model, dataLoader, optimizer_G, loss_l2): 
    model.train()
    loss_gen = 0
    loss_gen_Norm = 0  
    accum_iter = 1 
    optimizer_G.zero_grad()
    for inp, target in tqdm (dataLoader):
        inp_var = Variable(inp).cuda()
        target_var = Variable(target).cuda()
        output = model(inp_var)
        lossl2 = loss_l2(output, target_var)
        lossgrad = gradient_loss(output, target_var)
        lossvalue = lossl2 + lossgrad * lamda 
        lossvalue /= accum_iter
        lossvalue.backward()
        
        if ((loss_gen_Norm + 1) % accum_iter == 0) or (loss_gen_Norm + 1 == len(dataLoader)):
            optimizer_G.step()
            optimizer_G.zero_grad()
            
        loss_gen += lossvalue.data * accum_iter
        loss_gen_Norm += 1 
    
    loss_gen /= loss_gen_Norm
    torch.cuda.empty_cache()
    return loss_gen
    
#-------------------------------------------------------------------------------- 
def epochVal(model, dataLoader, optimizer_G, loss_l2, results_folder): 
    model.eval()
    loss_gen = 0
    loss_gen_Norm = 0  
    weight_factor = 1000  

    with torch.no_grad():
        for inp, target in tqdm (dataLoader):
            inp_var = Variable(inp).cuda()
            target_var = Variable(target).cuda()
            output = model(inp_var)
            
            lossl2 = loss_l2(output, target_var)
            lossgrad = gradient_loss(output, target_var)
            lossvalue = lossl2 + lossgrad * lamda 
            loss_gen += lossvalue.data
            
            if loss_gen_Norm % 10 == 0: 
                target_array = target[:,0,:,:].numpy() * weight_factor 
                write_binary_dataset(os.path.join(results_folder, 'target_{:04d}.raw'.format(loss_gen_Norm)), target_array) 
                output_array = output[:,0,:,:].cpu().detach().numpy() * weight_factor
                write_binary_dataset(os.path.join(results_folder, 'output_{:04d}.raw'.format(loss_gen_Norm)), output_array) 
                input_array = inp[:,0,:,:].numpy() * weight_factor 
                write_binary_dataset(os.path.join(results_folder, 'input_{:04d}.raw'.format(loss_gen_Norm)), input_array) 
		
            loss_gen_Norm += 1 
    
    loss_gen /= loss_gen
    torch.cuda.empty_cache()
    return loss_gen
    
   
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/hyperparam_grad.json',
                        help='JSON file for configuration')
    return parser.parse_args()
    
if __name__ == '__main__':
    args = get_args()
    opt = parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = dict_to_nonedict(opt)
    global lamda
    lamda = opt["lamda"]
    train(param = opt)

    
    
    
    
