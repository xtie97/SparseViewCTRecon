import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn.functional as TF
import numpy as np
import random 
import torch.nn as nn 
import torchvision
from utils import read_binary_dataset 
 
class DataGenerator(Dataset):
    def __init__(self, root_dir, df, potential, num_views, transform=False, data_len=-1):
        """ 
    	It loads images from the specified folder 
    	root_dir: the dictionary contains all data
    	potential: 'high' or 'low' 
    	num_views: number of view angles after the downsampling, choose from [61, 123, 246] 
    	"""
        self.root_dir = root_dir
        self.high = df[potential].tolist() 
        self.num_views = num_views 
        self.transform = transform 
        if data_len==-1:
            self.data_len = 1e5  
        else:
            self.data_len = data_len
    
    def __len__(self):
    	return min(self.data_len, len(self.high)) 
        
    def __getitem__(self, index):
        weight_factor = 1000
        
        filename = self.high[index].replace('.raw', '_rcn.raw') 
        dense = read_binary_dataset(os.path.join(self.root_dir, 'dense_view', filename), (512, 512)) / weight_factor
        sparse = read_binary_dataset(os.path.join(self.root_dir, 'sparse_view_{}'.format(self.num_views), filename), (512, 512)) / weight_factor 

        inputs = np.concatenate((sparse, dense), axis=0)
        inputs = torch.from_numpy(inputs)
       
        if self.transform:
            inputs = transforms.RandomHorizontalFlip(p=0.5)(inputs)
            inputs = transforms.RandomVerticalFlip(p=0.5)(inputs)
            inputs = transforms.RandomRotation(degrees=180)(inputs) # 0 or -1
        
        return inputs[0:1,:,:], inputs[1:2,:,:]
        
class DataGenerator_sino(Dataset):
    def __init__(self, root_dir, df, potential, num_views, data_len=-1):
        """
        It loads sinograms from the specified folder 
    	root_dir: the dictionary contains all data
    	potential: 'high' or 'low' 
    	num_views: number of view angles after the downsampling, choose from [61, 123, 246] 
    	"""
        self.root_dir = root_dir
        self.high = df[potential].tolist() 
        self.num_views = num_views 
        if data_len==-1:
            self.data_len = 1e5  
        else:
            self.data_len = data_len
    
    def __len__(self):
    	return min(self.data_len, len(self.high)) 
    
    def __getitem__(self, index):
        weight_factor = 5e5
        
        filename = self.high[index]
        inputs = read_binary_dataset(os.path.join(self.root_dir, filename), (984, 888)) / weight_factor 
        targets = read_binary_dataset(os.path.join('/media/xintie/Elements/DeepEnChroma/Data/', filename), (1968,888)) / weight_factor 
        targets = targets[:, 0:1968:2, :] # The real acquisition is 984 views, 1968 views is caused by interpolation 
        
        mask = np.zeros_like(inputs)
        downsample_factor = 984 // self.num_views
        mask[0:984:downsample_factor, :] = 1.0 # actually sampled view angles 
        inputs = inputs *(1-mask) + targets * mask 
       
        return torch.from_numpy(inputs), torch.from_numpy(targets)
        
        
        
