#! /usr/bin/python
# -*- coding: utf8 -*-
import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import pandas as pd 
from tqdm import *
import math 
import re 
import argparse
from utils import read_binary_dataset, write_binary_dataset

def test(flag, rfov=250):
    ############### FBP ###############
    df_train = pd.read_csv('./data_sparse_view/{}_index_140kV.csv'.format(flag))
    filename_high = df_train.low.to_list() 
    
    nview_list = df_train.nview.tolist()
    start_angle_list = df_train.start_angle.tolist()
    start_angle_list = [-ii-180 for ii in start_angle_list] # -ii-180
    #ncol_list = df_train.ncol.tolist()
    #dshift_list = df_train.dshift.tolist() 
    
    rfov_list = df_train.FOV.to_list()
    xmin_list = df_train.ymin.to_list()
    ymin_list = df_train.xmin.to_list() 
    
    root_dir = '/media/xintie/Elements/DeepEnChroma/Data/'
    target_dir_dense = '/media/xintie/Elements/DeepEnChroma/Data_rcn/dense_view/'
    os.makedirs(target_dir_dense, exist_ok=True)
    downsample_factor = 16  
    target_dir_sparse = '/media/xintie/Elements/DeepEnChroma/Data_rcn/sparse_view_{}/'.format(984//downsample_factor)
    os.makedirs(target_dir_sparse, exist_ok=True)
    
    for fileid in tqdm(range(len(filename_high))):
        xmin = -rfov 
        ymin = -rfov 
        
        fstartangle = start_angle_list[fileid] 
        nview = nview_list[fileid] 
        
        readfile = root_dir + filename_high[fileid]
        savefile = target_dir_dense + filename_high[fileid].replace(".raw", "_rcn.raw")
        config_file = '/home/xintie/Mayo_LDCT/Fan_CUDAC/GPUReconConfig.in'
        modify_file(config_file, readfile, savefile, xmin, ymin, rfov, fstartangle, nview) #rfov = 180
        os.system("/home/xintie/Mayo_LDCT/Fan_CUDAC/FBP/GPUFBPRecon {}".format(config_file))
        
        prj = read_binary_dataset(readfile, (nview, 888)).squeeze()
        prj = prj[0:1968:2, :]
        prj = prj[0:984:downsample_factor, :] # 4->246, 8->123, 16->62 
        readfile_temp = './prj_temp.raw'
        write_binary_dataset(readfile_temp, prj) 
        savefile = target_dir_sparse + filename_high[fileid].replace(".raw", "_rcn.raw")
        config_file = '/home/xintie/Mayo_LDCT/Fan_CUDAC/GPUReconConfig.in'
        modify_file(config_file, readfile_temp, savefile, xmin, ymin, rfov, fstartangle, 1968//downsample_factor) 
        os.system("/home/xintie/Mayo_LDCT/Fan_CUDAC/FBP/GPUFBPRecon {}".format(config_file))
        
        
def modify_file(target_file, read_file, save_file, xmin, ymin, rfov, startangle=0, nview=1968, dshift=1.125):
    file = open("/home/xintie/Mayo_LDCT/Fan_CUDAC/FBP/GPUReconConfig.in", "r")
    replacement = ""
    nX = 512
    for line in file:
        line = line.strip()
        if "ProjectionFilename=" in line:
            line = "ProjectionFilename=" + read_file #"/home/xintie/DeepEnChroma/data/exam47_m2_50.raw"
        elif "ReconstructionFilename=" in line:
            line = "ReconstructionFilename=" + save_file #"/home/xintie/DeepEnChroma/data/exam47_m2_50_rcn.raw"
        elif "StartAngle=" in line:
            line = "StartAngle=" + "{}".format(startangle) 
        elif "ProjectionsPerRotation=" in line:
            line = "ProjectionsPerRotation=" + "{}".format(nview) # number of views = 1968
        elif "ProjectionNumber=" in line:
            line = "ProjectionNumber=" + "{}".format(nview)
        elif "XMin=" in line:
            line = "XMin=" + "{}".format(xmin)
        elif "YMin=" in line:
            line = "YMin=" + "{}".format(ymin)
        elif "DeltaX=" in line: 
            line = "DeltaX=" + "{}".format(rfov*2/nX)
        elif "DeltaY=" in line: 
            line = "DeltaY=" + "{}".format(rfov*2/nX)
        elif "FOVRadius=" in line:
            line = "FOVRadius=" + "{}".format(250)
        elif "DetectorOffset=" in line:
            line = "DetectorOffset=" + "{}".format(dshift)
            
        replacement = replacement + line + "\n"    

    file.close()
    #target_file="./GPUReconConfig.in"
    fout = open(target_file, "w")
    fout.write(replacement)
    fout.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train a model on a given dataset.')
    parser.add_argument('-rfov', dest='rfov', metavar='rfov', type=float, required=False, default=250, help='FOV')
    parser.add_argument('-flag', dest='flag', metavar='flag', type=str, required=False, default='test', help='test')
    return parser.parse_args()
         
if __name__ == '__main__':
    args = get_args()

    rfov = args.rfov # default 25cm 
    #test(flag='train', rfov=rfov)
    #test(flag='val', rfov=rfov)
    test(flag='test', rfov=rfov)

    
