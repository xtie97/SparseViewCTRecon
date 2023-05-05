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

def test(flag, rfov=250):
    ############### FP ###############
    df_train = pd.read_csv('./data_sparse_view/{}_index_140kV.csv'.format(flag))
    filename_high = df_train.high.to_list() 
    nview_list = df_train.nview.tolist()
    start_angle_list = df_train.start_angle.tolist()
    start_angle_list = [-ii-180 for ii in start_angle_list] # -ii-180
   
    rfov_list = df_train.FOV.to_list()
    xmin_list = df_train.ymin.to_list()
    ymin_list = df_train.xmin.to_list() 
    
    root_dir = '/media/xintie/Elements/DeepEnChroma/Data_rcn/results_grad_61/output/'
    target_dir = '/media/xintie/Elements/DeepEnChroma/Data_rcn/results_grad_61/output_sino/'
    os.makedirs(target_dir, exist_ok=True)
    
    for fileid in tqdm(range(len(filename_high))):
        filename = filename_high[fileid]
        xmin = -rfov
        ymin = -rfov 
        
        fstartangle = start_angle_list[fileid] 
        nview = 984 #nview_list[fileid] 
 
        recon_file = root_dir + filename_high[fileid].replace(".raw", "_rcn.raw")
        prj_file = target_dir + filename_high[fileid]
        config_file = '/home/xintie/Mayo_LDCT/Fan_CUDAC/GPUReconConfig.in'
        modify_file(config_file, prj_file, recon_file, xmin, ymin, rfov, fstartangle, nview=nview) 
        os.system("/home/xintie/Mayo_LDCT/Fan_CUDAC/FP/GPUFP {}".format(config_file))
            
       
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
            line = "FOVRadius=" + "{}".format(rfov)
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
    mul_factor = 10000
    start = 0
    rfov = args.rfov
    flag = args.flag # test 
    
    test(flag='test', rfov=rfov)
    
