#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import os.path as osp

cp_dir   = './experiments_final'
def cls_args():

    parser = argparse.ArgumentParser(description='PyTorch for DUL-classification')

    # -- env
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0])
    parser.add_argument('--workers', type=int,  default=0)  # TODO
    parser.add_argument('--seed', type=int,  default=123)  # TODO

    # -- model
    parser.add_argument('--drop_ratio', type=float, default=0.4)          # TODO
    parser.add_argument('--classnum',   type=int,   default=10)  #dawn      

    # -- loss function
    parser.add_argument('--loss_mode_EU',  type=str,    default='EDL', choices=['Re_EDL', 'EDL', 'UMSE'])
    parser.add_argument('--loss_mode_AU',  type=str,    default='ce')
    parser.add_argument('--kl_lambda',  type=float,  default=0.01)         # default = 0.01


    # -- cosinesoftmax
    parser.add_argument('--in_feats',   type=int,   default=512) # fixed
    parser.add_argument('--scale',      type=float,  default=16)           # FIXED default=64

    # -- optimizer
    parser.add_argument('--start_epoch', type=int,   default=1)        #
    parser.add_argument('--end_epoch',   type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=8)      # TODO | 64
    parser.add_argument('--base_lr',     type=float, default=0.01)      # default = 0.1
    parser.add_argument('--lr_adjust',   type=list,  default=[12, 20, 30, 42])
    parser.add_argument('--gamma',       type=float, default=0.3)      # FIXED
    parser.add_argument('--weight_decay',type=float, default=5e-4)     # FIXED
    parser.add_argument('--resume',      type=str,   default='')       # checkpoint

    # -- dataset
    parser.add_argument('--data_train_path', type=str,
                        default=r'/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/MSTAR/SOC/TRAIN')  # 训练数据路径 #dawn
    parser.add_argument('--data_test_dir',   type=str,  default=r'/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/MSTAR/SOC/TEST')   # TODO #dawn

    # -- save or print
    parser.add_argument('--is_debug',  type=str,   default=False)   # TODO
    parser.add_argument('--save_to',   type=str,   default=osp.join(cp_dir, 'bs8')) # dawn
    parser.add_argument('--print_freq',type=int,   default=200)  
    parser.add_argument('--save_freq', type=int,   default=3)  # TODO

    ## -- test model and save the metrics
    parser.add_argument('--save_excel_one_branch', type=str, default=r'./results/entropy_baseline.xlsx')
    parser.add_argument('--save_excel_two_branch', type=str, default=r'./results/ours_0.01.xlsx') # 0.01 means the value of kl_lambda, which is included in model name
    parser.add_argument('--uncertainty_mode', type=str, default='evidence', choices=['variance', 'entropy', 'evidence'])
    parser.add_argument('--model_path_one_branch', type=str, default='/scratch/project_2014644/OOD_uniform_framework/models_trained_final/bs8/seed_3407/one_branch_EDL_epoch_100.pth')
    parser.add_argument('--model_path_dual_branch', type=str, default='/scratch/project_2014644/OOD_uniform_framework/models_trained_final/bs8/seed_3407/trained_0.01_EDL_epoch_100.pth')

    args = parser.parse_args()
    return args
