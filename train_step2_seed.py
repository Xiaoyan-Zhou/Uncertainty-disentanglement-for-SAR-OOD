#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import torch
import shutil
import random
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import os.path as osp
import model as mlib
from data_utils import SAR_DatasetLoader
from train_cls_config_simplex import cls_args

from IPython import embed

import math
import copy
from torchvision.models import resnet18
import random as python_random


def setup_seed(seed):
    torch.cuda.set_device(0)
    python_random.seed(seed)
    
    torch.manual_seed(seed)         # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)    # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    np.random.seed(seed)            # Numpy模块的随机种子
    random.seed(seed)               # Python内置随机模块的种子

def set_seed(seed):
    # 设置 Python 的随机种子
    random.seed(seed)
    
    # 设置 Numpy 的随机种子
    np.random.seed(seed)
    
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)  # 设置 CPU 随机种子
    torch.cuda.manual_seed(seed)  # 设置 GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU，设置所有 GPU 的随机种子
    
    # 确保每次运行时的结果是相同的（如果要进行梯度计算时设置这项）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")

class DulClsTrainer(mlib.Faster1v1):

    def __init__(self, args):
        mlib.Faster1v1.__init__(self, args)
        self.args    = args
        self.model   = dict()
        self.data    = dict()
        self.result  = dict()
        self.softmax = torch.nn.Softmax(dim=1)
        self.use_gpu = args.use_gpu and torch.cuda.is_available()

    def _report_settings(self):
        ''' Report the settings '''

        str = '-' * 16
        print('%sEnvironment Versions%s' % (str, str))
        print("- Python    : {}".format(sys.version.strip().split('|')[0]))
        print("- PyTorch   : {}".format(torch.__version__))
        print("- TorchVison: {}".format(torchvision.__version__))
        print("- USE_GPU   : {}".format(self.use_gpu))
        print("- IS_DEBUG  : {}".format(self.args.is_debug))
        print('-' * 52)


    def _model_loader(self):
        
        model_resnet = resnet18(weights=None)
        self.model['backbone']  =  torch.nn.Sequential(*list(model_resnet.children())[:-2]) # Excludes the final FC layer
    
        # load weights of backbone
        self.step1_model_path = '%s/AU_backbone_%s_epoch_%02d.pth' % \
                    (self.args.save_to, str(self.args.kl_lambda)+'_'+self.args.loss_mode_AU, 
                    self.args.end_epoch)
        self.saved_model = torch.load(self.step1_model_path) #dawn
        self.model['backbone'].load_state_dict(self.saved_model['backbone'])

        self.model['EU_branch'] = mlib.EU_branch(self.args)
        self.model['criterion_EU'] = mlib.ClsLoss(self.args, branch='EU') # dawn 定义EU分支的损失函数
        # optimizer for EU branch only
        self.model['optimizer_EU'] = torch.optim.SGD(
                                    [{'params': self.model['EU_branch'].parameters()}],
                                    lr=self.args.base_lr,
                                    dampening=0,
                                    weight_decay=self.args.weight_decay,
                                    momentum=0.9,
                                    nesterov=False) #定义优化器
        self.model['scheduler_EU'] = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model['optimizer_EU'], float(self.args.end_epoch))
        
        if self.use_gpu:
            self.model['backbone']  = self.model['backbone'].cuda()
            self.model['EU_branch']  = self.model['EU_branch'].cuda()
            self.model['criterion_EU'] = self.model['criterion_EU'].cuda() #setting of loss function for model parameter
    
    def _data_loader(self):

        train_set = SAR_DatasetLoader('train', self.args.data_train_path)
        test_set = SAR_DatasetLoader('test', self.args.data_test_dir)
        self.data['train'] = DataLoader(dataset=train_set, batch_size=self.args.batch_size, shuffle=True, num_workers=0) 
        self.data['test'] = DataLoader(dataset=test_set, batch_size=self.args.batch_size, shuffle=True, num_workers=0)  

        print('Data loading was finished ...')

    def _train_one_epoch(self, epoch = 0, sample_time=2, num_classes=10):

        self.model['backbone'].train() # results in experiments
        # self.model['backbone'].eval()
        self.model['EU_branch'].train()
        predictions = []

        loss_recorder_eu, batch_acc = [], []
        for idx, (img, gty) in enumerate(self.data['train']):

            img.requires_grad = False
            gty.requires_grad = False

            if self.use_gpu:
                img = img.cuda()
                gty = gty.cuda()
            input_tensor = img

            # 每次计算相当于一次采样，计算均值，方差以及从分布中采样得到的特征，如果想要在这个地方引入特征层面的增强，可以多执行几次self.model['backbone'](input_tensor) 
            features = self.model['backbone'](input_tensor)
            out_eu_branch = self.model['EU_branch'](features)
 
            loss_EU = self.model['criterion_EU'](out_eu_branch, gty, epoch) #损失函数计算
  
            self.model['optimizer_EU'].zero_grad()
            loss_EU.backward()
            self.model['optimizer_EU'].step()
            predy   = np.argmax(out_eu_branch.data.cpu().numpy(), axis=1)  # TODO
            it_acc  = np.mean((predy == gty.data.cpu().numpy()).astype(int)) #计算训练时候的识别准确率
            batch_acc.append(it_acc)
            loss_recorder_eu.append(loss_EU.item())
            # print(loss_recorder)
            if (idx + 1) % self.args.print_freq == 0:
                print('epoch : %2d|%2d, iter : %4d|%4d,  loss EU : %.4f, batch_ave_acc : %.4f' % \
                      (epoch, self.args.end_epoch, idx+1, len(self.data['train']), np.mean(loss_recorder_eu), np.mean(batch_acc)))
        train_loss_eu = np.mean(loss_recorder_eu)
        print('train_loss EU : %.4f' % (train_loss_eu))

        return train_loss_eu

    def _test_one_epoch(self):
        #计算训练集上面的准确率
        self.model['backbone'].eval()
        self.model['EU_branch'].eval()
        cor_pre_au = 0
        cor_pre_eu = 0
        totoal_sample=0
        with torch.no_grad():
            for idx, (img, gty) in enumerate(self.data['test']):

                img.requires_grad = False
                gty.requires_grad = False

                if self.use_gpu:
                    img = img.cuda()
                    gty = gty.cuda()

                features = self.model['backbone'](img)
                output_eu  = self.model['EU_branch'](features)
                predy_eu   = np.argmax(output_eu.data.cpu().numpy(), axis=1)  # TODO
                cor_pre_eu += np.sum((predy_eu == gty.data.cpu().numpy()).astype(int))
                totoal_sample += len(predy_eu)
        return cor_pre_eu/totoal_sample

    
    def _save_model(self):
        ''' save the weights during the process of training '''
        save_name_all = '%s/trained_%s_epoch_%02d.pth' % \
                    (self.args.save_to, str(self.args.kl_lambda)+'_'+self.args.loss_mode_EU, self.result['epoch'])
        torch.save({
            'AU_optimizer': self.saved_model['optimizer'],
            'AU_scheduler': self.saved_model['scheduler'],
            'backbone': self.saved_model['backbone'],
            'AU_branch': self.saved_model['AU_branch'],
            'EU_optimizer': self.model['optimizer_EU'].state_dict(),
            'EU_scheduler': self.model['scheduler_EU'].state_dict(),
            'EU_branch': self.model['EU_branch'].state_dict()}, save_name_all)
       
    def _dul_training(self):
        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):

            start_time = time.time()
            self.result['epoch'] = epoch
            train_loss = self._train_one_epoch(epoch)
            self.model['scheduler_EU'].step()
            test_acc_eu= self._test_one_epoch()
            print('test_acc_eu: {}'.format(test_acc_eu))
            end_time = time.time()
            print('Single epoch cost time : %.2f mins' % ((end_time - start_time)/60))

            if self.args.is_debug:
                break

    def train_runner(self):
        self._report_settings()

        self._model_loader()

        self._data_loader()

        self._dul_training()

        self._save_model()


if __name__ == "__main__":
    args = cls_args()
    setup_seed(args.seed)
    print(vars(args))
    
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)
    dul_cls = DulClsTrainer(args)
    dul_cls.train_runner()