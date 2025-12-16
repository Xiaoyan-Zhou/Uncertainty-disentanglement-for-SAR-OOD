#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### train model considering the data uncertainty, using data uncertianty as a control factor for label smoothing
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
    # torch.backends.cudnn.enabled = False
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
        model_resnet = resnet18(pretrained=False)
        self.model['backbone']  =  torch.nn.Sequential(*list(model_resnet.children())[:-2]) # Excludes the final FC layer
    
        self.model['EU']  = mlib.EU_branch(self.args) #dawn 我们使用两层的MLP作为分类头
        self.model['criterion'] = mlib.ClsLoss(self.args, branch='one') # dawn 定义损失函数，loss_mode='evidential'对应我们的方法
        self.model['optimizer'] = torch.optim.SGD(
                                    [{'params': self.model['backbone'].parameters()},
                                    {'params': self.model['EU'].parameters()}],
                                    lr=self.args.base_lr,
                                    dampening=0,
                                    weight_decay=self.args.weight_decay,
                                    momentum=0.9,
                                    nesterov=False) #定义优化器
        self.model['scheduler'] = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model['optimizer'], float(self.args.end_epoch))
        if self.use_gpu:
            self.model['backbone']  = self.model['backbone'].cuda()
            self.model['EU']  = self.model['EU'].cuda()
            self.model['criterion'] = self.model['criterion'].cuda() #setting of loss function for model parameter

        if self.use_gpu and len(self.args.gpu_ids) > 1:
            self.model['backbone'] = torch.nn.DataParallel(self.model['backbone'], device_ids=self.args.gpu_ids)
            self.model['EU'] = torch.nn.DataParallel(self.model['EU'], device_ids=self.args.gpu_ids)
            print('Parallel mode was going ...')
        elif self.use_gpu:
            print('Single-gpu mode was going ...')
        else:
            print('CPU mode was going ...')

        if len(self.args.resume) > 2:
            checkpoint = torch.load(self.args.resume, map_location=lambda storage, loc: storage)
            self.args.start_epoch = checkpoint['epoch']
            self.model['backbone'].load_state_dict(checkpoint['backbone'])
            self.model['EU'].load_state_dict(checkpoint['EU'])
            print('Resuming the train process at %3d epoches ...' % self.args.start_epoch)
        print('Model loading was finished ...')
    
    def _data_loader(self):
        
        train_set = SAR_DatasetLoader('train', self.args.data_train_path)
        test_set = SAR_DatasetLoader('test', self.args.data_test_dir)
        self.data['train'] = DataLoader(dataset=train_set, batch_size=self.args.batch_size, shuffle=True, num_workers=0) # 丢弃最后一个batch , drop_last=True
        self.data['test'] = DataLoader(dataset=test_set, batch_size=self.args.batch_size, shuffle=True, num_workers=0)  # 丢弃最后一个batch , drop_last=True

        print('Data loading was finished ...')

    def _train_one_epoch(self, epoch = 0, sample_time=2, num_classes=10):

        self.model['backbone'].train()
        self.model['EU'].train()
        predictions = []

        loss_recorder, batch_acc = [], []
        for idx, (img, gty) in enumerate(self.data['train']):

            img.requires_grad = False
            gty.requires_grad = False

            if self.use_gpu:
                img = img.cuda()
                gty = gty.cuda()
            input_tensor = img

            # 每次计算相当于一次采样，计算均值，方差以及从分布中采样得到的特征，如果想要在这个地方引入特征层面的增强，可以多执行几次self.model['backbone'](input_tensor) 
            embedding = self.model['backbone'](input_tensor)
            output  = self.model['EU'](embedding) # 分类
 
            loss    = self.model['criterion'](output, gty, epoch) #损失函数计算
            self.model['optimizer'].zero_grad()
            loss.backward()
            self.model['optimizer'].step()
            predy   = np.argmax(output.data.cpu().numpy(), axis=1)  # TODO
            it_acc  = np.mean((predy == gty.data.cpu().numpy()).astype(int)) #计算训练时候的识别准确率
            batch_acc.append(it_acc)
            loss_recorder.append(loss.item())
            # print(loss_recorder)
            if (idx + 1) % self.args.print_freq == 0:
                print('epoch : %2d|%2d, iter : %4d|%4d,  loss : %.4f, batch_ave_acc : %.4f' % \
                      (epoch, self.args.end_epoch, idx+1, len(self.data['train']), np.mean(loss_recorder), np.mean(batch_acc)))
        train_loss = np.mean(loss_recorder)
        print('train_loss : %.4f' % train_loss)

        return train_loss

    def _test_one_epoch(self):
        #计算训练集上面的准确率
        self.model['backbone'].eval()
        self.model['EU'].eval()
        cor_pre = 0
        totoal_sample=0
        with torch.no_grad():
            for idx, (img, gty) in enumerate(self.data['test']):

                img.requires_grad = False
                gty.requires_grad = False

                if self.use_gpu:
                    img = img.cuda()
                    gty = gty.cuda()

                embedding = self.model['backbone'](img)
                output  = self.model['EU'](embedding)
                predy   = np.argmax(output.data.cpu().numpy(), axis=1)  # TODO
                cor_pre += np.sum((predy == gty.data.cpu().numpy()).astype(int))
                totoal_sample += len(predy)
        return cor_pre/totoal_sample

    
    def _save_model(self):
        ''' save the weights during the process of training '''
        save_name = '%s/one_branch_%s_epoch_%02d.pth' % \
                    (self.args.save_to, self.args.loss_mode_EU, self.result['epoch'])
        if not os.path.exists(self.args.save_to):
            os.mkdir(self.args.save_to)
        torch.save({
            'optimizer': self.model['optimizer'].state_dict(),
            'scheduler': self.model['scheduler'].state_dict(),
            'backbone': self.model['backbone'].state_dict(),
            'EU': self.model['EU'].state_dict(),}, save_name)

            
    def _dul_training(self):

        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):

            start_time = time.time()
            self.result['epoch'] = epoch
            train_loss = self._train_one_epoch(epoch)
            self.model['scheduler'].step()
            test_acc = self._test_one_epoch()
            print('test_acc', test_acc)
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