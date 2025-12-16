#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython import embed
from model.evidential_losses import edl_mse_loss

def one_hot_embedding(labels, num_classes=10,device = 'cuda'):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes).cuda()
    return y[labels].cuda()

class ClsLoss(nn.Module):
    ''' Classic loss function for face recognition '''

    def __init__(self, args, branch='EU'):

        super(ClsLoss, self).__init__()
        self.args     = args
        self.branch = branch
        # experiments20250819_seed annealing_step=10
        # experiments20250616_seed annealing_step=50
        # models_trained_final annealing_step=50
        # models_trained_final_tau10 annealing_step=10
        self.annealing_step=50


    def forward(self, predy, target, epoch_num, mu = None, logvar = None):


        loss = None
        #这里的分类损失函数采用evidential uncertainty 使得模型能够估计模型不确定性
        if (self.args.loss_mode_EU == 'EDL') and (self.branch in ['EU', 'one']):
            target_one_hot = one_hot_embedding(target)
            loss = edl_mse_loss(predy, target_one_hot.float(), epoch_num, num_classes=10, annealing_step=self.annealing_step, device='cuda', KL_flag = True, loglikelihood_var_flag = True) # dawn
        elif (self.args.loss_mode_EU == 'UMSE') and (self.branch in ['EU', 'one']):
            target_one_hot = one_hot_embedding(target)
            loss = edl_mse_loss(predy, target_one_hot.float(), epoch_num, num_classes=10, annealing_step=self.annealing_step, device='cuda', KL_flag = False, loglikelihood_var_flag = True) # dawn
        elif (self.args.loss_mode_EU == 'Re_EDL') and (self.branch in ['EU', 'one']):
            target_one_hot = one_hot_embedding(target)
            loss = edl_mse_loss(predy, target_one_hot.float(), epoch_num, num_classes=10, annealing_step=self.annealing_step, device='cuda', KL_flag = False, loglikelihood_var_flag = False) # dawn
        if (self.args.loss_mode_AU == 'ce') and (self.branch == 'AU'): 
            loss = F.cross_entropy(predy, target)
            # print('use cross entropy')
        elif (self.args.loss_mode_AU == 'Re_EDL') and (self.branch == 'AU'):
            target_one_hot = one_hot_embedding(target)
            loss = edl_mse_loss(predy, target_one_hot.float(), epoch_num=0, num_classes=10, annealing_step=self.annealing_step, device='cuda', KL_flag = False, loglikelihood_var_flag = False) # dawn

        if (mu is not None) and (logvar is not None):
            kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2
            kl_loss = kl_loss.sum(dim=1).mean()
            loss    = loss + self.args.kl_lambda * kl_loss
        return loss


class RegLoss(nn.Module):

    def __init__(self, feat_dim = 512, classnum = 85742):
        super(RegLoss, self).__init__()
        self.feat_dim = feat_dim
        self.classnum = classnum
        self.centers  = torch.Tensor(classnum, feat_dim)
        
        
    def fetch_center_from_fc_layer(self, fc_state_dict):
        weights_key = 'module.weight' if 'module.weight' in fc_state_dict.keys() else 'weight'
        try:
            weights = fc_state_dict[weights_key]
        except Exception as e:
            print(e)
        else:
            assert weights.size() == torch.Size([self.classnum, self.feat_dim]), \
                'weights.size can not match with (classnum, feat_dim)'
            self.centers = weights
            print('Fetch the center from fc-layer was finished ...')

            
    def forward(self, mu, logvar, labels):
        fit_loss = (self.fc_weights[labels] - mu).pow(2) / (1e-10 + torch.exp(logvar))
        reg_loss = (fit_loss + logvar) / 2.0
        reg_loss = torch.sum(reg_loss, dim=1).mean()
        return reg_loss
