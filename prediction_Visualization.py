import torch
import numpy as np
from torch.utils.data import DataLoader
from data_utils import SingleImageDataset
import model as mlib
from train_cls_config_simplex import cls_args
from torchvision import transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
import seaborn as sns
from matplotlib import pyplot as plt
import os
import statistics
from evidential_decomposition import getDisn, cal_uncertainty
import metrics
import pandas as pd
from torchvision.models import resnet18
from draw_figs.draw_calibrated_prediction import draw_calibrated

def image_test_AU(model, test_loader, sample_time=10, num_classes=10, use_gpu=True):
    model['backbone'].eval()
    model['AU_branch'].eval()
    model['backbone'] = model['backbone'].cuda()
    model['AU_branch'] = model['AU_branch'].cuda()

    with torch.no_grad():
        for idx, (img, gty) in enumerate(test_loader):
            img.requires_grad = False
            gty.requires_grad = False

            if use_gpu:
                img = img.cuda()
                gty = gty.cuda()
            predictions = []
            for T in range(sample_time):
                embedding = model['backbone'](img)
                _, _, output = model['AU_branch'](embedding)
                predp = F.softmax(output, dim=1)
                predictions.append(predp)
            # Stack all predicted probabilities into a tensor with shape: [sample_time, batch_size, num_classes]
            predictions = torch.stack(predictions)
            calibrated_pre = predictions.mean(dim=0)
            predy = predictions.mean(dim=0).argmax(dim=1)
            # Calculate the entropy of each sample, the shape is [sample_time, batch_size]
            entropies = -torch.sum(predictions * torch.log(predictions + 1e-7), dim=2)
            # Calculate data uncertainty (Aleatoric Uncertainty), average entropy
            aleatoric_uncertainty = torch.mean(entropies, dim=0)  # shape [batch_size]
    return aleatoric_uncertainty, predy, gty, calibrated_pre
            

def image_test_EU(model, test_loader, num_classes=10, use_gpu=True):
    model['backbone'].eval()
    model['EU_branch'].eval()
    model['backbone'] = model['backbone'].cuda()
    model['EU_branch'] = model['EU_branch'].cuda()
    epistemic_uncertainty_list = []
    with torch.no_grad():
        for idx, (img, gty) in enumerate(test_loader):

            img.requires_grad = False
            gty.requires_grad = False

            if use_gpu:
                img = img.cuda()
                gty = gty.cuda()
            levidence = [] # list for evidence
            
            embedding = model['backbone'](img)
            output = model['EU_branch'](embedding)
            evidence = relu_evidence(output)
            levidence.append(evidence)
            np_evidence = torch.stack(levidence)
            evidence_mean = torch.mean(np_evidence, dim=0)  # 形状为 [batch_size, num_classes]
            alpha = evidence_mean + 1
            epistemic_uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True) # model uncertainty u=K/S
            epistemic_uncertainty_list.extend(epistemic_uncertainty.cpu().numpy())
            epistemic_uncertainty_list1 = [arr.item() for arr in epistemic_uncertainty_list]
            calibrated_pre = alpha / torch.sum(alpha, dim=1, keepdim=True)
    return epistemic_uncertainty_list1, calibrated_pre

           
def relu_evidence(y):
    return F.relu(y)


def decompose_uncertainty(model, test_loader, uncertainty_mode, num_classes=10, use_gpu=True):
    model['backbone'].eval()
    model['EU'].eval()
    model['backbone'] = model['backbone'].cuda()
    model['EU'] = model['EU'].cuda()
    aleatoric_uncertainty_list = []
    epistemic_uncertainty_list = []
    cor_aleatoric = []
    wrn_aleatoric = []
    with torch.no_grad():
        for idx, (img, gty) in enumerate(test_loader):
            img.requires_grad = False
            gty.requires_grad = False
            if use_gpu:
                img = img.cuda()
                gty = gty.cuda()
            predictions = []
            embedding = model['backbone'](img)
            output = model['EU'](embedding)
            predy = output.argmax(dim=1)
            evidence = relu_evidence(output)
            lalpha = evidence + 1
            AU, EU = cal_uncertainty(lalpha, uncertainty_mode)
            calibrated_pre = lalpha / lalpha.sum(dim=1, keepdim=True)
    return AU, EU, calibrated_pre



def img_calibration_visualization(img_path, save_fig_path, OOD_val=False):
    model = dict()
    args = cls_args()
    model_path_dual = args.model_path_dual_branch
    print('model_path_dual', model_path_dual)

    # load image
    test_img = SingleImageDataset(img_path)
    test_loader = DataLoader(dataset=test_img, batch_size=1, shuffle=False, num_workers=0)

    ## load dual-branch model weights
    model_resnet = resnet18(pretrained=False)
    model['backbone']  =  torch.nn.Sequential(*list(model_resnet.children())[:-2]) # Excludes the final FC layer

    model['EU_branch'] = mlib.EU_branch(args)
    model['AU_branch'] = mlib.AU_branch(args)

    saved_model = torch.load(model_path_dual)
    model['backbone'].load_state_dict(saved_model['backbone'])
    model['AU_branch'].load_state_dict(saved_model['AU_branch'])
    model['EU_branch'].load_state_dict(saved_model['EU_branch'])
    
    AU, predy, label, calibrated_pre_AU = image_test_AU(model, test_loader, sample_time=10, num_classes=10, use_gpu=True)
    EU, calibrated_pre_EU = image_test_EU(model, test_loader, num_classes=10, use_gpu=True)

    # load one-branch model
    print('#'*10, "evaluation with single branch")
    model_path_single = args.model_path_one_branch
    uncertainty_mode = 'evidence'

    model_resnet = resnet18(weights=None)
    model['backbone']  =  torch.nn.Sequential(*list(model_resnet.children())[:-2]) 
    model['EU']  = mlib.EU_branch(args) 
    saved_model = torch.load(model_path_single)
    model['backbone'].load_state_dict(saved_model['backbone'])
    model['EU'].load_state_dict(saved_model['EU'])
    for uncertainty_mode in ['variance', 'entropy', 'evidence']:
        AU, EU, calibrated_pre = decompose_uncertainty(model, test_loader, uncertainty_mode, num_classes=10, use_gpu=True)
    if OOD_val:
        draw_calibrated(calibrated_pre_EU.squeeze().cpu(), calibrated_pre.squeeze().cpu(), save_fig_path)
    else:
        draw_calibrated(calibrated_pre_AU.squeeze().cpu(), calibrated_pre.squeeze().cpu(), save_fig_path)

if __name__ == '__main__':
    # img_dir = 'draw_figs/calibration_examples/OOD_Generalization'
    # fig_dir = 'draw_figs/calibration_results/OOD_Generalization'
    # if not os.path.exists(fig_dir):
    #     os.makedirs(fig_dir)
    # img_list = os.listdir(img_dir)
    # for img in img_list:
    #     img_path = os.path.join(img_dir, img)
    #     print('img_path', img_path)
    #     save_fig_path = os.path.join(fig_dir, img)
    #     img_calibration_visualization(img_path, save_fig_path, OOD_val=False)
    
    img_dir = 'draw_figs/calibration_examples/OOD_Detection'
    fig_dir = 'draw_figs/calibration_results/OOD_Detection'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    img_list = os.listdir(img_dir)
    for img in img_list:
        img_path = os.path.join(img_dir, img)
        print('img_path', img_path)
        save_fig_path = os.path.join(fig_dir, img)
        img_calibration_visualization(img_path, save_fig_path, OOD_val=True)

