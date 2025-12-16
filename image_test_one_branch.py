import torch
import numpy as np
from torch.utils.data import DataLoader
from data_utils import SAR_DatasetLoader
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

def dataset_test(model, test_loader, use_gpu=True):
    try:
        model['backbone'].eval()
        model['EU'].eval()
        model['backbone'] = model['backbone'].cuda()
        model['EU'] = model['EU'].cuda()
        
    except:
        model = model.cuda()

    cor_pre = 0
    totoal_sample = 0
    with torch.no_grad():
        for idx, (img, gty) in enumerate(test_loader):

            img.requires_grad = False
            gty.requires_grad = False

            if use_gpu:
                img = img.cuda()
                gty = gty.cuda()

            try:
                features = model['backbone'](img)
                output = model['EU'](features) #dawn
            except:
                output = model(img)
            predy = np.argmax(output.data.cpu().numpy(), axis=1)  # TODO
            cor_pre += np.sum((predy == gty.data.cpu().numpy()).astype(int))
            totoal_sample += len(predy)
   
    return cor_pre / totoal_sample


# decompose uncertainty based on paper: Evidential Uncertainty Quantification: A Variance-Based Perspective (WACV2024)
def decompose_uncertainty(model, test_loader, uncertainty_mode, num_classes=10, use_gpu=True, all_flag = True, misclas=False):
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

            if AU.shape == torch.Size([]):
                AU    = AU.reshape(-1) 
            if all_flag:  ### calculate the AU of all data
                aleatoric_uncertainty_list.extend(AU.cpu().numpy())
            else: ### calculate the AU of the wrong predictions
                aleatoric_uncertainty_list.extend(AU[predy != gty].cpu().numpy())
            if misclas:
                cor_aleatoric.extend(AU[predy == gty].cpu().numpy())
                wrn_aleatoric.extend(AU[predy != gty].cpu().numpy())
            epistemic_uncertainty_list.extend(EU.cpu().numpy())
    if misclas:
        return cor_aleatoric, wrn_aleatoric, epistemic_uncertainty_list
    else:
        return aleatoric_uncertainty_list, epistemic_uncertainty_list

def relu_evidence(y):
    return F.relu(y)

def plot_distribution(value_list, label_list, savepath='nll'):
    sns.set(style="white", palette="muted")
    # palette = ['#A8BAE3', '#55AB83']
    palette = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'gray', 'pink', 'orange', 'purple', 'brown', 'olive']

    # sns.displot({label_list[0]: value_list[0], label_list[1]:  value_list[1]}, label="id", kind = "kde", palette=palette, fill = True, alpha = 0.8)
    dict_value = {label_list[i]: value_list[i] for i in range(len(label_list))}
    sns.displot(dict_value, kind="kde", palette=palette, fill=True, alpha=0.5)
    plt.savefig(savepath, dpi=300)            

if __name__ == '__main__':
    model = dict()
    args = cls_args()
    # decompose uncertainty from different perspective
    uncertainty_mode = args.uncertainty_mode #  uncertainty_mode in ['variance', 'entropy', 'evidence']
    print('*'*20)
    print('uncertainty_mode', uncertainty_mode)
    name = uncertainty_mode + args.loss_mode_EU
    model_path = args.model_path_one_branch #dawn 
    save_root = args.save_to
    print('*'*20)
    print('model_path', model_path)


    # the name of dataset which is the same as legend
    Aleaoric_label_list = ['MSTAR(clean)', '-0.001','-0.003', '-0.005', '-0.007', 'lowres0.5', 'lowres1.0', 'speckle0.7', 'speckle0.8', 'speckle0.9', 'speckle1.0']

    # dataset list for test the ability of aleatoric uncertainty estimation
    Aleaoric_DATASET_DIR_LIST = [
    # '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/MSTAR/SOC/TEST',
    '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/MSTAR/SOC/TEST',
     '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/defocus/-0.001/MSTAR/SOC/TEST', 
    '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/defocus/-0.003/MSTAR/SOC/TEST',
    '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/defocus/-0.005/MSTAR/SOC/TEST',
     '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/defocus/-0.007/MSTAR/SOC/TEST',
    '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/lowres/0.5/MSTAR/SOC/TEST', 
    '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/lowres/1/MSTAR/SOC/TEST',
    '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/speckle/0.7/MSTAR/SOC/TEST',
     '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/speckle/0.8/MSTAR/SOC/TEST',
    '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/speckle/0.9/MSTAR/SOC/TEST',
     '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/speckle/1.0/MSTAR/SOC/TEST']

    # use out-of-distribution data to test the ability of epistemic uncertainty
    # dataset list for test the ability of epistemic uncertainty estimation
    Epistemic_label_list = ['MSTAR(ID)', 'SAMPLE(OOD)', 'FUSAR-ship(OOD)', 'SAR-ACD(OOD)']
    Epistemic_DATASET_DIR_LIST = ['/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/MSTAR/SOC/TEST', 
    '/scratch/project_2002243/zhouxiaoyan/SAR-OOD/SAMPLE', 
    '/scratch/project_2002243/zhouxiaoyan/SAR-OOD/SHIP/FUSAR-ship', 
    '/scratch/project_2002243/zhouxiaoyan/SAR-OOD/AIRPLANE/SAR-ACD-main']

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    model_resnet = resnet18(weights=None)
    model['backbone']  =  torch.nn.Sequential(*list(model_resnet.children())[:-2]) 
    model['EU']  = mlib.EU_branch(args) 
    saved_model = torch.load(model_path)
    model['backbone'].load_state_dict(saved_model['backbone'])
    model['EU'].load_state_dict(saved_model['EU'])

    batch_size = 256

    aleatoric_uncertainty_list = []
    cor_aleatoric_list=[]
    wrn_aleatoric_list = []
    epistemic_uncertainty_list = []
    acc_list = []
    dict_acc = {}

    for DATASET_DIR in Aleaoric_DATASET_DIR_LIST:
        test_set = SAR_DatasetLoader('test', DATASET_DIR)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0)

        cor_aleatoric, wrn_aleatoric, _ = decompose_uncertainty(model, test_loader, uncertainty_mode, all_flag=False, misclas=True) # decompose uncertainty learned by EDL
        cor_aleatoric_list.append(cor_aleatoric)
        wrn_aleatoric_list.append(wrn_aleatoric)
        # aleatoric_uncertainty_list.append(aleatoric_uncertainty)
        acc = dataset_test(model, test_loader)
        acc_list.append(acc*100)
        dict_acc[DATASET_DIR] = acc
        print("DATASET_DIR", DATASET_DIR)
        print('acc {:.2f}'.format(acc*100))
    # draw aleatoric distribution and save
    # plot_distribution(aleatoric_uncertainty_list[:5], Aleaoric_label_list[:5], savepath=os.path.join(save_root, name+'aleatoric_result1.png'))
    # plot_distribution([aleatoric_uncertainty_list[0]]+ aleatoric_uncertainty_list[5:7], [Aleaoric_label_list[0]]+ Aleaoric_label_list[5:7], savepath=os.path.join(save_root, name+'aleatoric_result2.png'))
    # plot_distribution([aleatoric_uncertainty_list[0]]+ aleatoric_uncertainty_list[7:], [Aleaoric_label_list[0]]+ Aleaoric_label_list[7:], savepath=os.path.join(save_root, name+'aleatoric_result3.png'))
    # use OOD meteric to evaluate the performance
    results_au = {}
    for i in range(len(Aleaoric_DATASET_DIR_LIST)):
        # results_au[Aleaoric_label_list[i]] = metrics.cal_metric(list(map(lambda x: -x, aleatoric_uncertainty_list[0])), list(map(lambda x: -x, aleatoric_uncertainty_list[i])))
        results_au[Aleaoric_label_list[i]] = metrics.cal_metric(list(map(lambda x: -x, cor_aleatoric_list[i])), list(map(lambda x: -x, wrn_aleatoric_list[i])))
        print(Aleaoric_label_list[i], results_au[Aleaoric_label_list[i]])

    batch_size = 1
    ### use OOD data to evaluate epistemic uncertainty estimation performance
    for DATASET_DIR in Epistemic_DATASET_DIR_LIST:
        if DATASET_DIR == Epistemic_DATASET_DIR_LIST[0]:
            test_set = SAR_DatasetLoader('test', DATASET_DIR)
        else:
            test_set = SAR_DatasetLoader('OOD', DATASET_DIR)
            
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        
        _, epistemic_uncertainty = decompose_uncertainty(model, test_loader, uncertainty_mode) 
        epistemic_uncertainty_list.append(epistemic_uncertainty)
        print("DATASET_DIR", DATASET_DIR)
    
    # draw epistemic distribution and save
    plot_distribution(epistemic_uncertainty_list, Epistemic_label_list, savepath=os.path.join(save_root, name+'epistemic.png'))
    results_eu = {}
    for i in range(1, len(Epistemic_DATASET_DIR_LIST)):
        results_eu[Epistemic_label_list[i]] = metrics.cal_metric(list(map(lambda x: -x, epistemic_uncertainty_list[0])), list(map(lambda x: -x, epistemic_uncertainty_list[i])))
        print(Epistemic_label_list[i], results_eu[Epistemic_label_list[i]])
    
    # Convert the dictionary to a DataFrame and then transpose it
    df_transposed_acc = pd.DataFrame(list(dict_acc.items()), columns=['Data Path', 'Accuracy'])
    df_transposed_au = pd.DataFrame(results_au).T
    df_transposed_eu = pd.DataFrame(results_eu).T
    
    with pd.ExcelWriter(args.save_excel_one_branch) as writer:
        df_transposed_acc.to_excel(writer, index=True, header=['Data Path', 'Accuracy'], sheet_name='accuracy_dataset') ## performance of OOD generalization
        df_transposed_au.to_excel(writer, index=True, header=['AUROC','AUPR', 'FPR95'], sheet_name='aleatoric uncertainty') ### use OOD detection metrics to evaluate aleatoric uncertainty estimation (not used in paper)
        df_transposed_eu.to_excel(writer, index=True, header=['AUROC','AUPR', 'FPR95'], sheet_name='epistemic uncertainty') ### epistemic uncertainty in donwnstream task OOD detection 
