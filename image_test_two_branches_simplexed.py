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

def dataset_test(model, test_loader, use_gpu=True, branch='AU', test_time = 10):
    try:
        model['backbone'].eval()
        model['backbone'] = model['backbone'].cuda()
        if branch == 'AU':
            model['AU_branch'].eval()
            model['AU_branch'] = model['AU_branch'].cuda()
        else:
            model['EU_branch'].eval()
            model['EU_branch'] = model['EU_branch'].cuda()
        
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

            features = model['backbone'](img)
            if branch == 'AU':
                for T in range(test_time):
                    _,_,output = model['AU_branch'](features)
                    predy = np.argmax(output.data.cpu().numpy(), axis=1)
                    cor_pre += np.sum((predy == gty.data.cpu().numpy()).astype(int))
                    totoal_sample += len(predy)
            else:
                output = model['EU_branch'](features)
                predy = np.argmax(output.data.cpu().numpy(), axis=1)  # TODO
                cor_pre += np.sum((predy == gty.data.cpu().numpy()).astype(int))
                totoal_sample += len(predy)
   
    return cor_pre / totoal_sample

# aleatoric uncertainty estimation using entropy
def compute_aleatoric_uncertainty(model, test_loader, sample_time=10, num_classes=10, use_gpu=True, all_flag = True, misclas=False):
    model['backbone'].eval()
    model['AU_branch'].eval()
    model['backbone'] = model['backbone'].cuda()
    model['AU_branch'] = model['AU_branch'].cuda()
    cor_pre = 0
    totoal_sample = 0
    aleatoric_uncertainty_list = []
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
            for T in range(sample_time):
                embedding = model['backbone'](img)
                _, _, output = model['AU_branch'](embedding)
                predp = F.softmax(output, dim=1)
                predictions.append(predp)
            # 将所有预测概率堆叠成张量，形状为 [sample_time, batch_size, num_classes]
            predictions = torch.stack(predictions)
            predy = predictions.mean(dim=0).argmax(dim=1)
            # 计算每次采样的熵，形状为 [sample_time, batch_size]
            entropies = -torch.sum(predictions * torch.log(predictions + 1e-7), dim=2)
            # 计算数据不确定性（Aleatoric Uncertainty），对熵取平均
            aleatoric_uncertainty = torch.mean(entropies, dim=0)  # 形状为 [batch_size]
            if all_flag:
                aleatoric_uncertainty_list.extend(aleatoric_uncertainty.cpu().numpy())
            else:
                aleatoric_uncertainty_list.extend(aleatoric_uncertainty[predy != gty].cpu().numpy())
            if misclas:
                cor_aleatoric.extend(aleatoric_uncertainty[predy == gty].cpu().numpy())
                wrn_aleatoric.extend(aleatoric_uncertainty[predy != gty].cpu().numpy())
    if misclas:
        return cor_aleatoric, wrn_aleatoric
    else:
        return aleatoric_uncertainty_list
            
def relu_evidence(y):
    return F.relu(y)

# epistemic uncertainty estimation based on evidence vacuity
def compute_epistemic_uncertainty(model, test_loader, num_classes=10, use_gpu=True):
    model['backbone'].eval()
    model['EU_branch'].eval()
    model['backbone'] = model['backbone'].cuda()
    model['EU_branch'] = model['EU_branch'].cuda()
    cor_pre = 0
    totoal_sample = 0
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
            evidence_mean = torch.mean(np_evidence, dim=0)  
            alpha = evidence_mean + 1
            # epistemic uncertainty u=K/S
            epistemic_uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True) 
    
            epistemic_uncertainty_list.extend(epistemic_uncertainty.cpu().numpy())
            epistemic_uncertainty_list1 = [arr.item() for arr in epistemic_uncertainty_list]
    return epistemic_uncertainty_list1

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
    model_path = args.model_path_dual_branch
    model_name = (model_path.split('/')[-1]).split('.pt')[0]
    name = 'ours_' + model_name

    save_root = args.save_to
    print('model_path', model_path)

    # the name of dataset which is the same as legend
    Aleaoric_label_list = ['MSTAR(clean)', '-0.001','-0.003', '-0.005', '-0.007', 'lowres0.5', 'lowres1.0', 'speckle0.7', 'speckle0.8', 'speckle0.9', 'speckle1.0']
    # dataset list for test the ability of aleatoric uncertainty estimation
    Aleaoric_DATASET_DIR_LIST = ['/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/MSTAR/SOC/TEST', '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/defocus/-0.001/MSTAR/SOC/TEST', 
    '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/defocus/-0.003/MSTAR/SOC/TEST',
    '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/defocus/-0.005/MSTAR/SOC/TEST', '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/defocus/-0.007/MSTAR/SOC/TEST',
    '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/lowres/0.5/MSTAR/SOC/TEST', '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/lowres/1/MSTAR/SOC/TEST',
    '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/speckle/0.7/MSTAR/SOC/TEST', '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/speckle/0.8/MSTAR/SOC/TEST',
    '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/speckle/0.9/MSTAR/SOC/TEST', '/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/speckle/1.0/MSTAR/SOC/TEST']
    

    Epistemic_label_list = ['MSTAR(ID)', 'SAMPLE(OOD)', 'FUSAR-ship(OOD)', 'SAR-ACD(OOD)']
    # dataset list for test the ability of epistemic uncertainty estimation
    Epistemic_DATASET_DIR_LIST = ['/scratch/project_2002243/zhouxiaoyan/SAR-DG/data/MSTAR/SOC/TEST', 
    '/scratch/project_2002243/zhouxiaoyan/SAR-OOD/SAMPLE', 
    '/scratch/project_2002243/zhouxiaoyan/SAR-OOD/SHIP/FUSAR-ship', 
    '/scratch/project_2002243/zhouxiaoyan/SAR-OOD/AIRPLANE/SAR-ACD-main']

    # label_list = ['train_data','lowresolution']
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    model_resnet = resnet18(pretrained=False)
    model['backbone']  =  torch.nn.Sequential(*list(model_resnet.children())[:-2]) # Excludes the final FC layer

    model['EU_branch'] = mlib.EU_branch(args)
    model['AU_branch'] = mlib.AU_branch(args)

    saved_model = torch.load(model_path)
    model['backbone'].load_state_dict(saved_model['backbone'])
    model['AU_branch'].load_state_dict(saved_model['AU_branch'])
    model['EU_branch'].load_state_dict(saved_model['EU_branch'])
    
    batch_size = 256
  
    aleatoric_uncertainty_list = []
    cor_aleatoric_list = []
    epistemic_uncertainty_list = []
    wrn_aleatoric_list = []
    dict_acc_AU = {}
    dict_acc_EU = {}
    for DATASET_DIR in Aleaoric_DATASET_DIR_LIST:
        test_set = SAR_DatasetLoader('test', DATASET_DIR)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        # if DATASET_DIR == Aleaoric_DATASET_DIR_LIST[0]: # Aleaoric_DATASET_DIR_LIST[0] is the raw data path
        #     cor_aleatoric, wrn_aleatoric = compute_aleatoric_uncertainty(model, test_loader, sample_time=10, misclas=True) # dawn; sample_time
        # else:
        cor_aleatoric, wrn_aleatoric = compute_aleatoric_uncertainty(model, test_loader, sample_time=10, all_flag=False, misclas=True) # dawn; sample_time
        # aleatoric_uncertainty_list.append(aleatoric_uncertainty)
        wrn_aleatoric_list.append(wrn_aleatoric)
        cor_aleatoric_list.append(cor_aleatoric)
        acc_AU = dataset_test(model, test_loader, test_time=10)
        acc_EU = dataset_test(model, test_loader, branch = 'EU', test_time=10)
        print("DATASET_DIR", DATASET_DIR)
        print('AU branch acc {:.2f}, EU branch acc {:.2f}'.format(acc_AU*100, acc_EU*100))
        dict_acc_AU[DATASET_DIR] = acc_AU
        dict_acc_EU[DATASET_DIR] = acc_EU
    # draw aleatoric distribution and save
    # plot_distribution(aleatoric_uncertainty_list[:5], Aleaoric_label_list[:5], savepath=os.path.join(save_root, name+'aleatoric_result1.png'))
    # plot_distribution([aleatoric_uncertainty_list[0]]+ aleatoric_uncertainty_list[5:7], [Aleaoric_label_list[0]]+ Aleaoric_label_list[5:7], savepath=os.path.join(save_root, name+'aleatoric_result2.png'))
    # plot_distribution([aleatoric_uncertainty_list[0]]+ aleatoric_uncertainty_list[7:], [Aleaoric_label_list[0]]+ Aleaoric_label_list[7:], savepath=os.path.join(save_root, name+'aleatoric_result3.png'))

    # use OOD meteric to evaluate the performance of misclassification; misclassification detection
    results_au = {}
    for i in range(len(Aleaoric_DATASET_DIR_LIST)):

        results_au[Aleaoric_label_list[i]] = metrics.cal_metric(list(map(lambda x: -x, cor_aleatoric_list[i])), list(map(lambda x: -x, wrn_aleatoric_list[i])))
        print(Aleaoric_label_list[i], results_au[Aleaoric_label_list[i]])

    batch_size = 1
    for DATASET_DIR in Epistemic_DATASET_DIR_LIST:
        if DATASET_DIR == Epistemic_DATASET_DIR_LIST[0]:
            test_set = SAR_DatasetLoader('test', DATASET_DIR)
        else:
            test_set = SAR_DatasetLoader('OOD', DATASET_DIR)

        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        
        epistemic_uncertainty = compute_epistemic_uncertainty(model, test_loader)# dawn
        epistemic_uncertainty_list.append(epistemic_uncertainty)
    
    # draw epistemic distribution and save
    plot_distribution(epistemic_uncertainty_list, Epistemic_label_list, savepath=os.path.join(save_root, name+'epistemic.png'))
    results_eu = {}
    for i in range(1, len(Epistemic_DATASET_DIR_LIST)):
        results_eu[Epistemic_label_list[i]] = metrics.cal_metric(list(map(lambda x: -x, epistemic_uncertainty_list[0])), list(map(lambda x: -x, epistemic_uncertainty_list[i])))
        print(Epistemic_label_list[i], results_eu[Epistemic_label_list[i]])
    

     # Convert the dictionary to a DataFrame and then transpose it
    df_transposed_acc_AU = pd.DataFrame(list(dict_acc_AU.items()), columns=['Data Path', 'Accuracy'])
    df_transposed_acc_EU = pd.DataFrame(list(dict_acc_EU.items()), columns=['Data Path', 'Accuracy'])

    df_transposed_au = pd.DataFrame(results_au).T
    df_transposed_eu = pd.DataFrame(results_eu).T
    
    with pd.ExcelWriter(args.save_excel_two_branch) as writer:
        df_transposed_acc_AU.to_excel(writer, index=True, header=['Data Path', 'Accuracy'], sheet_name='AU accuracy_dataset') ## performance of OOD generalization
        df_transposed_acc_EU.to_excel(writer,index=True, header=['Data Path', 'Accuracy'], sheet_name='EU accuracy_dataset') ## performance of OOD generalization
        df_transposed_au.to_excel(writer, index=True, header=['AUROC','AUPR', 'FPR95'], sheet_name='aleatoric uncertainty') ### use OOD detection metrics to evaluate aleatoric uncertainty estimation (not used in paper)
        df_transposed_eu.to_excel(writer, index=True, header=['AUROC','AUPR', 'FPR95'], sheet_name='epistemic uncertainty') ### epistemic uncertainty in donwnstream task OOD detection 
