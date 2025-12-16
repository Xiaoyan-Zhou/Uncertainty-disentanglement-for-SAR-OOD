
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self, args):
        super(ClassificationHead, self).__init__()
        self.used_as = args.used_as
        # 定义一个全连接层，将输入维度映射到类别数num_classes
        if self.used_as == 'backbone':
            self.fc1 = nn.Linear(args.in_feats*7*7, 256)
        else:
            self.fc1 = nn.Linear(args.in_feats, 256)  # 第一个全连接层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(256, args.classnum)  # 第二个全连接层，输出类别数
        
        
        # # 可选：可以添加Dropout层来防止过拟合
        # self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        if self.used_as == 'backbone':
            x = x.view(x.size(0), -1)
        x = self.fc1(x)  # 通过第一个全连接层
        x = self.relu(x)  # 激活
        # x = self.dropout(x)  # 可选：Dropout
        x = self.fc2(x)  # 通过第二个全连接层，得到分类输出
        return x