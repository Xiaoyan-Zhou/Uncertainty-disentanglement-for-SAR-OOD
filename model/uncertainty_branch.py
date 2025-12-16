### use one branch to estimation epistemic uncertainty
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.classifer import ClassificationHead


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CosineSoftmax(nn.Module):
    def __init__(self, args):
        super(CosineSoftmax, self).__init__()
        self.scale = args.scale
        self.weight = nn.Parameter(torch.Tensor(args.classnum, args.in_feats))
        nn.init.xavier_uniform_(self.weight)
        # self.weight = nn.Parameter(torch.FloatTensor(args.classnum, args.in_feats))
    def forward(self, x):
        # # 归一化输入特征和权重
        # x_norm = F.normalize(x, p=2, dim=1)
        # weight_norm = F.normalize(self.weight, p=2, dim=1)

        # # 计算余弦相似度
        # cos_theta = F.linear(x_norm, weight_norm).clamp(-1, 1)
        cos_theta  = F.linear(F.normalize(x), F.normalize(self.weight)).clamp(-1, 1)
        # 缩放余弦相似度
        cos_theta *= self.scale

        return cos_theta  # 返回logits以与CrossEntropyLoss一起使用

class EU_branch_V2(nn.Module):
    def __init__(self, args):
        super(EU_branch_V2, self).__init__()

        self.f1 = nn.Sequential(
        Flatten(),
        nn.Linear(512 * 1 * 1 * 1, 512))

        # 定义一个全连接层，将输入维度映射到类别数num_classes
        # self.fc1 = nn.Linear(args.in_feats*7*7, 512)
        self.fc2 = nn.Linear(512, 256)  # 第二个全连接层，输出类别数
        self.relu = nn.ReLU()  # 激活函数
        self.fc3 = nn.Linear(256, args.classnum)  # 第二个全连接层，输出类别数
        
    def forward(self, x):
        x = self.f1(x)  # 
        x = self.relu(x)  # 激活
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)  # 通过第二个全连接层，得到分类输出
        return x

class EU_branch(nn.Module):
    def __init__(self, args):
        super(EU_branch, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(output_size=(6,6))

        self.f1 = nn.Sequential(
        nn.BatchNorm2d(512 * 1, eps=2e-5, affine=False),
        nn.Dropout(p=args.drop_ratio),
        Flatten(),
        nn.Linear(512 * 1 * 6 * 6, 512),
        nn.BatchNorm1d(512, eps=2e-5))

        # 定义一个全连接层，将输入维度映射到类别数num_classes
        # self.fc1 = nn.Linear(args.in_feats*7*7, 512)
        self.fc2 = nn.Linear(512, 256)  # 第二个全连接层，输出类别数
        self.relu = nn.ReLU()  # 激活函数
        self.fc3 = nn.Linear(256, args.classnum)  # 第二个全连接层，输出类别数
        
    def forward(self, x):
        x = self.pool(x)
        x = self.f1(x)  # 
        x = self.relu(x)  # 激活
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)  # 通过第二个全连接层，得到分类输出
        return x

class AU_branch(nn.Module):
    def __init__(self, args):
        super(AU_branch, self).__init__()
        feat_dim = 512
        # define mu head

        self.pool = nn.AdaptiveAvgPool2d(output_size=(6,6))
        self.mu_head = nn.Sequential(
        nn.BatchNorm2d(512 * 1, eps=2e-5, affine=False),
        nn.Dropout(p=args.drop_ratio),
        Flatten(),
        nn.Linear(512 * 1 * 6 * 6, feat_dim),
        nn.BatchNorm1d(feat_dim, eps=2e-5))

        # define logvar head
        self.logvar_head = nn.Sequential(
        nn.BatchNorm2d(512 * 1, eps=2e-5, affine=False),
        nn.Dropout(p=args.drop_ratio),
        Flatten(),
        nn.Linear(512 * 1 * 6 * 6, feat_dim),
        nn.BatchNorm1d(feat_dim, eps=2e-5))



        self.fc = CosineSoftmax(args)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std
    
    def forward(self, x): # x is feature which obtained by backbone
        x = self.pool(x)
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        x = self._reparameterize(mu, logvar)
        logit = self.fc(x)
        return (mu, logvar, logit)
