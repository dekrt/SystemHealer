import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import pdb
import datetime
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
import json
import skdim
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import zscore
from hyperopt import hp, fmin, tpe, Trials
from torch.utils.data import DataLoader, TensorDataset



# Convtran model
class ConvTranBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvTranBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ConvTran(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ConvTran, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        features = F.adaptive_avg_pool1d(out, 1)
        out = features.view(out.size(0), -1)
        out = self.linear(out)
        if return_features:
            return features.squeeze(2).detach().cpu().numpy()
        else:
            return out


# 定义训练函数
def train(model, train_loader, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        model.train()
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# 定义计算验证集上的损失的函数
def compute_val_loss(model, val_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(val_loader)

# 定义超参数空间
space = {
    'lr': hp.loguniform('lr', -5, -2),
    'weight_decay': hp.uniform('weight_decay', 0, 0.1),
    'num_blocks1': hp.choice('num_blocks1', [1, 2, 3, 4]),
    'num_blocks2': hp.choice('num_blocks2', [1, 2, 3, 4]),
    'num_blocks3': hp.choice('num_blocks3', [1, 2, 3, 4]),
    'num_blocks4': hp.choice('num_blocks4', [1, 2, 3, 4])
}

# 定义目标函数
def objective(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvTran(ConvTranBlock, [params['num_blocks1'], params['num_blocks2'], params['num_blocks3'], params['num_blocks4']], 6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    data = np.loadtxt("./data_filled.csv", delimiter=',', skiprows=1)

    y = data[:, -1]
    X = data[:, 1:-1]

    train_data, val_data, train_label, val_label = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)

    # sampler = SMOTE()
    # train_data, train_label = sampler.fit_resample(train_data, train_label)

    train_data = torch.Tensor(train_data)
    val_data = torch.Tensor(val_data)
    train_label = torch.LongTensor(train_label)
    val_label = torch.LongTensor(val_label)
    # pdb.set_trace()

    # 创建数据集
    train_dataset = TensorDataset(train_data, train_label)
    val_dataset = TensorDataset(val_data, val_label)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 调用训练函数进行训练
    train(model, train_loader, criterion, optimizer, 30)

    # 计算验证集上的损失
    val_loss = compute_val_loss(model, val_loader, criterion)

    return val_loss

# 使用贝叶斯优化来找到最佳的超参数
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

