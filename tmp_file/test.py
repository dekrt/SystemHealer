from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim


def train_preprocess(data):
    # 去掉重复值
    data.drop_duplicates(inplace=True)

    # 箱线图去掉异常值
    # threshold = 1.5
    # Q1 = data.quantile(0.1)
    # Q3 = data.quantile(0.9)
    # IQR = Q3 - Q1
    # data_cleaned = data[~((data < (Q1 - threshold * IQR)) | (data > (Q3 + threshold * IQR))).any(axis=1)]

    # 填充缺失值
    imputer = IterativeImputer(random_state=0)
    # data_imputed = imputer.fit_transform(data_cleaned)
    data_imputed = imputer.fit_transform(data)
    data_imputed = pd.DataFrame(data_imputed, columns=data.columns)
    data_imputed = data_imputed.dropna()
    y = data_imputed['label']
    columns_to_drop = ['label']
    data_imputed = data_imputed.drop(columns_to_drop, axis=1)

    # 标准化
    scaler = StandardScaler()
    data_standerd = scaler.fit_transform(data_imputed)

    return data_standerd, y

def val_preprocess(data):
    # 去掉重复值
    data.drop_duplicates(inplace=True)

    # 填充缺失值
    imputer = IterativeImputer(random_state=0)
    data_imputed = imputer.fit_transform(data)
    data_imputed = pd.DataFrame(data_imputed, columns=data.columns)
    data_imputed = data_imputed.dropna()
    y = data_imputed['label']
    columns_to_drop = ['label']
    data_imputed = data_imputed.drop(columns_to_drop, axis=1)

    # 标准化
    scaler = StandardScaler()
    data_standerd = scaler.fit_transform(data_imputed)

    return data_standerd, y 
    
# 构建模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def pca_transform(X, mean, top_k_eigenvectors):
    # 中心化数据
    X_centered = X - mean
    
    # 将数据投影到选取的特征向量上
    X_transformed = np.dot(X_centered, top_k_eigenvectors)
    
    return X_transformed

if (__name__ == "__main__"):
    df_train = pd.read_csv("train_10000.csv")
    df_val = pd.read_csv("validate.csv")
    mean1 = df_train.mean()
    mean2 = df_val.mean()
    columns_to_drop = mean1[abs(mean1 - mean2) > 100].index
    df_train = df_train.drop(columns_to_drop, axis=1)
    df_val = df_val.drop(columns_to_drop, axis=1)
    X_train, y_train = train_preprocess(df_train)
    X_val, y_val = val_preprocess(df_val)
    print(X_train.shape)
    print(X_val.shape)
    
    # 主成分分析
    # X_train_array = X_train
    # X_val_array = X_val
    # pca = PCA(n_components=40)  # 设置保留的主成分数量
    # X_train = pca.fit_transform(X_train_array)
    # X_val = pca.transform(X_val_array)
    
    # train
    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # -----------------------------MLP-----------------------------------#
    # 转换为张量
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.LongTensor(y_train.values)
    y_test = torch.LongTensor(y_test.values)
    X_val = torch.Tensor(X_val)
    y_val = torch.LongTensor(y_val.values)
    input_size = X_train.shape[1]  # 输入特征的维度
    hidden_size = 64  # 隐藏层的大小
    num_classes = 6  # 类别数量
    model = MLP(input_size, hidden_size, num_classes)


    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # 训练模型
    num_epochs = 100
    batch_size = 32
    for epoch in range(num_epochs):
        permutation = torch.randperm(X_train.size()[0])
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            predict = model(X_val)
            loss_validate = criterion(predict, y_val)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每个epoch打印损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validate Loss: {loss_validate.item():.4f}")
        
        # 在测试集上进行评估
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            print(f"Test Accuracy: {accuracy:.4f}")

        with torch.no_grad():
            outputs = model(X_val)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_val).sum().item() / y_val.size(0)
            print(f"Validation Accuracy: {accuracy:.4f}")

    # 在测试集上进行评估
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f"Test Accuracy: {accuracy:.4f}")

    with torch.no_grad():
        outputs = model(X_val)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_val).sum().item() / y_val.size(0)
        print(f"Validation Accuracy: {accuracy:.4f}")
