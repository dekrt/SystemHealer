import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import KMeansSMOTE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def train_preprocess(data):
    # 去掉重复值
    data.drop_duplicates(inplace=True)
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
    data_standard = scaler.fit_transform(data_imputed)

    return data_standard, y


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        out = self.activation(out)
        out = self.fc4(out)
        return out

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_val_features = None  # 添加这一行来保存最优的val_features

    def __call__(self, val_loss, model, val_features, path):  # 在这里添加val_features参数
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, val_features, path)  # 在这里传入val_features参数
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, val_features, path)  # 在这里传入val_features参数
            self.counter = 0

    def save_checkpoint(self, val_loss, model, val_features, path):  # 在这里添加val_features参数
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
        self.best_val_features = val_features  # 在这里保存最优的val_features

if (__name__ == "__main__"):
    data_train = pd.read_csv("train_10000.csv")
    data_val = pd.read_csv("validate.csv")
    mean1 = data_train.mean()
    mean2 = data_val.mean()
    columns_to_drop = mean1[abs(mean1 - mean2) > 100].index
    data_train = data_train.drop(columns_to_drop, axis=1)
    data_val = data_val.drop(columns_to_drop, axis=1)
    X_train, y_train = train_preprocess(data_train)
    X_val, y_val = val_preprocess(data_val)
    
    # kmeans_smote = KMeansSMOTE(cluster_balance_threshold=0.064, random_state=42)
    kmeans_smote = KMeansSMOTE()
    X_train, y_train = kmeans_smote.fit_resample(X_train, y_train)


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
    hidden_size = 200  # 隐藏层的大小
    num_classes = 6  # 类别数量
    model = MLP(input_size, hidden_size, num_classes)


    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # 训练模型
    num_epochs = 100
    batch_size = 30
    patience = 5  # patience for early stopping
    best_loss = np.inf
    stop_counter = 0
    early_stopping = EarlyStopping(patience=7, verbose=True)

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
        # Check for early stopping
        if loss_validate < best_loss:
            best_loss = loss_validate
            stop_counter = 0
        else:
            stop_counter += 1
            if stop_counter >= patience:
                print("Early stopping")
                break

        # 每个epoch打印损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validate Loss: {loss_validate.item():.4f}")

        # early_stopping(loss.item(), model, val_features, 'checkpoint.pt')
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
