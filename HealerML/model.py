import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn.functional as F
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight
import pdb
import datetime
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.impute import KNNImputer

def preprocess(input_path, output_path):
    # 读取CSV文件
    data = pd.read_csv(input_path)

    # 创建KNNImputer对象
    imputer = KNNImputer(n_neighbors=3)

    # 使用fit_transform填补缺失数据
    data_filled = imputer.fit_transform(data)

    # 将填补后的数据转换为DataFrame
    data_filled_df = pd.DataFrame(data_filled, columns=data.columns)

    # 删除仍有缺失的行
    data_filled_df = data_filled_df.dropna()

    # 保存到新的CSV文件
    data_filled_df.to_csv(output_path, index=False)


def Resnet(input_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare data
    data = np.loadtxt(input_path, delimiter=',', skiprows=1)
    y = data[:, -1]
    X = data[:, 1:-1]

    train_data,val_data, train_label, val_label = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)

    #sampler = TomekLinks()
    #train_data, train_label = sampler.fit_resample(train_data, train_label)
    #class_weights = compute_class_weight('balanced', classes=np.unique(train_label), y=train_label)
    train_data = torch.Tensor(train_data)
    val_data = torch.Tensor(val_data)
    train_label = torch.LongTensor(train_label)
    val_label = torch.LongTensor(val_label)
    #pdb.set_trace()

    #class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # ResNet18 model
    class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(p=0.5)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.dropout(out)
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    class ResNet(nn.Module):
        def __init__(self, input_dim, num_classes, num_blocks):
            super(ResNet, self).__init__()
            self.in_channels = 64

            self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(64)
            self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512, num_classes)

        def _make_layer(self, out_channels, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(BasicBlock(self.in_channels, out_channels, stride))
                self.in_channels = out_channels
            return nn.Sequential(*layers)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = F.avg_pool1d(x, x.size(2))
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            return x

    def ResNet18(input_dim, num_classes):
        return ResNet(input_dim, num_classes, [2, 2, 2, 2])

    def save_checkpoint(model, optimizer, epoch, path):
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, path)

    input_size = train_data.shape[1]
    #pdb.set_trace()
    num_classes = 6
    model = ResNet18(input_size, num_classes).to(device)

    # Loss and optimizer
    #criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    num_epochs = 10
    batch_size = 32
    for epoch in range(num_epochs):
        permutation = torch.randperm(train_data.size()[0])
        for i in range(0, train_data.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = train_data[indices], train_label[indices]

            batch_X = batch_X.unsqueeze(1).to(device)  # Add a dimension for the channel and move to device
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # save_checkpoint(model, optimizer, epoch, f'./__model/resnet_epoch_{epoch}_time{current_time}.pt')
    # Evaluation
        with torch.no_grad():
            val_data_unsqueezed =val_data.unsqueeze(1).to(device)  # Add a dimension for the channel and move to device
            val_label = val_label.to(device)
            outputs = model(val_data_unsqueezed)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == val_label).sum().item() / val_label.size(0)
            print(f"Validation Accuracy at Epoch {epoch + 1}: {accuracy:.4f}")
    predicted_labels = predicted.cpu().numpy()
    val_ids = np.arange(len(val_label))
    # np.savetxt(output_path, np.column_stack((val_ids, predicted_labels)), delimiter=',', header='id,label', comments='')
    np.savetxt(output_path, np.column_stack((val_ids, predicted_labels)), fmt='%i', delimiter=',', header='id,label', comments='')
