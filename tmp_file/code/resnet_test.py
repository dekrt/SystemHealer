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

def preprocess(input_path, output_path, train=True):
    # 读取CSV文件
    data = pd.read_csv(input_path)
    if train == True:
        new_df1 = data.drop_duplicates()

    z_scores = data.apply(zscore, nan_policy='omit')

    # Define outliers as observations with a Z-score above 3 or below -3
    outliers_zscore = (z_scores > 3) | (z_scores < -3)
    data_cleaned = data.mask(outliers_zscore)


    # # 创建KNNImputer对象
    # imputer = KNNImputer(n_neighbors=3)
    #
    # # 使用fit_transform填补缺失数据
    # data_filled = imputer.fit_transform(data.select_dtypes(include=[np.number]))
    #
    # # 将填补后的数据转换为DataFrame
    # data_filled_df = pd.DataFrame(data_filled, columns=data.select_dtypes(include=[np.number]).columns)
    #
    # # 将降维后的数据转换为DataFrame
    # data_reduced_df = pd.DataFrame(data_reduced)

    # Define the imputer
    imputer = IterativeImputer(random_state=0)

    # Apply the imputer
    data_imputed = imputer.fit_transform(data_cleaned)

    # Convert the result back to a DataFrame (since the output of the imputer is a numpy array)
    data_imputed = pd.DataFrame(data_imputed, columns=data.columns)

    # 删除仍有缺失的行
    data_imputed = data_imputed.dropna()

    # 保存到新的CSV文件
    data_imputed.to_csv(output_path, index=False)




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
    def __init__(self, input_dim, num_classes, num_blocks, dropout_rate=0.5):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = F.avg_pool1d(x, x.size(2))
        x = features.view(x.size(0), -1)
        x = self.linear(x)

        if return_features:
            return features.squeeze(2).detach().cpu().numpy()
        else:
            return x

def ResNet18(input_dim, num_classes, dropout_rate=0.5):
    return ResNet(input_dim, num_classes, [2, 2, 2, 2], dropout_rate)
def train(input_path, basic_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare data
    data = np.loadtxt(input_path, delimiter=',', skiprows=1)

    y = data[:, -1]
    X = data[:, 1:-1]

    train_data, val_data, train_label, val_label = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)

    # sampler = SMOTE()
    # train_data, train_label = sampler.fit_resample(train_data, train_label)
    class_weights = compute_class_weight('balanced', classes=np.unique(train_label), y=train_label)
    train_data = torch.Tensor(train_data)
    val_data = torch.Tensor(val_data)
    train_label = torch.LongTensor(train_label)
    val_label = torch.LongTensor(val_label)
    # pdb.set_trace()

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)


    def save_checkpoint(model, optimizer, epoch, path):
        torch.save(
            {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
            path)

    input_size = train_data.shape[1]
    # pdb.set_trace()
    num_classes = 6
    dropout_rate = 0.6906059730252032
    weight_decay = 1e-06
    num_epochs = 60
    lr = 0.007325438539211768

    model = ResNet18(input_size, num_classes, dropout_rate=dropout_rate).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=7, verbose=True)
    # Training

    batch_size = 32
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0
        num_batches = 0
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
            running_train_loss += loss.item()
            num_batches += 1

        train_loss = running_train_loss / num_batches
        train_losses.append(train_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # save_checkpoint(model, optimizer, epoch, f'./模型/resnet_epoch_{epoch}_time{current_time}.pt')
        # Evaluation
        model.eval()
        running_val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            val_data_unsqueezed = val_data.unsqueeze(1).to(device)  # Add a dimension for the channel and move to device
            val_label = val_label.to(device)
            outputs = model(val_data_unsqueezed)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == val_label).sum().item() / val_label.size(0)
            print(f"Validation Accuracy at Epoch {epoch + 1}: {accuracy:.4f}")
            running_val_loss += criterion(outputs, val_label).item()
            num_val_batches += 1

            val_loss = running_val_loss / num_val_batches
            val_losses.append(val_loss)
            val_features = model(val_data_unsqueezed, return_features=True)
            early_stopping(val_loss, model, val_features, f'./resnet_epoch_{epoch}_time{current_time}.pt')
        if early_stopping.early_stop:
            print("Early stopping")
            break
    best_val_features = early_stopping.best_val_features
    val_features_tsne = TSNE(n_components=2, random_state=33, init='pca',
                         learning_rate='auto').fit_transform(best_val_features)
    font = {"color": "darkred",
            "size": 13,
            "family": "serif"}

    plt.style.use("dark_background")
    plt.figure(figsize=(9, 8))

    plt.scatter(val_features_tsne[:, 0], val_features_tsne[:, 1], c=val_label.cpu().numpy(), alpha=0.6,
                cmap=plt.cm.get_cmap('rainbow', num_classes))
    plt.title("t-SNE", fontdict=font)
    cbar = plt.colorbar(ticks=range(num_classes))
    cbar.set_label(label='Class label', fontdict=font)
    plt.clim(-0.5, num_classes - 0.5)
    plt.tight_layout()
    plt.savefig(basic_path + f'_tsne.png', dpi=300)

    # Save predicted labels to a CSV file
    predicted_labels = predicted.cpu().numpy()
    val_ids = np.arange(len(val_label)) + 1
    np.savetxt(output_path, np.column_stack((val_ids, predicted_labels)), fmt='%i', delimiter=',',
               header='id,label', comments='')
    unique_labels, counts = np.unique(predicted_labels, return_counts=True)
    label_count_dict = dict(zip(unique_labels, counts))

    result = {"train_losses": train_losses, "val_losses": val_losses, "label_counts": label_count_dict}
    return result

def test(path, test_path, input_size, num_classes, dropout_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(input_size, num_classes, dropout_rate=dropout_rate).to(device)
    model.load_state_dict(torch.load(path))

    # 对测试集进行预测
    test_data = np.loadtxt(test_path, delimiter=',', skiprows=1)
    test_data = torch.Tensor(test_data).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        predicted_labels = predicted.cpu().numpy()

    # 将预测结果保存为json文件
    results = {str(i): int(predicted_labels[i]) for i in range(len(predicted_labels))}
    with open('results.json', 'w') as f:
        json.dump(results, f)

if (__name__ == "__main__"):
    preprocess("./train_10000.csv","./train_10000_processed.csv")
    train("./train_10000_processed.csv","./pic","./result.csv")
