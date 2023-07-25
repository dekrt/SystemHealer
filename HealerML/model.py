# import necessary libraries
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import KMeansSMOTE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# define global variables
input_size_global = 0
model_path = ''


# function to preprocess the data before training
def preprocess(data):
    # remove duplicates
    data.drop_duplicates(inplace=True)
    # fill missing values
    imputer = IterativeImputer(random_state=0)
    data_imputed = imputer.fit_transform(data)
    data_imputed = pd.DataFrame(data_imputed, columns=data.columns)
    data_imputed = data_imputed.dropna()
    y = data_imputed['label']
    columns_to_drop = ['label']
    data_imputed = data_imputed.drop(columns_to_drop, axis=1)
    # standardize the data
    scaler = StandardScaler()
    data_standerd = scaler.fit_transform(data_imputed)
    return data_standerd, y


# function to preprocess the data before testing
def test_preprocess(data):
    # fill missing values
    imputer = IterativeImputer(random_state=0)
    data_imputed = imputer.fit_transform(data)
    data_imputed = pd.DataFrame(data_imputed, columns=data.columns)
    data_imputed = data_imputed.dropna()
    # standardize the data
    scaler = StandardScaler()
    data_standerd = scaler.fit_transform(data_imputed)
    return data_standerd


# define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)
        self.activation = nn.LeakyReLU()

    def forward(self, x, return_features=False):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        out = self.activation(out)
        if return_features:
            return out.detach().cpu().numpy()
        out = self.fc4(out)
        return out


# define the EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


# function to train the model
def train(file_basic_path):
    global input_size_global, model_path
    input_path = file_basic_path + '.csv'
    train_loss_list = []
    test_loss_list = []
    data = pd.read_csv(input_path)
    X_train, y_train = preprocess(data)
    kmeans_smote = KMeansSMOTE(cluster_balance_threshold=0.064, random_state=42)
    X_train, y_train = kmeans_smote.fit_resample(X_train, y_train)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # convert to tensors
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.LongTensor(y_train.values)
    y_test = torch.LongTensor(y_test.values)
    input_size = X_train.shape[1]  # input feature dimensions
    input_size_global = X_train.shape[1]
    hidden_size = 200  # size of the hidden layer
    num_classes = 6  # number of classes
    model = MLP(input_size, hidden_size, num_classes)

    # define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # train the model
    num_epochs = 100
    batch_size = 30
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in range(num_epochs):
        permutation = torch.randperm(X_train.size()[0])
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            # forward propagation
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            predict = model(X_test)
            loss_test = criterion(predict, y_test)

            # backward propagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print the loss at each epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validate Loss: {loss_test.item():.4f}")
        train_loss_list.append(loss.item())
        test_loss_list.append(loss_test.item())

        # evaluate on the test set
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            print(f"Test Accuracy: {accuracy:.4f}")
            val_loss = criterion(outputs, y_test)
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("early stop")
                break

    # evaluate on the test set
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f"Test Accuracy: {accuracy:.4f}")
    label_count_dict = [0, 0, 0, 0, 0, 0]
    for value in data['label']:
        label_count_dict[value] += 1
    val_features = model(X_test, return_features=True)
    val_features_tsne = TSNE(n_components=2, random_state=33, init='pca',
                             learning_rate='auto').fit_transform(
        val_features)
    font = {"color": "darkred",
            "size": 13,
            "family": "serif"}

    plt.style.use("dark_background")
    plt.figure(figsize=(9, 8))

    plt.scatter(val_features_tsne[:, 0], val_features_tsne[:, 1], c=y_test.cpu().numpy(), alpha=0.6,
                cmap=plt.cm.get_cmap('rainbow', num_classes))
    plt.title("t-SNE", fontdict=font)
    cbar = plt.colorbar(ticks=range(num_classes))
    cbar.set_label(label='Class label', fontdict=font)
    plt.clim(-0.5, num_classes - 0.5)
    plt.tight_layout()
    plt.savefig(file_basic_path + f'_tsne.png', dpi=300)
    result = {"train_losses": train_loss_list, "val_losses": test_loss_list, "label_counts": label_count_dict}
    torch.save(model.state_dict(), file_basic_path + '_model.pth')
    model_path = file_basic_path + '_model.pth'
    return result


# function to test the model
def test(file_basic_path):
    global input_size_global, model_path
    input_path = file_basic_path + '.csv'
    data = pd.read_csv(input_path)
    data = test_preprocess(data)
    input_size = input_size_global  # input feature dimensions
    hidden_size = 200  # size of the hidden layer
    num_classes = 6  # number of classes
    model = MLP(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        data = torch.Tensor(data)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)

    label_count_list = [0, 0, 0, 0, 0, 0]
    for value in predicted:
        label_count_list[value] += 1

    result_dict = {}
    predicted_new = predicted.tolist()
    for i, value in enumerate(predicted_new, start=1):
        result_dict[str(i)] = value
    with open(file_basic_path + '_predicted.json', 'w') as json_file:
        json.dump(result_dict, json_file)
    result = {"label_counts": label_count_list}
    val_features = model(data, return_features=True)
    val_features_tsne = TSNE(n_components=2, random_state=33, init='pca',
                             learning_rate='auto').fit_transform(
        val_features)
    font = {"color": "darkred",
            "size": 13,
            "family": "serif"}

    plt.style.use("dark_background")
    plt.figure(figsize=(9, 8))

    plt.scatter(val_features_tsne[:, 0], val_features_tsne[:, 1], c=predicted.cpu().numpy(), alpha=0.6,
                cmap=plt.cm.get_cmap('rainbow', num_classes))
    plt.title("t-SNE", fontdict=font)
    cbar = plt.colorbar(ticks=range(num_classes))
    cbar.set_label(label='Class label', fontdict=font)
    plt.clim(-0.5, num_classes - 0.5)
    plt.tight_layout()
    plt.savefig(file_basic_path + f'_tsne.png', dpi=300)
    return result
