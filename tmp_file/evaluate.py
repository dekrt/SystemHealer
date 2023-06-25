import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import ResNet18, BasicBlock
import os

# 加载模型
def load_model(model_path, input_size, num_classes):
    model = ResNet18(input_size, num_classes)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# 准备新的验证集
def prepare_data(data_path):
    data = np.loadtxt(data_path, delimiter=',', skiprows=1)
    y = data[:, -1]
    X = data[:, 1:-1]

    scaler = StandardScaler()
    X_test = scaler.fit_transform(X)

    X_test = torch.Tensor(X_test)
    y_test = torch.LongTensor(y)

    return X_test, y_test

# 计算准确率
def compute_accuracy(model, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_test_unsqueezed = X_test.unsqueeze(1).to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        outputs = model(X_test_unsqueezed)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)

    return accuracy

if __name__ == "__main__":
    model_directory = './best/'
    data_path = './验证集/validate.csv'
    input_size = 107
    num_classes = 6

    X_test, y_test = prepare_data(data_path)

    for model_file in sorted(os.listdir(model_directory)):
        if model_file.endswith('.pt'):
            model_path = os.path.join(model_directory, model_file)
            print(f"Evaluating model: {model_path}")
            model = load_model(model_path, input_size, num_classes)
            accuracy = compute_accuracy(model, X_test, y_test)
            print(f"Accuracy on the new validation set for {model_file}: {accuracy:.4f}\n")
