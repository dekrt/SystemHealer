import timm
# from torch.nn import Linear
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
# import torch.nn.functional as F
# from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight
# import pdb
# import datetime
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import TomekLinks
# from imblearn.ensemble import BalancedBaggingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Prepare data
data = np.loadtxt('./data_filled.csv', delimiter=',', skiprows=1)
y = data[:, -1]
X = data[:, 1:-1]

train_data,val_data, train_label, val_label = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)

#sampler = SMOTE()
#train_data, train_label = sampler.fit_resample(train_data, train_label)
class_weights = compute_class_weight('balanced', classes=np.unique(train_label), y=train_label)
train_data = torch.Tensor(train_data)
val_data = torch.Tensor(val_data)
train_label = torch.LongTensor(train_label)
val_label = torch.LongTensor(val_label)
#pdb.set_trace()

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# ResNet18 model
class CustomViT(nn.Module):
    def __init__(self, input_dim, num_classes, pretrained=True):
        super(CustomViT, self).__init__()
        self.embedding = nn.Linear(input_dim, 224 * 224)
        self.vit = timm.create_model("vit_large_patch16_224", pretrained=pretrained, num_classes=0)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.shape[0], 1, 224, 224)
        x = x.repeat(1, 3, 1, 1)
        x = self.vit(x)
        x = self.fc(x)
        return x

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, path)

input_size = train_data.shape[1]
#pdb.set_trace()
num_classes = 6
dropout_rate = 0.6906059730252032
weight_decay = 1e-06
num_epochs = 94
lr = 0.007325438539211768

model = CustomViT(input_size, num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
#criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Training

batch_size = 16
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
        accumulation_steps = 4  # 可以根据需要调整
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        running_train_loss += loss.item()
        num_batches += 1

    train_loss = running_train_loss / num_batches
    train_losses.append(train_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # save_checkpoint(model, optimizer, epoch, f'./模型/clip_epoch_{epoch}_time{current_time}.pt')
# Evaluation
    model.eval()
    running_val_loss = 0
    num_val_batches = 0
    with torch.no_grad():
        val_data_unsqueezed =val_data.unsqueeze(1).to(device)  # Add a dimension for the channel and move to device
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
        # val_features_tsne = TSNE(n_components=2, random_state=33, init='pca', learning_rate='auto').fit_transform(val_features)


        # font = {"color": "darkred",
        # "size": 13, 
        # "family" : "serif"}

        # plt.style.use("dark_background")
        # plt.figure(figsize=(9, 8))

        # plt.scatter(val_features_tsne[:, 0], val_features_tsne[:, 1], c=val_label.cpu().numpy(), alpha=0.6, cmap=plt.cm.get_cmap('rainbow', num_classes))
        # plt.title("t-SNE", fontdict=font)
        # cbar = plt.colorbar(ticks=range(num_classes)) 
        # cbar.set_label(label='Class label', fontdict=font)
        # plt.clim(-0.5, num_classes - 0.5)
        # plt.tight_layout()

        # plt.savefig(f'./pic/tsne_visualization_{epoch}.png', dpi=300)


        # Save predicted labels to a CSV file
    # predicted_labels = predicted.cpu().numpy()
    # val_ids = np.arange(len(val_label)) + 1
    # np.savetxt('predicted_labels.csv', np.column_stack((val_ids, predicted_labels)), fmt='%i', delimiter=',', header='id,label', comments='')
    # unique_labels, counts = np.unique(predicted_labels, return_counts=True)
    # label_count_dict = dict(zip(unique_labels, counts))
    # plt.figure()
    # # 绘制饼状图
    # plt.style.use('default')
    # fig, ax = plt.subplots()
    # ax.pie(counts, labels=unique_labels, autopct='%1.1f%%', startangle=90)
    # ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # plt.title("Predicted Class Distribution")
    # plt.savefig("./pic/pie_chart.png", dpi=300)


# epochs = np.arange(1, num_epochs + 1)
# plt.figure()
# plt.style.use('classic')
# plt.plot(epochs, train_losses, label='Train Loss')
# plt.plot(epochs, val_losses, label='Validation Loss')

# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss Curves')
# plt.legend()

# plt.savefig("./pic/loss_curves.png", dpi=300)

# 生成分类报告
report = classification_report(val_label.cpu().numpy(), predicted.cpu().numpy(), output_dict=True)

# 从分类报告中提取精确率、召回率和 F1 分数
macro_precision = report['macro avg']['precision']
macro_recall = report['macro avg']['recall']
macro_f1 = report['macro avg']['f1-score']

# 打印 MacroF1
print("MacroF1:", macro_f1)
# 将结果写入文件
# with open("results.txt", "w") as f:
#     f.write("MacroF1: {}\n".format(macro_f1))
#     f.write("\nClassification Report:\n")
#     f.write(classification_report(val_label.cpu().numpy(), predicted.cpu().numpy()))