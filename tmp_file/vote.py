import numpy as np
import pandas as pd
import catboost as cb
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import KMeansSMOTE
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.ensemble import VotingClassifier

# 创建一个包装器类，对CatBoostClassifier进行包装
class CatBoostClassifierWrapper(cb.CatBoostClassifier):
    def predict(self, X):
        return super().predict(X).ravel()

# 读取数据集
df_train = pd.read_csv("train_10000.csv")
df_val = pd.read_csv("validate_1000.csv")

# 缺失值处理
df_train = df_train.fillna(df_train.mean())
df_val = df_val.fillna(df_val.mean())

# 切分数据集
X_train = np.array(df_train.drop(["label", "sample_id"], axis=1))
y_train = np.array(df_train["label"])

X_val = np.array(df_val.drop(["label", "sample_id"], axis=1))
y_val = np.array(df_val["label"])

# 标准化数据
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)

# 过采样数据
kmeans_smote = KMeansSMOTE(cluster_balance_threshold=0.064, random_state=42)
X_train_resampled, y_train_resampled = kmeans_smote.fit_resample(X_train_scaled, y_train)

# 创建模型
catboost_model = CatBoostClassifierWrapper(random_seed=42, verbose=False)
gradientboost_model = GradientBoostingClassifier(random_state=42)
svm_model = svm.SVC(random_state=42)

voting_model = VotingClassifier(estimators=[
    ('cb', catboost_model),
    ('gb', gradientboost_model),
    ('svm', svm_model)],
    voting='hard')

# 训练模型
voting_model = voting_model.fit(X_train_resampled, y_train_resampled)

# 预测
y_val_pred = voting_model.predict(X_val_scaled)
macro_f1 = f1_score(y_val, y_val_pred, average='macro')
print(macro_f1)
print(classification_report(y_val, y_val_pred))

y_train_pred = voting_model.predict(X_train_scaled)
macro_f1 = f1_score(y_train, y_train_pred, average='macro')
print(macro_f1)
print(classification_report(y_train, y_train_pred))
