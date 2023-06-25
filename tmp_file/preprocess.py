import pandas as pd
from sklearn.impute import KNNImputer


def preprcess(input_path, output_path):
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
    data_filled_df.to_csv('data\\validate.csv', index=False)