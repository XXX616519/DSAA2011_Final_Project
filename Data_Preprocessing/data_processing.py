from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.io import arff
import pandas as pd

def preprocess_beans_data(input_path, output_path):
    
    data = arff.loadarff(input_path)
    df = pd.DataFrame(data[0])
    df['Class'] = df['Class'].str.decode('utf-8')

    # 1. 处理缺失值（验证完整性）
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.mean(numeric_only=True))

    # 2. 类别特征编码
    encoder = OneHotEncoder(sparse_output=False, drop="if_binary")
    class_encoded = encoder.fit_transform(df[['Class']])
    class_df = pd.DataFrame(class_encoded, 
                          columns=encoder.categories_[0],
                          dtype=int)

    # 3. 数值特征标准化
    numeric_cols = df.columns[df.columns != 'Class'].tolist()
    scaler = StandardScaler()
    df_numeric = pd.DataFrame(scaler.fit_transform(df[numeric_cols]),
                             columns=numeric_cols)

    # 4. 合并数据集
    df_processed = pd.concat([df_numeric, class_df], axis=1)

    # 5. 保存处理结果
    df_processed.to_csv(output_path, index=False)
    
    # 验证输出
    print(f"处理完成！保存至: {output_path}")
    print("最终数据结构:")
    print(f"- 样本数: {df_processed.shape[0]}")
    print(f"- 特征数: {df_processed.shape[1]}")
    print(f"- 数值特征: {len(numeric_cols)} (已标准化)")
    print(f"- 类别特征: {len(encoder.categories_[0])} (独热编码)")
    print(df_processed[numeric_cols].describe().loc[['mean', 'std']].round(2))

if __name__ == "__main__":
    input_file = "DryBeanDataset/Dry_Bean_Dataset.arff"
    output_file = "DryBeanDataset/Processed_Dry_Beans.csv"
    preprocess_beans_data(input_file, output_file)

# python Data_Preprocessing/data_processing.py