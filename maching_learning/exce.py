import sys
import pandas as pd
import numpy as np


def read_data():
    # 读取特征名称
    while True:
        header_line = sys.stdin.readline()
        if header_line.strip():
            break
    headers = header_line.strip().split(',')
    data = []
    # 读取样本数据
    for line in sys.stdin:
        if line == '\n':  # 如果遇到空行，退出循环
            break
        if line.strip():
            values = line.strip().split(',')
            if len(values) != len(headers):
                raise ValueError("数据行的特征数量与标题行不匹配。")
            data.append(values)
    # 创建 DataFrame
    df = pd.DataFrame(data, columns=headers)
    return df


def process_missing_values(df):
    # 区分数值型和类别型特征
    numeric_cols = []
    categorical_cols = []
    for col in df.columns:
        try:
            df = df.replace('', np.nan)
            df_nonan = df[col].dropna()
            df_nonan.astype(float)
            numeric_cols.append(col)
        except ValueError:
            categorical_cols.append(col)
    # 处理数值型特征的缺失值
    for col in numeric_cols:
        df[col] = df[col].replace('', np.nan).astype(float)
        mean_value = df[col].mean()
        df[col] = df[col].fillna(mean_value)

    # 处理类别型特征的缺失值
    for col in categorical_cols:
        df[col] = df[col].replace('', np.nan)
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)
    return df, numeric_cols, categorical_cols


def standardize_numeric_features(df, numeric_cols):
    # 标准化数值型特征
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = df[col].apply(lambda x: round((x - mean) / std, 4))
    return df


def one_hot_encode(df, categorical_cols):
    # 对类别型特征进行独热编码
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    for col in df_encoded.columns:
        # 判断列是否只含0,1两种值
        if set(df_encoded[col].dropna().unique()) <= {0, 1}:
            df_encoded[col] = df_encoded[col].astype(int)

    return df_encoded


def print_processed_data(df):
    # 打印处理后的数据
    print(','.join(df.columns))
    for _, row in df.iterrows():
        print(','.join(str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
                       for x in row.values))


if __name__ == "__main__":
    # 读取数据
    df = read_data()
    # 处理缺失值
    df, numeric_cols, categorical_cols = process_missing_values(df)
    # 标准化数值型特征
    df = standardize_numeric_features(df, numeric_cols)
    # 独热编码
    df = one_hot_encode(df, categorical_cols)
    # 打印处理后的数据
    print_processed_data(df)