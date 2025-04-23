import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import warnings
import random
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path, is_training=True):
    """
    加载并预处理数据
    :param file_path: 数据文件路径
    :param is_training: 是否为训练数据（包含label和网易筛选结论）
    :return: 特征矩阵X和标签y（如果是训练数据）
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 处理缺失值
    experience_cols = ['经验-1', '经验-2', '经验-3']
    requirement_cols = ['公司要求-1', '公司要求-2', '公司要求-3']
    
    for col in experience_cols + requirement_cols:
        df[col] = df[col].fillna(0)
    
    # 添加特征交互
    for i in range(1, 4):
        # 基础匹配度
        df[f'匹配度-{i}'] = df[f'经验-{i}'] - df[f'公司要求-{i}']

        # 相对匹配度（相对于要求的比例）
        df[f'相对匹配度-{i}'] = df[f'经验-{i}'] / (df[f'公司要求-{i}'] + 1e-6)  # 避免除以0

        # 是否满足要求（二值特征）
        df[f'满足要求-{i}'] = (df[f'经验-{i}'] >= df[f'公司要求-{i}']).astype(int)

    # 计算总体匹配度（使用原始特征，让模型学习权重）
    df['总体匹配度'] = df[[f'匹配度-{i}' for i in range(1, 4)]].abs().mean(axis=1)

    # 添加跨工作经历的特征
    df['最大经验'] = df[experience_cols].max(axis=1)
    df['最小经验'] = df[experience_cols].min(axis=1)
    df['经验方差'] = df[experience_cols].var(axis=1)

    # 添加满足要求的统计特征
    df['满足要求总数'] = df[[f'满足要求-{i}' for i in range(1, 4)]].sum(axis=1)
    df['最近工作满足要求'] = df['满足要求-1']
    
    # 分离特征和标签
    if is_training:
        X = df.drop(columns=['label', '网易筛选结论'])
        y = df['label']
        return X, y
    else:
        # 预测数据不需要label和网易筛选结论
        X = df
        return X

def evaluate_metrics(y_true, y_prob, threshold):
    # 预测标签：概率 >= threshold 为1，否则为0
    y_pred = (y_prob >= threshold).astype(int)
    
    # 计算各项指标
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        # 预测通过率
        'pass_rate': np.mean(y_pred),
        # 具体数值
        'true_positives': np.sum((y_true == 1) & (y_pred == 1)),
        'predicted_positives': np.sum(y_pred == 1),
        'total_positives': np.sum(y_true == 1),
        # 添加预测为正例的比例
        'predicted_positive_ratio': np.mean(y_pred)
    }
    
    return metrics

def cross_validate(X, y, threshold=0.4, n_splits=5, n_repeats=3):
    # 存储所有轮次的结果
    all_repeat_results = []
    
    # 打印数据集的总体信息
    print(f"\n数据集总体信息：")
    print(f"总样本数: {len(y)}")
    print(f"正样本数: {sum(y == 1)}")
    print(f"负样本数: {sum(y == 0)}")
    
    # 进行多轮交叉验证
    for repeat in range(n_repeats):
        print(f"\n第 {repeat+1}/{n_repeats} 轮交叉验证：")
        
        # 存储当前轮次所有折的结果
        repeat_results = []
        
        # 设置5折交叉验证（移除random_state）
        kfold = KFold(n_splits=n_splits, shuffle=True)
        
        # 进行5折交叉验证
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 初始化模型
            model = XGBClassifier(
                max_depth=4,
                learning_rate=0.01,
                n_estimators=500,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1
            )
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测概率
            y_prob = model.predict_proba(X_val)[:, 1]
            
            # 评估指标
            metrics = evaluate_metrics(y_val, y_prob, threshold)
            
            # 添加折的信息
            metrics['fold'] = fold + 1
            metrics['repeat'] = repeat + 1
            
            # 记录结果
            repeat_results.append(metrics)
        
        # 计算当前轮次的平均结果
        repeat_avg = pd.DataFrame(repeat_results).mean()
        print(f"\n第 {repeat+1} 轮平均结果：")
        print(f"F1值: {repeat_avg['f1']:.4f}")
        print(f"召回率: {repeat_avg['recall']:.4f}")
        print(f"精确率: {repeat_avg['precision']:.4f}")
        
        all_repeat_results.extend(repeat_results)
    
    # 将结果转换为DataFrame
    results_df = pd.DataFrame(all_repeat_results)
    
    # 计算总体统计信息
    print("\n总体统计信息：")
    for metric in ['f1', 'recall', 'precision', 'accuracy', 'auc']:
        print(f"\n{metric}的统计信息：")
        print(f"平均值: {results_df[metric].mean():.4f}")
        print(f"标准差: {results_df[metric].std():.4f}")
        print(f"最小值: {results_df[metric].min():.4f}")
        print(f"最大值: {results_df[metric].max():.4f}")
        print(f"中位数: {results_df[metric].median():.4f}")
    
    return results_df

def random_search(X, y, threshold=0.4, n_iter=200):
    # 定义参数分布
    param_dist = {
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800],
        'min_child_weight': [1, 2, 3, 4, 5],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'reg_alpha': [0, 0.1, 0.5, 1, 2],
        'reg_lambda': [0.5, 1, 1.5, 2, 2.5]
    }
    
    # 存储最佳结果
    best_f1 = 0
    best_recall = 0
    best_precision = 0
    best_params = None
    best_results = None
    
    # 记录所有尝试的结果
    all_results = []
    
    # 进行随机搜索
    for i in range(n_iter):
        # 随机选择参数组合
        params = {
            'max_depth': random.choice(param_dist['max_depth']),
            'learning_rate': random.choice(param_dist['learning_rate']),
            'n_estimators': random.choice(param_dist['n_estimators']),
            'min_child_weight': random.choice(param_dist['min_child_weight']),
            'subsample': random.choice(param_dist['subsample']),
            'colsample_bytree': random.choice(param_dist['colsample_bytree']),
            'gamma': random.choice(param_dist['gamma']),
            'reg_alpha': random.choice(param_dist['reg_alpha']),
            'reg_lambda': random.choice(param_dist['reg_lambda'])
        }
        
        print(f"\n第 {i+1}/{n_iter} 次尝试，参数组合: {params}")
        
        # 进行5轮10折交叉验证
        results_df = cross_validate(X, y, threshold, n_splits=10, n_repeats=5)
        
        # 计算平均指标
        avg_f1 = results_df['f1'].mean()
        avg_recall = results_df['recall'].mean()
        avg_precision = results_df['precision'].mean()
        
        # 计算标准差
        std_f1 = results_df['f1'].std()
        std_recall = results_df['recall'].std()
        std_precision = results_df['precision'].std()
        
        print(f"平均F1值: {avg_f1:.4f} (±{std_f1:.4f})")
        print(f"平均召回率: {avg_recall:.4f} (±{std_recall:.4f})")
        print(f"平均精确率: {avg_precision:.4f} (±{std_precision:.4f})")
        
        # 记录当前结果
        current_result = {
            'params': params,
            'f1': avg_f1,
            'f1_std': std_f1,
            'recall': avg_recall,
            'recall_std': std_recall,
            'precision': avg_precision,
            'precision_std': std_precision,
            'results': results_df
        }
        all_results.append(current_result)
        
        # 更新最佳结果（考虑稳定性）
        if avg_f1 > best_f1 or (avg_f1 == best_f1 and std_f1 < best_results['f1'].std()):
            best_f1 = avg_f1
            best_recall = avg_recall
            best_precision = avg_precision
            best_params = params
            best_results = results_df
    
    # 打印最佳结果
    print("\n最佳参数组合:")
    print(best_params)
    print(f"最佳平均F1值: {best_f1:.4f} (±{best_results['f1'].std():.4f})")
    print(f"对应召回率: {best_recall:.4f} (±{best_results['recall'].std():.4f})")
    print(f"对应精确率: {best_precision:.4f} (±{best_results['precision'].std():.4f})")
    
    # 打印前5个最佳结果
    print("\n前5个最佳结果:")
    sorted_results = sorted(all_results, key=lambda x: (x['f1'], -x['f1_std']), reverse=True)
    for i, result in enumerate(sorted_results[:5]):
        print(f"\n第 {i+1} 名:")
        print(f"参数: {result['params']}")
        print(f"F1值: {result['f1']:.4f} (±{result['f1_std']:.4f})")
        print(f"召回率: {result['recall']:.4f} (±{result['recall_std']:.4f})")
        print(f"精确率: {result['precision']:.4f} (±{result['precision_std']:.4f})")
    
    return best_params, best_results

def main():
    # 数据路径
    file_path = 'file/xgboost_input.xlsx'
    
    # 加载数据
    X, y = load_and_preprocess_data(file_path)
    
    # 设置阈值
    threshold = 0.4
    
    # 进行随机搜索
    best_params, best_results = random_search(X, y, threshold, n_iter=200)
    
    # 使用最佳参数训练最终模型
    final_model = XGBClassifier(**best_params)
    final_model.fit(X, y)
    
    # 保存模型
    joblib.dump(final_model, 'best_xgboost_model.joblib')
    print("\n最终模型已保存为 'best_xgboost_model.joblib'")
    
    # 输出特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特征重要性:")
    print(feature_importance.head(10))

def train():
    # 数据路径
    file_path = 'file/xgboost_input.xlsx'
    
    # 加载数据
    X, y = load_and_preprocess_data(file_path)
    
    # 打印数据集信息
    print(f"\n训练数据集信息：")
    print(f"总样本数: {len(y)}")
    print(f"正样本数: {sum(y == 1)}")
    print(f"负样本数: {sum(y == 0)}")
    
    # 使用最优参数配置
    best_params = {
        'max_depth': 3,
        'learning_rate': 0.001,
        'n_estimators': 500,
        'min_child_weight': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'gamma': 0,
        'reg_alpha': 0.5,
        'reg_lambda': 2
    }
    
    print("\n使用的最优参数配置:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # 训练模型
    print("\n开始训练模型...")
    model = XGBClassifier(**best_params)
    model.fit(X, y)
    
    # 保存模型
    model_path = 'best_xgboost_model.joblib'
    joblib.dump(model, model_path)
    print(f"\n模型已保存为 '{model_path}'")
    
    # 输出特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特征重要性:")
    print(feature_importance.head(10))
    
    # 计算训练集上的指标
    y_prob = model.predict_proba(X)[:, 1]
    metrics = evaluate_metrics(y, y_prob, threshold=0.45)
    
    print("\n训练集上的性能指标：")
    print(f"F1值: {metrics['f1']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    
    return model

def validate_new_features():
    # 数据路径
    file_path = 'file/xgboost_input.xlsx'
    
    # 加载数据
    X, y = load_and_preprocess_data(file_path)
    
    # 使用之前的最优参数
    best_params = {
        'max_depth': 3,
        'learning_rate': 0.001,
        'n_estimators': 500,
        'min_child_weight': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'gamma': 0,
        'reg_alpha': 0.5,
        'reg_lambda': 2
    }
    
    print("\n使用的最优参数配置:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # 进行5折交叉验证
    print("\n进行5折交叉验证...")
    results_df = cross_validate(X, y, threshold=0.4, n_splits=5, n_repeats=1)
    
    # 计算平均指标
    avg_metrics = results_df.mean()
    print("\n交叉验证平均结果：")
    print(f"F1值: {avg_metrics['f1']:.4f}")
    print(f"召回率: {avg_metrics['recall']:.4f}")
    print(f"精确率: {avg_metrics['precision']:.4f}")
    
    # 输出特征重要性
    model = XGBClassifier(**best_params)
    model.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特征重要性（前20名）:")
    print(feature_importance.head(20))

def predict_new_data(file_path, model_path='best_xgboost_model.joblib', threshold=0.4):
    """
    预测新数据并将结果添加到原始Excel文件中
    :param file_path: 新数据文件路径
    :param model_path: 模型文件路径
    :param threshold: 分类阈值
    :return: 预测结果DataFrame
    """
    # 加载新数据
    print(f"\n加载新数据: {file_path}")
    df = pd.read_excel(file_path)
    
    # 预处理数据（is_training=False表示这是预测数据）
    X_new = load_and_preprocess_data(file_path, is_training=False)
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = joblib.load(model_path)
    
    # 预测概率
    y_prob = model.predict_proba(X_new)[:, 1]
    
    # 预测标签
    y_pred = (y_prob >= threshold).astype(int)
    
    # 将预测结果添加到原始DataFrame
    df['predict_score'] = y_prob
    df['predict_result'] = ['通过' if p == 1 else '不通过' for p in y_pred]
    
    # 输出统计信息
    print("\n预测结果统计：")
    print(f"总样本数: {len(y_pred)}")
    print(f"预测通过数: {sum(y_pred == 1)}")
    print(f"预测不通过数: {sum(y_pred == 0)}")
    print(f"通过率: {sum(y_pred == 1) / len(y_pred):.2%}")
    
    # 输出特征重要性
    feature_importance = pd.DataFrame({
        '特征': X_new.columns,
        '重要性': model.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    print("\n特征重要性（前10名）:")
    print(feature_importance.head(10))
    
    # 保存结果到Excel
    output_path = file_path.replace('.xlsx', '_with_predict.xlsx')
    df.to_excel(output_path, index=False)
    print(f"\n预测结果已保存到: {output_path}")
    
    return df

if __name__ == "__main__":
    # 验证新特征的效果
    # validate_new_features()
    
    # 如果需要重新搜索最优参数，取消下面这行的注释
    main()
    
    # 使用最优参数直接训练模型
    # train()
    
    # 预测新数据
    # predict_new_data('file/two_xgboost_input.xlsx')