import sys
import numpy as np

def read_data(N, M):
    data = []
    labels = []
    for _ in range(N):
        while True:
            line = sys.stdin.readline()
            if line.strip():
                break
        elements = line.strip().split()
        if len(elements) != M + 1:
            raise ValueError("样本的特征数量与指定的 M 不匹配。")
        features = list(map(float, elements[:-1]))
        label = elements[-1]
        data.append(features)
        labels.append(label)
    return np.array(data), labels

def lda(data, labels, num_components):
    class_labels = np.unique(labels)
    mean_overall = np.mean(data, axis=0)
    S_W = np.zeros((data.shape[1], data.shape[1]))
    S_B = np.zeros((data.shape[1], data.shape[1]))
    for label in class_labels:
        data_c = data[np.array(labels) == label]
        mean_c = np.mean(data_c, axis=0)
        S_W += np.dot((data_c - mean_c).T, (data_c - mean_c))
        n_c = data_c.shape[0]
        mean_diff = (mean_c - mean_overall).reshape(-1, 1)
        S_B += n_c * np.dot(mean_diff, mean_diff.T)
    # 解广义特征值问题
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
    # 取前 num_components 个特征向量
    idxs = np.argsort(-np.abs(eigvals))
    W = eigvecs[:, idxs[:num_components]].real
    return W

def compute_distance_matrix(transformed_data):
    N = transformed_data.shape[0]
    distance_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            diff = transformed_data[i] - transformed_data[j]
            distance = np.linalg.norm(diff)
            distance_matrix[i, j] = round(distance, 4)
    return distance_matrix

if __name__ == "__main__":
    # 读取样本数量和特征数量
    while True:
        first_line = sys.stdin.readline()
        if first_line.strip():
            break
    N_str, M_str = first_line.strip().split()
    N = int(N_str)
    M = int(M_str)
    if N <= 0 or M <= 0:
        raise ValueError("N 和 M 必须为正整数。")
    # 读取数据和标签
    data, labels = read_data(N, M)
    # 线性判别分析，降到 1 维
    W = lda(data, labels, num_components=1)
    # 投影到新空间
    transformed_data = np.dot(data, W)
    # 计算距离矩阵
    distance_matrix = compute_distance_matrix(transformed_data)
    # 输出结果
    for row in distance_matrix:
        print(' '.join(f"{val:.4f}" for val in row))