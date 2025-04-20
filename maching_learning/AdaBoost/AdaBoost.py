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
        elements = list(map(float, line.strip().split()))
        if len(elements) != M + 1:
            raise ValueError("Number of features does not match specified M.")
        data.append(elements[:-1])
        labels.append(elements[-1])
    return np.array(data), np.array(labels)

def build_stump(data, labels, weights):
    N, M = data.shape
    min_error = float('inf')
    best_stump = {}
    for feature_index in range(M):
        feature_values = data[:, feature_index]  # 第一列特征
        thresholds = np.unique(feature_values)
        for thresh in thresholds:   # 提取特征值中的唯一元素，作为后续划分的阈值
            for inequality in ['lt', 'gt']:
                predictions = np.ones(N)
                if inequality == 'lt':
                    predictions[data[:, feature_index] <= thresh] = 1
                    predictions[data[:, feature_index] > thresh] = -1
                else:
                    predictions[data[:, feature_index] > thresh] = 1
                    predictions[data[:, feature_index] <= thresh] = -1
                errors = (predictions != labels).astype(float)
                weighted_error = np.dot(weights, errors)
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_stump['feature_index'] = feature_index
                    best_stump['threshold'] = thresh
                    best_stump['inequality'] = inequality
                    best_stump['predictions'] = predictions.copy()
    return best_stump, min_error

def adaBoost_train(data, labels, T):
    N = data.shape[0]
    weights = np.ones(N) / N
    classifiers = []
    for t in range(T):
        stump, error = build_stump(data, labels, weights)
        alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))
        stump['alpha'] = alpha
        classifiers.append(stump)
        # Update weights
        weights *= np.exp(-alpha * labels * stump['predictions'])
        weights /= np.sum(weights)
        # Output the weak classifier information
        feature_index = stump['feature_index']
        threshold = stump['threshold']
        inequality = stump['inequality']
        alpha_value = round(alpha, 4)
        print(f"{feature_index} {threshold} {inequality} {alpha_value}")
    return classifiers

if __name__ == "__main__":
    while True:
        first_line = sys.stdin.readline()
        if first_line.strip():
            break
    N_str, M_str = first_line.strip().split()
    N = int(N_str)
    M = int(M_str)
    if N <= 0 or M <= 0:
        raise ValueError("N and M must be positive integers.")
    data, labels = read_data(N, M)
    while True:
        T_line = sys.stdin.readline()
        if T_line.strip():
            break
    T = int(T_line.strip())
    if T <= 0:
        raise ValueError("T must be a positive integer.")
    adaBoost_train(data, labels, T)