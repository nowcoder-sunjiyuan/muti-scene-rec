import sys
import numpy as np
import math


'''
10 2
1.2 2.0
2.0 1.0
1.5 1.5
8.0 8.0
9.0 8.0
8.5 9.0
0.0 0.0
9.0 9.0
10.0 10.0
5.0 5.0
5 3
'''

# 设置调整因子的函数
def c_factor(n):
    if n > 2:
        h = math.log(n - 1) + 0.5772  # 调和数近似
        return 2 * h - (2 * (n - 1) / n)
    elif n == 2:
        return 1
    else:
        return 0


# 随机数生成器设置
def set_random_seed(seed):
    np.random.seed(seed)


# 隔离树（Isolation Tree）
class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.root = None

    def fit(self, X):
        self.root = self._fit(X, current_height=0)

    def _fit(self, X, current_height):
        if current_height >= self.height_limit or X.shape[0] <= 1:
            return {'size': X.shape[0]}
        else:
            q = np.random.randint(0, X.shape[1] - 1)  # 使用numpy生成随机数, 随机选择特征列
            min_val = X[:, q].min()
            max_val = X[:, q].max()
            if min_val == max_val:
                return {'size': X.shape[0]}
            p = np.random.uniform(min_val, max_val)  # 使用numpy生成随机数
            left_indices = X[:, q] < p
            right_indices = X[:, q] >= p
            return {
                'feature': q,
                'threshold': p,
                'left': self._fit(X[left_indices], current_height + 1),
                'right': self._fit(X[right_indices], current_height + 1)
            }

    def path_length(self, x):
        return self._path_length(x, self.root, current_height=0)

    def _path_length(self, x, node, current_height):
        if 'size' in node:
            return current_height + c_factor(node['size'])
        q = node['feature']
        p = node['threshold']
        if x[q] < p:
            return self._path_length(x, node['left'], current_height + 1)
        else:
            return self._path_length(x, node['right'], current_height + 1)


# 孤立森林（Isolation Forest）
class IsolationForest:
    def __init__(self, n_trees=100, sample_size=256):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.trees = []
        self.c = c_factor(self.sample_size)

    def fit(self, X):
        self.trees = []
        height_limit = math.ceil(math.log2(self.sample_size))
        for _ in range(self.n_trees):
            if X.shape[0] > self.sample_size:
                X_sample = X[np.random.choice(X.shape[0], self.sample_size, replace=False)]  # 使用numpy随机选择
            else:
                X_sample = X
            tree = IsolationTree(height_limit)
            tree.fit(X_sample)
            self.trees.append(tree)

    def anomaly_score(self, X):
        scores = []
        for x in X:
            path_lengths = [tree.path_length(x) for tree in self.trees]
            E_hx = np.mean(path_lengths)
            score = 2 ** (-E_hx / self.c)  # 异常分数计算公式
            scores.append(score)
        return np.array(scores)


if __name__ == "__main__":
    # 设置随机种子（可选）
    set_random_seed(0)

    # 读取 N 和 M
    while True:
        line = sys.stdin.readline()
        if line.strip():
            break
    N_str, M_str = line.strip().split()
    N = int(N_str)
    M = int(M_str)

    # 读取数据
    data = []
    for _ in range(N):
        while True:
            line = sys.stdin.readline()
            if line.strip():
                break
        elements = list(map(float, line.strip().split()))
        data.append(elements)
    X = np.array(data)

    # 读取 T 和 psi
    while True:
        line = sys.stdin.readline()
        if line.strip():
            break
    T_str, psi_str = line.strip().split()
    T = int(T_str)
    psi = int(psi_str)

    # 训练孤立森林模型
    forest = IsolationForest(n_trees=T, sample_size=psi)
    forest.fit(X)

    # 计算异常分数
    scores = forest.anomaly_score(X)
    # 设定阈值 tau = 0.5
    tau = 0.5
    labels = (scores > tau).astype(int)
    # 输出结果
    for score, label in zip(scores, labels):
        print(f"{score:.4f} {label}")