import sys
import numpy as np

def read_matrix(m, n):
    matrix = []
    for _ in range(m):
        while True:
            line = sys.stdin.readline().strip()
            if line:
                break
        elements = list(map(int, line.strip().split()))
        if len(elements) != n:
            raise ValueError("Number of elements in the row does not match specified column size.")
        matrix.append(elements)
    return np.array(matrix, dtype=float)

def svd_reconstruct(matrix, k):
    # 进行奇异值分解
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    # 取前 k 个奇异值
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    # 重构矩阵
    reconstructed_matrix = np.dot(np.dot(U_k, S_k), VT_k)
    # 保留两位小数
    reconstructed_matrix = np.round(reconstructed_matrix, 2)
    return reconstructed_matrix

if __name__ == "__main__":
    # 读取矩阵尺寸
    while True:
        size_line = sys.stdin.readline().strip()
        if size_line:
            break
    m_str, n_str = size_line.strip().split()
    m = int(m_str)
    n = int(n_str)
    # 检查矩阵尺寸
    if m <= 0 or n <= 0:
        raise ValueError("Matrix dimensions must be positive integers.")
    # 读取矩阵
    matrix = read_matrix(m, n)
    # 读取 k 值
    while True:
        k_line = sys.stdin.readline().strip()
        if k_line:
            break
    k = int(k_line)
    # 检查 k 值
    if k <= 0 or k > min(m, n):
        raise ValueError("k must be a positive integer less than or equal to min(m, n).")
    # 进行矩阵重构
    reconstructed_matrix = svd_reconstruct(matrix, k)
    # 输出结果
    for row in reconstructed_matrix:
        print(' '.join(map(str, row)))