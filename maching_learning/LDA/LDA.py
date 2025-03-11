import sys
import numpy as np
import re
from collections import defaultdict, Counter

def preprocess_documents(documents):
    processed_docs = []
    vocab = set()
    for doc in documents:
        # 移除标点符号并转换为小写
        doc = re.sub(r'[^\w\s]', '', doc.lower())
        words = doc.strip().split()
        processed_docs.append(words)
        vocab.update(words)
    word2id = {word: idx for idx, word in enumerate(sorted(vocab))}
    id2word = {idx: word for word, idx in word2id.items()}
    return processed_docs, word2id, id2word

def initialize_model(docs, K, V):
    D = len(docs)
    z_dn = []
    n_dk = np.zeros((D, K))  # 矩阵：每个文档下每个主题 权重（其实就是单词数量）
    n_kw = np.zeros((K, V))  # 矩阵：每个主题k下单词的 分配次数
    n_k = np.zeros(K)    # 主题？？？
    for d, doc in enumerate(docs):
        current_doc_topics = []
        for word in doc:
            topic = np.random.randint(0, K)   # 每个单词随机分配一个主题
            current_doc_topics.append(topic)
            word_id = word2id[word]
            n_dk[d, topic] += 1   # 每个文档下的主题权重 ，这里加1代表，这个主题下单词数多了一个
            n_kw[topic, word_id] += 1  # 每个主题下的，单词分配次数
            n_k[topic] += 1    # 这个主题下的单词数量
        z_dn.append(current_doc_topics)
    return z_dn, n_dk, n_kw, n_k

def gibbs_sampling(docs, z_dn, n_dk, n_kw, n_k, K, V, D, alpha, beta, iterations):
    for _ in range(iterations):
        for d, doc in enumerate(docs):
            for i, word in enumerate(doc):
                word_id = word2id[word]
                topic = z_dn[d][i]
                # 减少计数
                n_dk[d, topic] -= 1
                n_kw[topic, word_id] -= 1
                n_k[topic] -= 1
                # 计算概率
                p_z = (n_dk[d] + alpha) * (n_kw[:, word_id] + beta) / (n_k + V * beta)
                p_z /= np.sum(p_z)
                # 采样新主题
                new_topic = np.random.choice(K, p=p_z)
                z_dn[d][i] = new_topic
                # 增加计数
                n_dk[d, new_topic] += 1
                n_kw[new_topic, word_id] += 1
                n_k[new_topic] += 1
    return n_kw

def output_topics(n_kw, id2word, K, M):
    for k in range(K):
        word_counts = n_kw[k]
        top_word_ids = word_counts.argsort()[-M:][::-1]
        top_words = [id2word[i] for i in top_word_ids]
        print(f"Topic {k+1}:")
        print(' '.join(top_words))

if __name__ == "__main__":
    np.random.seed(0)
    #random.seed(0)
    # 读取文档数量
    while True:
        N_line = sys.stdin.readline()
        if N_line.strip():
            break
    N = int(N_line.strip())
    # 读取文档
    documents = []
    for _ in range(N):
        while True:
            doc_line = sys.stdin.readline()
            if doc_line.strip():
                break
        documents.append(doc_line.strip())
    # 预处理文档
    processed_docs, word2id, id2word = preprocess_documents(documents)
    V = len(word2id)
    D = len(processed_docs)  # 文档数
    # 读取LDA参数
    while True:
        params_line = sys.stdin.readline()
        if params_line.strip():
            break
    K_str, T_str, alpha_str, beta_str = params_line.strip().split()
    K = int(K_str)   # 主题数量
    iterations = int(T_str)  # 迭代次数
    alpha = float(alpha_str)
    beta = float(beta_str)
    # 读取M（每个主题的高频词数量）
    while True:
        M_line = sys.stdin.readline()
        if M_line.strip():
            break
    M = int(M_line.strip())
    # 初始化模型
    z_dn, n_dk, n_kw, n_k = initialize_model(processed_docs, K, V)
    # 运行Gibbs采样
    n_kw = gibbs_sampling(processed_docs, z_dn, n_dk, n_kw, n_k, K, V, D, alpha, beta, iterations)
    # 输出主题
    output_topics(n_kw, id2word, K, M)