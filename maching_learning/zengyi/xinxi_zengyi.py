import math

def calculate_entropy(data):
    label_counts = {}
    for item in data:
        label = item[-1]
        label_counts[label] = label_counts.get(label, 0) + 1
    entropy = 0.0
    total = len(data)
    for count in label_counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def split_dataset(data, feature_index):
    subsets = {}
    for item in data:
        key = item[feature_index]
        if key not in subsets:
            subsets[key] = []
        subsets[key].append(item)
    return subsets

def calculate_information_gain(data, feature_index):
    base_entropy = calculate_entropy(data)
    subsets = split_dataset(data, feature_index)
    new_entropy = 0.0
    total = len(data)
    for subset in subsets.values():
        weight = len(subset) / total
        subset_entropy = calculate_entropy(subset)
        new_entropy += weight * subset_entropy
    info_gain = base_entropy - new_entropy
    return base_entropy, new_entropy, info_gain

if __name__ == "__main__":
    import sys
    import ast
    lines = sys.stdin.read().splitlines()
    # 去除空行
    lines = [line for line in lines if line.strip()]
    # 最后一行为特征索引，其余为数据集
    feature_index_input = lines[-1]
    data_input = ''.join(lines[:-1])
    data = ast.literal_eval(data_input.strip())
    feature_index = int(feature_index_input.strip())
    base_entropy, new_entropy, info_gain = calculate_information_gain(data, feature_index)
    print(round(base_entropy, 2))
    print(round(new_entropy, 2))
    print(round(info_gain, 2))