import datetime
import os
import tensorflow as tf

start_time = datetime.datetime(2024, 5, 11, 0, 0, 0)
train_days_num = 10
test_days_num = 1

data_path = '/home/web/sunjiyuan/data/essm/v1'

train_files, test_files = [], []
# 加载训练文件
for i in range(train_days_num):
    cur = (start_time + datetime.timedelta(days=i)).strftime('%Y%m%d')
    if not os.path.exists(os.path.join(data_path, cur)):
        print("不存在文件夹")
        continue
    file_list = os.listdir(os.path.join(data_path, cur))
    train_files += [os.path.join(data_path, cur, fn) for fn in file_list]
train_files.sort()

# 加载测试文件
test_start_time = start_time + datetime.timedelta(days=train_days_num)
for i in range(test_days_num):
    cur = (test_start_time + datetime.timedelta(days=i)).strftime('%Y%m%d')
    file_list = os.listdir(os.path.join(data_path, cur))
    test_files += [os.path.join(data_path, cur, fn) for fn in file_list]
test_files.sort()

print(f"添加的文件数量：训练集 {len(train_files)} 测试集 {len(test_files)}")

dataset = tf.data.TFRecordDataset(train_files)
print(dataset)
