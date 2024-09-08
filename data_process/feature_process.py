import utils.base_tool as base_tool
import random
import tensorflow as tf
import os

career_job_voc = ["<nan>", "11213", "144158", "11210", "11222", "11240", "144152", "142961", "144034", "143802",
                  "144377", "11201", "143789", "143895", "144198", "144214", "11200", "144110", "143909", "144310",
                  "144313", "11202", "143811", "143790", "11219", "144068", "143849", "144341", "144409", "11217",
                  "144194", "144092", "11266", "11265", "11204", "144241", "144234", "144335", "11208", "11209",
                  "11224", "11203", "11212", "144173", "143833", "144011", "143883", "143843", "11207", "144396",
                  "142700", "11206", "143818", "11220", "11221", "144287", "143743", "144400", "11223", "143850",
                  "11216", "11214", "144001", "144330", "144302", "11205", "11211", "143069", "144270", "144053",
                  "11218", "143846", "11225", "144277", "144185", "11260", "11215", "144246", "144047", "144181",
                  "144101"]
edu_level_voc = ["<nan>", "其他", "专科", "博士及以上", "学士", "硕士", "高中及以下"]
platform_voc = ["<nan>", "Android", "IOS", "web"]

# 获取当前脚本文件的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录
current_dir = os.path.dirname(current_script_path)
# 构建school_vocabulary.txt的绝对路径
school_voc_file_path = os.path.join(current_dir, 'school_vocabulary.txt')
school_major_voc_file_path = os.path.join(current_dir, 'school_major_vocabulary.txt')
school_voc = base_tool.read_vocabulary(school_voc_file_path)
school_major_voc = base_tool.read_vocabulary(school_major_voc_file_path)

'''
"school": 从school_vocabulary.txt文件中随机选择
"school_major": 从school_major_vocabulary.txt文件中随机选择
"work_status_detail": 从 0 - 4中选择
"career_job1_2": career_job_voc 中随机选择
"work_year": 0 - 8之间随机选择
"edu_level": edu_level_voc中随机选择
"uid": 随机的9位数字的字符串
"manual_career_job_1": career_job_voc 中随机选择,

"author_uid": 随机的9位数字的字符串,
"author_career_job1_2": career_job_voc 中随机选择,
"author_school": 从school_vocabulary.txt文件中随机选择
"author_school_major": 从school_major_vocabulary.txt文件中随机选择
"author_edu_level": edu_level_voc中随机选择
"author_work_year": 0 - 8之间随机选择
"entity_id": 74000拼接上随机的六位字符串
"platform": platform_voc随机选择
'''


def generate_random_tensor_from_list(value_list, exclude_value=None, dtype=tf.string):
    if exclude_value is not None:
        value_list = [value for value in value_list if value != exclude_value.numpy()[0].decode("utf-8")]
    random_value = random.choice(value_list)
    return tf.constant([random_value], dtype=dtype)


def generate_random_tensor_from_list_tf(value_list, exclude_value=None, dtype=tf.string):
    # 转换 value_list 为 Tensor
    value_list_tensor = tf.constant(value_list, dtype=dtype)

    # 如果有排除值，我们需要过滤掉这个值
    if exclude_value is not None:
        # 创建一个掩码，标记出所有不等于 exclude_value 的值
        mask = tf.not_equal(value_list_tensor, exclude_value)
        # 使用掩码过滤 value_list_tensor
        filtered_values = tf.boolean_mask(value_list_tensor, mask)
    else:
        filtered_values = value_list_tensor

    # 从过滤后的列表中随机选择一个索引
    random_index = tf.random.uniform(shape=[], maxval=tf.size(filtered_values), dtype=tf.int32)
    # 使用随机索引从列表中获取一个值
    random_value = tf.gather(filtered_values, random_index)
    # 将选择的值包装成一个张量返回
    return tf.reshape(random_value, [1])


def generate_random_int_tensor(min_value, max_value):
    random_value = random.randint(min_value, max_value)
    return tf.constant([random_value], dtype=tf.int64)


def generate_random_string_tensor(length):
    random_value = ''.join(random.choices('0123456789', k=length))
    return tf.constant([random_value], dtype=tf.string)


def generate_random_int_tensor_tf(batch_size, min_value, max_value):
    random_value = tf.random.uniform(shape=[batch_size,], minval=min_value, maxval=max_value, dtype=tf.int64)
    return tf.reshape(random_value, [batch_size, 1])


def generate_random_string_tensor_tf(length):
    random_values = tf.strings.as_string(tf.random.uniform(shape=[length], minval=0, maxval=10, dtype=tf.int32))
    random_string = tf.strings.reduce_join(random_values)
    return tf.reshape(random_string, [1])
