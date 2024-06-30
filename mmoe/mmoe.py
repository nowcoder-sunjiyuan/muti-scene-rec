import keras.src.layers

from data_process import data_process
import numpy as np
import tensorflow as tf
import utils.nn_utils as nn

from keras import layers

## 可以去看 dataset 的 jyputer
dataset = data_process.train_test_dataset(1024)

l2_reg = 0.0000025
L2REG = keras.regularizers.L2(l2_reg)

TAXONOMY1_VOC = ["<nan>", "10010", "10011", "10012", "10013", "10014", "10015", "10053", "10064", "10016", "-1"]
TAXONOMY2_VOC = ["<nan>", "10017", "10018", "10019", "10020", "10061", "10054", "10021", "10022", "10023", "10024",
                 "10025", "10026", "10034", "10035", "10067", "10186", "10030", "10031", "10032", "10060", "10066",
                 "10086", "10121", "10065", "10187", "10059", "10033", "10057", "10058", "10080", "10081", "10082",
                 "10079", "10078", "10075", "10051", "10083", "10047", "10048", "10072", "10049", "10050", "10074",
                 "10071", "10073", "10122", "10055", "10056", "-1"]
CAREER_JOB1_2_VOC = ["<nan>", "11200", "11201", "11202", "11203", "11204", "11205", "11206", "11207", "11208",
                     "11209", "11210", "11211", "11212", "11213", "11214", "11215", "11216", "11217", "11218",
                     "11219", "11220", "11221", "11222", "11223", "11224", "11225", "11240", "11260", "11265",
                     "11266", "142700", "142961", "143069", "143743", "143789", "143790", "143802", "143811",
                     "143818", "143833", "143843", "143846", "143849", "143850", "143883", "143895", "143909",
                     "-1"]
CAREER_JOB1_1_VOC = ["<nan>", "11226", "11227", "11228", "11229", "11230", "11231", "11232", "11233", "11264", "142960",
                     "143750", "143848", "143882", "-1"]
SCHOOL_TYPE_VOC = ["<nan>", "211", "985", "一本", "二本", "初高中", "双一流", "海外", "海外QS_TOP100", "其他"]
GENDER_VOC = ["<nan>", "其他", "男", "女"]

# 字符串类特征
take_dataset = dataset.take(1)
[features] = take_dataset

# (1,)
gender_col = features['gender']
gender_emb = nn.string_lookup_embedding(
    inputs=gender_col,
    voc_list=GENDER_VOC,
    embedding_dimension=16,
    embedding_regularizer=L2REG, embedding_initializer='glorot_normal', name='gender')

# hash类特征 (none, 5)
st_company_col = features['short_term_companies']
st_company_emb_origin = nn.hash_lookup_embedding(st_company_col, 10000, 16,
                                                 L2REG, 'glorot_normal', 'short_term_companies')
# softmax应用在最后一个维度上
st_companies_weights = tf.expand_dims(tf.nn.softmax(features["short_term_companies_weights"]), axis=-1)  # (None,5,1)
st_companies_emb = tf.reduce_sum(st_company_emb_origin * st_companies_weights, axis=1) # (None, 5, 16)
print(st_companies_emb)