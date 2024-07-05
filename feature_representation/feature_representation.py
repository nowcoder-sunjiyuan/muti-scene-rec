import keras.src.layers

from data_process import data_process
import numpy as np
import tensorflow as tf
import utils.nn_utils as nn
import utils.base_tool as base_tool

from keras import layers

# 可以去看 dataset 的 jupyter
dataset, test_dataset = data_process.train_test_dataset(1024)

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
EDU_WORK_STATUS_VOC = [0, 1, 2]
WORK_YEAR_BOUND = [-3, -2, -1, 0, 1, 3, 5, 10]
EDU_LEVEL_VOC = ["<nan>", "其他", "专科", "博士及以上", "学士", "硕士", "高中及以下"]

SCHOOL_VOC_PATH = "../data_process/school_vocabulary.txt"
SCHOOL_MAJOR_VOC_PATH = "../data_process/school_major_vocabulary.txt"

take_dataset = dataset.take(1)
[(features, labels)] = take_dataset
features = base_tool.MultiIODict(features)


def get_basic_feature_representation():
    # 给所有特征建立 tensor
    inputs = data_process.build_input_tensor()
    not_input_features = "label"
    input_list = [v for k, v in inputs.items() if k not in not_input_features]

    # 最后tensor的结果dict，这是一个可以多进多出的 tensor
    tensor_dict = base_tool.MultiIODict({})

    """
    基本的特征
    """
    # (none, 1)  -> (none, 16) 性别
    tensor_dict['gender', 'author_gender'] = nn.string_lookup_embedding(inputs=inputs['gender', 'author_gender'],
                                                                        voc_list=GENDER_VOC, name='gender')
    # (none, 1)  -> (none, 16) 学校
    tensor_dict['school', 'author_school'] = nn.string_lookup_embedding(inputs=inputs['school', 'author_school'],
                                                                        voc_list=SCHOOL_TYPE_VOC, name='school')
    # (none, 1) -> (none, 16) 专业
    tensor_dict['school_major', 'author_school_major'] = nn.string_lookup_embedding(
        inputs=inputs['school_major', 'author_school_major'],
        voc_list=SCHOOL_MAJOR_VOC_PATH,
        name='school_major'
    )
    # (none, 1) -> (none, 16) 学校类型，211，985
    tensor_dict['school_type', 'author_school_type'] = nn.string_lookup_embedding(
        inputs=inputs['school_type', 'author_school_type'],
        voc_list=SCHOOL_TYPE_VOC,
        name='school_type'
    )
    # (none, 1) -> (none, 16) ??? 工作状态？？暂时不确定
    tensor_dict['edu_work_status', 'author_edu_work_status'] = nn.integer_lookup_embedding(
        inputs=inputs['edu_work_status', 'author_edu_work_status'],
        voc_list=EDU_WORK_STATUS_VOC,
        name='edu_work_status'
    )

    # 一级意向职位 (none, 1) -> (none, 16)
    tensor_dict['career_job1_1', 'author_career_job1_1', 'manual_career_job_1'] = nn.string_lookup_embedding(
        inputs=inputs['career_job1_1', 'author_career_job1_1', 'manual_career_job_1'],
        voc_list=CAREER_JOB1_1_VOC,
        name='career_job1_1'
    )

    # 二级意向职位 (none, 1) -> (none, 16)
    cj2 = ("career_job1_2", "career_job2_2", "career_job3_2", "author_career_job1_2", "manual_career_job_2")
    cj2 = tuple(fea for fea in cj2 if fea in inputs)
    tensor_dict[cj2] = nn.string_lookup_embedding(
        inputs=inputs[cj2],
        voc_list=CAREER_JOB1_1_VOC,
        name='career_job1_2'
    )

    # 三级意向职位 (none, 1) -> (none, 16)
    cj3 = ("career_job1_3", "career_job2_3", "career_job3_3", "author_career_job1_3")
    cj3 = tuple(fea for fea in cj3 if fea in inputs)
    tensor_dict[cj3] = nn.hash_lookup_embedding(
        inputs=inputs[cj3],
        name="career_job1_3",
        num_bins=1000,
    )

    # 工作时间，毕业时间距离今年的距离，可能为负数， workYea已经被处理成区间表示
    work_year_embedding_layer = layers.Embedding(
        input_dim=len(WORK_YEAR_BOUND) + 1,
        output_dim=16,
        embeddings_regularizer=L2REG,
        embeddings_initializer='glorot_normal',
        name='work_year_embedding_layer'
    )
    reshape_layer = layers.Reshape((16,))
    tensor_dict["work_year"] = reshape_layer(work_year_embedding_layer(inputs["work_year"]))
    tensor_dict["author_work_year"] = reshape_layer(work_year_embedding_layer(inputs["author_work_year"]))

    # 学历, 专科，本科，硕士，(none, 1) -> (none, 16)
    tensor_dict["edu_level", "author_edu_level"] = nn.string_lookup_embedding(
        inputs=inputs["edu_level", "author_edu_level"],
        name="edu_level",
        voc_list=EDU_LEVEL_VOC
    )

    """
    5个公司id，和对应的权重，short_term_company (hash) (none, 5) 代表五个公司的id
    """
    st_company_col = features['short_term_companies']
    st_company_emb_origin = nn.hash_lookup_embedding(inputs=inputs['short_term_companies'], num_bins=10000,
                                                     embedding_dimension=16,
                                                     embedding_regularizer=L2REG, embedding_initializer='glorot_normal',
                                                     name='short_term_companies')  # (none, 5, 16)
    # softmax作用在最后一个维度, 从(none, 5) -> (none, 5, 1)
    st_companies_weights = layers.Softmax(axis=-1)(inputs['short_term_companies_weights'])  # (none, 5)
    st_companies_weights = nn.ExpandDimsLayer(-1)(st_companies_weights)  # (None, 5, 1)
    # 将这5个向量加权求和  reduce((None,5,16) * (none,5,1))
    tensor_dict['st_companies_emb'] = nn.ReduceSumLayer(1)(st_company_emb_origin * st_companies_weights)  # (none, 16)
    # st_companies_weights = tf.expand_dims(tf.nn.softmax(features["short_term_companies_weights"]), axis=-1)  # (None,5,1)
    # st_companies_emb = tf.reduce_sum(st_company_emb_origin * st_companies_weights, axis=1) # (none, 16)

    """
    company_keyword (hash) (none, 3) 这个是公司的关键字, 对公司向量求平均
    """
    company_keyword_col = features['company_keyword']
    company_keyword_origin = nn.hash_lookup_embedding(inputs=inputs['company_keyword'], num_bins=10000,
                                                      embedding_dimension=16,
                                                      embedding_regularizer=L2REG,
                                                      embedding_initializer='glorot_normal',
                                                      name='company_keyword')  # (none, 3, 16)
    company_keyword_max_len_col = features['company_keyword_max_len']  # (none, 1)
    tensor_dict['company_keyword_emb'] = nn.ReduceMeanWithMask(3)(company_keyword_origin,
                                                                  inputs['company_keyword_max_len'])  # (none, 16)
    return input_list, tensor_dict


def get_basic_feature_representation_case():

    # 最后tensor的结果dict，这是一个可以多进多出的 tensor
    tensor_dict = base_tool.MultiIODict({})

    tensor_dict['gender', 'author_gender'] = nn.string_lookup_embedding(inputs=features['gender', 'author_gender'],
                                                                        voc_list=GENDER_VOC, name='gender')
    # (none, 1)  -> (none, 16) 学校
    tensor_dict['school', 'author_school'] = nn.string_lookup_embedding(inputs=features['school', 'author_school'],
                                                                        voc_list=SCHOOL_TYPE_VOC, name='school')
    # (none, 1) -> (none, 16) 专业
    tensor_dict['school_major', 'author_school_major'] = nn.string_lookup_embedding(
        inputs=features['school_major', 'author_school_major'],
        voc_list=SCHOOL_MAJOR_VOC_PATH,
        name='school_major'
    )
    # (none, 1) -> (none, 16) 学校类型，211，985
    tensor_dict['school_type', 'author_school_type'] = nn.string_lookup_embedding(
        inputs=features['school_type', 'author_school_type'],
        voc_list=SCHOOL_TYPE_VOC,
        name='school_major'
    )
    # (none, 1) -> (none, 16) ??? 工作状态？？暂时不确定
    tensor_dict['edu_work_status', 'author_edu_work_status'] = nn.integer_lookup_embedding(
        inputs=features['edu_work_status', 'author_edu_work_status'],
        voc_list=EDU_WORK_STATUS_VOC,
        name='edu_work_status'
    )

    # 一级意向职位 (none, 1) -> (none, 16)
    tensor_dict['career_job1_1', 'author_career_job1_1', 'manual_career_job_1'] = nn.string_lookup_embedding(
        inputs=features['career_job1_1', 'author_career_job1_1', 'manual_career_job_1'],
        voc_list=CAREER_JOB1_1_VOC,
        name='career_job1_1'
    )

    # 二级意向职位 (none, 1) -> (none, 16)
    cj2 = ("career_job1_2", "career_job2_2", "career_job3_2", "author_career_job1_2", "manual_career_job_2")
    cj2 = tuple(fea for fea in cj2 if fea in features)
    tensor_dict[cj2] = nn.string_lookup_embedding(
        inputs=features[cj2],
        voc_list=CAREER_JOB1_1_VOC,
        name='career_job1_2'
    )

    # 三级意向职位 (none, 1) -> (none, 16)
    cj3 = ("career_job1_3", "career_job2_3", "career_job3_3", "author_career_job1_3")
    cj3 = tuple(fea for fea in cj3 if fea in features)
    tensor_dict[cj3] = nn.hash_lookup_embedding(
        inputs=features[cj3],
        name="career_job1_3",
        num_bins=1000,
    )
    return tensor_dict
