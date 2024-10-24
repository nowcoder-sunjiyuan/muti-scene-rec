import tensorflow as tf
import utils.nn_utils as nn
from keras import layers, regularizers, models
from data_process.feature_process import career_job_voc

TAXONOMY1_VOC = ["<nan>", "10010", "10011", "10012", "10013", "10014", "10015", "10053", "10064", "10016", "-1"]
TAXONOMY2_VOC = ["<nan>", "10017", "10018", "10019", "10020", "10061", "10054", "10021", "10022", "10023", "10024",
                 "10025", "10026", "10034", "10035", "10067", "10186", "10030", "10031", "10032", "10060", "10066",
                 "10086", "10121", "10065", "10187", "10059", "10033", "10057", "10058", "10080", "10081", "10082",
                 "10079", "10078", "10075", "10051", "10083", "10047", "10048", "10072", "10049", "10050", "10074",
                 "10071", "10073", "10122", "10055", "10056", "-1"]

SCHOOL_TYPE_VOC = ["<nan>", "211", "985", "一本", "二本", "初高中", "双一流", "海外", "海外QS_TOP100", "其他"]
GENDER_VOC = ["<nan>", "其他", "男", "女"]
EDU_WORK_STATUS_VOC = [0, 1, 2]
WORK_YEAR_BOUND = [-3, -2, -1, 0, 1, 3, 5, 10]

REPLY_NUMBER_BOUND = [1, 2, 3, 4, 5, 7, 9, 15, 21, 30, 50, 100, 500]
LIKE_NUMBER_BOUND = [1, 2, 3, 4, 5, 7, 9, 15, 21, 30, 50, 100, 500]
VIEW_NUMBER_BOUND = [1, 5, 10, 15, 30, 50, 100, 150, 200, 300, 500, 750, 1000, 2000, 5000, 10000]

CTR_BOUND = [0.03, 0.05, 0.08, 0.1, 0.15, 0.20, 0.25]
DAY_DELTA_BOUND = [1, 3, 5, 7, 14, 28, 45]

PIT_VAR_BOUND = [20, 40, 60, 80, 100, 150, 200]

EDU_LEVEL_VOC = ["<nan>", "其他", "专科", "博士及以上", "学士", "硕士", "高中及以下"]

SCHOOL_VOC_PATH = "../data_process/school_vocabulary.txt"
SCHOOL_MAJOR_VOC_PATH = "../data_process/school_major_vocabulary.txt"

l2_reg = 0.0000025
L2REG = regularizers.L2(l2_reg)


class FeatureConvertAndEmbModel(tf.keras.layers.Layer):
    """
    "school": ["string", 1],
  "school_major": ["string", 1],
  "work_status_detail": ["int64", 1],
  "career_job1_2": ["string", 1],
  "work_year": ["int64", 1],
  "edu_level": ["string", 1],
  "uid": ["string", 1],
  "manual_career_job_1": ["string", 1],

  "author_uid": ["string", 1],
  "author_career_job1_2": ["string", 1],
  "author_school": ["string", 1],
  "author_school_major": ["string", 1],
  "author_edu_level": ["string", 1],
  "author_work_year": ["int64", 1],
  "entity_id": ["string", 1],
  "platform": ["string", 1]
    """

    def __init__(self, **kwargs):
        """
        :param embedding_dim: 每个特征的嵌入维度。
        """
        super(FeatureEmbModel, self).__init__(**kwargs)

        work_year_input = layers.Input(shape=(1,), dtype='int64', name=f'work_year_input')
        # Embedding层
        embedding_layer = layers.Embedding(
            input_dim=len(WORK_YEAR_BOUND) + 1,
            output_dim=16,
            embeddings_regularizer=L2REG,
            embeddings_initializer='glorot_normal',
            name='work_year_embedding_layer'
        )(work_year_input)
        # Reshape层
        reshape_layer = layers.Reshape((16,))(embedding_layer)
        # 创建并返回一个Model，这个Model本身就是一个Layer
        work_year_emb_layer = models.Model(inputs=work_year_input, outputs=reshape_layer, name=f'work_year_model_layer')

        school_layer = nn.StringLookupEmbeddingLayer(SCHOOL_TYPE_VOC, name="school")
        school_major_layer = nn.StringLookupEmbeddingLayer(SCHOOL_MAJOR_VOC_PATH, name="school_major")
        career_job_layer = nn.StringLookupEmbeddingLayer(career_job_voc, name="career_job")
        uid_layer = nn.HashLookupEmbeddingLayer(num_bins=100000, embedding_dimension=32, name="uid")
        edu_level_layer = nn.StringLookupEmbeddingLayer(EDU_LEVEL_VOC, name="edu_level")

        self.feature_emb_dict = {
            "school": school_layer,
            "school_major": school_major_layer,
            "work_status_detail": nn.HashLookupEmbeddingLayer(num_bins=5, name="work_status_detail"),
            "career_job1_2": career_job_layer,
            "work_year": work_year_emb_layer,
            "edu_level": edu_level_layer,
            "uid": uid_layer,
            "manual_career_job_1": career_job_layer,

            "author_uid": uid_layer,
            "author_career_job1_2": career_job_layer,
            "author_school": school_layer,
            "author_school_major": school_major_layer,
            "author_edu_level": edu_level_layer,
            "author_work_year": work_year_emb_layer,
            "entity_id": nn.HashLookupEmbeddingLayer(num_bins=100000, name="entity_id"),
            "platform": nn.StringLookupEmbeddingLayer(["<nan>", "iOS, Android", "web"], name="platform")
        }

    def call(self, input_dict):
        """
        :param input_dict: 字典，键为特征名称，值为该特征的输入张量。
        """
        # for feature_name, layer in self.feature_emb_dict.items():
        #     if feature_name in input_dict:
        #         max_index = tf.reduce_max(input_dict[feature_name])
        #         min_index = tf.reduce_min(input_dict[feature_name])
        #         print(f"{feature_name} max index: {max_index}, min index: {min_index}")
        #         # 检查是否超出预期范围
        #         assert max_index < layer.input_dim, f"{feature_name} layer's input_dim is too small."

        embedded_features = []
        # 遍历特征嵌入字典
        for feature_name, layer in self.feature_emb_dict.items():
            # 检查输入字典中是否存在当前特征
            if feature_name in input_dict:
                # 应用对应的嵌入层
                embedded_feature = layer(input_dict[feature_name])
                # 将嵌入层的输出添加到列表中
                embedded_features.append(embedded_feature)
        # 检查是否有嵌入特征
        if not embedded_features:
            raise ValueError("输入字典中没有找到任何匹配的特征。")
        # 将所有嵌入层的输出沿最后一个维度拼接
        # concatenated_features = tf.concat(embedded_features, axis=-1)
        concatenated_features = layers.concatenate(embedded_features)
        return concatenated_features

    def compute_output_shape(self, input_shape):
        """
        假设所有输入特征的形状都是(batch_size, 1)，每个特征转化为16维向量。
        """
        # 假设所有特征嵌入后的维度都是16，计算总维度
        total_embedding_dim = sum(layer.output_shape[-1] for layer in self.feature_emb_dict.values())
        # 返回模型的输出形状
        return (input_shape[0], total_embedding_dim)


class FeatureEmbModel(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        """
        :param embedding_dim: 每个特征的嵌入维度。
        """
        super(FeatureEmbModel, self).__init__(**kwargs)

        school_layer = layers.StringLookup(vocabulary=SCHOOL_VOC_PATH, name=f"school_lookup_layer")
        school_major_layer = layers.StringLookup(vocabulary=SCHOOL_MAJOR_VOC_PATH, name=f"school_major_lookup_layer")
        career_job_layer = layers.StringLookup(vocabulary=career_job_voc, name=f"career_job_lookup_layer")
        uid_layer = layers.Hashing(num_bins=100000, name=f"uid_hash_layer")
        edu_level_layer = layers.StringLookup(vocabulary=EDU_LEVEL_VOC, name=f"edu_level_lookup_layer")
        entity_id_layer = layers.Hashing(num_bins=100000, name=f"entity_layer")
        # 注意转换的时候，<nan>是1，iOS是2，Android是3，web是4
        platform_layer = layers.StringLookup(vocabulary=["<nan>", "iOS", "Android", "web"], name=f"platform_lookup_layer")

        self.feature_parse_dict = {
            "school": school_layer,
            "school_major": school_major_layer,
            "work_status_detail": None,
            "career_job1_2": career_job_layer,
            "work_year": None,
            "edu_level": edu_level_layer,
            "uid": uid_layer,
            "manual_career_job_2": career_job_layer,

            "author_uid": uid_layer,
            "author_career_job1_2": career_job_layer,
            "author_school": school_layer,
            "author_school_major": school_major_layer,
            "author_edu_level": edu_level_layer,
            "author_work_year": None,
            "entity_id": entity_id_layer,
            "platform": platform_layer
        }

        self.feature_range_dict = {
            "school": len(school_layer.get_vocabulary()) + 1,
            "school_major": len(school_major_layer.get_vocabulary()) + 1,
            "work_status_detail": 5,
            "career_job1_2": len(career_job_layer.get_vocabulary()) + 1,
            "work_year": len(WORK_YEAR_BOUND) + 1,
            "edu_level": len(edu_level_layer.get_vocabulary()) + 1,
            "uid": 100000,
            "manual_career_job_2": len(career_job_layer.get_vocabulary()) + 1,

            "author_uid": 100000,
            "author_career_job1_2": len(career_job_layer.get_vocabulary()) + 1,
            "author_school": len(school_layer.get_vocabulary()) + 1,
            "author_school_major": len(school_major_layer.get_vocabulary()) + 1,
            "author_edu_level": len(edu_level_layer.get_vocabulary()) + 1,
            "author_work_year": len(WORK_YEAR_BOUND) + 1,
            "entity_id": 100000,
            "platform": len(platform_layer.get_vocabulary()) + 1
        }

        school_emb = layers.Embedding(input_dim=len(school_layer.get_vocabulary()) + 1, output_dim=16,
                                      embeddings_regularizer=L2REG, embeddings_initializer='glorot_normal',
                                      name=f"school_embedding_layer")
        school_major_emb = layers.Embedding(input_dim=len(school_major_layer.get_vocabulary()) + 1, output_dim=16,
                                            embeddings_regularizer=L2REG, embeddings_initializer='glorot_normal',
                                            name=f"school_major_embedding_layer")
        work_status_emb = layers.Embedding(input_dim=5, output_dim=16, embeddings_initializer='glorot_normal',
                                           embeddings_regularizer=L2REG)
        career_job_emb = layers.Embedding(input_dim=len(career_job_layer.get_vocabulary()) + 1, output_dim=16,
                                          embeddings_regularizer=L2REG, embeddings_initializer='glorot_normal',
                                          name=f"career_job_embedding_layer")
        work_year_emb = layers.Embedding(input_dim=len(WORK_YEAR_BOUND) + 1, output_dim=16,
                                         embeddings_regularizer=L2REG, embeddings_initializer='glorot_normal',
                                         name='work_year_embedding_layer')
        edu_level_emb = layers.Embedding(input_dim=len(edu_level_layer.get_vocabulary()) + 1, output_dim=16,
                                         embeddings_regularizer=L2REG, embeddings_initializer='glorot_normal',
                                         name=f"edu_level_embedding_layer")
        uid_emb = layers.Embedding(input_dim=100000, output_dim=32, embeddings_initializer='glorot_normal',
                                   embeddings_regularizer=L2REG)
        entity_id_emb = layers.Embedding(input_dim=100000, output_dim=32, embeddings_initializer='glorot_normal',
                                         embeddings_regularizer=L2REG)
        platform_emb = layers.Embedding(input_dim=len(platform_layer.get_vocabulary()) + 1, output_dim=16,
                                        embeddings_regularizer=L2REG, embeddings_initializer='glorot_normal',
                                        name=f"platform_embedding_layer")
        self.emb_layers = {
            "school": school_emb,
            "school_major": school_major_emb,
            "work_status_detail": work_status_emb,
            "career_job1_2": career_job_emb,
            "work_year": work_year_emb,
            "edu_level": edu_level_emb,
            "uid": uid_emb,
            "manual_career_job_2": career_job_emb,

            "author_uid": uid_emb,
            "author_career_job1_2": career_job_emb,
            "author_school": school_emb,
            "author_school_major": school_major_emb,
            "author_edu_level": edu_level_emb,
            "author_work_year": work_year_emb,
            "entity_id": entity_id_emb,
            "platform": platform_emb
        }

    def _process_features(self, input_dict, layers_dict):
        """
        通用特征处理方法，根据提供的层字典处理输入字典中的特征。
        """
        result_dict = {}
        for feature_name, feature_value in input_dict.items():
            # 获取对应的处理层，如果没有对应层，则直接返回输入值
            layer = layers_dict.get(feature_name)
            if layer is not None:
                result_dict[feature_name] = layer(feature_value)
            else:
                result_dict[feature_name] = feature_value
        return result_dict

    def call(self, input_dict, mode="lookup"):
        if mode == "lookup":
            return self._process_features(input_dict, self.feature_parse_dict), self.feature_range_dict
        elif mode == "embedding":
            temp_dict = self._process_features(input_dict, self.emb_layers)
            # 在拼接前先进行reshape
            reshaped_features = [tf.reshape(feature, (tf.shape(feature)[0], -1)) for feature in temp_dict.values()]
            concatenated_features = tf.keras.layers.concatenate(reshaped_features, axis=-1)
            return concatenated_features

