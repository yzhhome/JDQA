'''
@Author: dengzaiyong
@Date: 2021-08-21 15:16:08
@LastEditTime: 2021-08-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: 使用hnswlib训练hnsw模型
@FilePath: /JDQA/retrieval/hnsw_hnswlib.py
'''

import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import hnswlib
import config
from preprocessor import clean
from utils.tools import create_logger

logger = create_logger(config.root_path + '/logs/hnsw_hnswlib.log')

def sentence_embedding(sentence, w2v_model):
    '''
    通过词向量均值的方式生成句向量
    sentence: 待生成句向量的句子
    w2v_model: word2vec模型
    return: 句子中所有词向量的均值
    '''
    embedding = []
    for word in clean(sentence).split():
        if word not in w2v_model.wv.index_to_key:
            embedding.append(np.random.randn(1, config.embed_dim))
        else:
            embedding.append(w2v_model.wv.get_vector(word))

    # 所有词向量的均值为句向量
    return np.mean(np.array(embedding), axis=0).reshape(1, -1)

class HNSW(object):
    def __init__(self,
                 w2v_path,                      # word2vec模型路径
                 ef=config.ef_construction,     # 搜索时保存最近邻的动态列表大小
                 M=config.M,                    # 节点的邻结点的数量
                 model_path=None,               # hnsw模型保存路径
                 data_path=None):               # 数据文件路径

        self.w2v_model = KeyedVectors.load(w2v_path)
        self.data = self.load_data(data_path)

        # 加载hnsw模型
        if model_path and os.path.exists(model_path):
            self.index = self.load_hnsw(model_path)            
        # 训练hnsw模型
        elif data_path:
            self.index = self.build_hnsw(model_path, ef=ef, m=M)
        else:
            logger.error('No existing model and no building data provided.')

        
    def load_data(self, data_path):
        '''
        读取数据，并生成句向量        
        data_path：问答pair数据所在路径
        return: 包含句向量的dataframe
        '''
        data = pd.read_csv(data_path)

        # 生成custom每条记录的句向量
        data['custom_vec'] = data['custom'].apply(
            lambda x: sentence_embedding(x, self.w2v_model))

        # 确保句向量的维度为300
        data['custom_vec'] = data['custom_vec'].apply(
            lambda x: x[0][0] if x.shape[1] != config.embed_dim else x)
        data = data.dropna()
        
        return data
    
    def build_hnsw(self, to_file, ef=2000, m=64):
        """
        训练hnsw模型
        """
        logger.info('building hnsw index')

        # 所有的句向量拼接
        vecs = np.stack(self.data['custom_vec'].values).reshape(-1, config.embed_dim)
        vecs = vecs.astype('float32')

        dim = self.w2v_model.vector_size
        num_elements = self.data['custom'].shape[0]

         # 构建索引
        index = hnswlib.Index(space='l2', dim=dim)

        index.init_index(max_elements=num_elements, ef_construction=ef, M=m)
        index.set_ef(10)
        index.set_num_threads(4)
        index.add_items(vecs)

        labels, distances = index.knn_query(vecs, k=1)

        print('labels: ', labels)
        print('distances: ', distances)

        logger.info("Recall:{}".format(np.mean(labels.reshape(-1) == np.arange(len(hnsw)))))

         # 保存hnsw模型
        index.save_index(to_file)  

        return index

    def load_hnsw(self, model_path):
        logger.info(f"Loading hnsw from {model_path}")
        hnsw = hnswlib.Index(space='l2', dim=self.w2v_model.vector_size)
        hnsw.load_index(model_path)
        return hnsw

    def search(self, text, k=5):
        """
        通过hnsw检索topk
        """
        logger.info(f'Searching for {text}.')

        # 转换句向量
        test_vec = sentence_embedding(clean(text), self.w2v_model)

        # 搜索相似度最高的k个句向量
        I, D = self.hnsw.knn_query(test_vec, k=k)

        print('index of top k similarity:\n')
        print(I)

        return pd.concat((self.data.iloc[I[0]]['custom'].reset_index(),
                        self.data.iloc[I[0]]['assistance'].reset_index(drop=True),
                        pd.DataFrame(D.reshape(-1, 1), columns=['q_distance'])),
                        axis=1)

if __name__ == '__main__':
    hnsw = HNSW(config.w2v_path,
            config.ef_construction,
            config.M,
            config.hnswlib_path,
            config.train_path)

    text = '我要转人工'
    print(hnsw.search(text, k=10))