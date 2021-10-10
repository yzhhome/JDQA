'''
@Author: dengzaiyong
@Date: 2021-08-21 15:16:08
@LastEditTime: 2021-08-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: 使用Faiss训练hnsw模型
@FilePath: /JDQA/retrieval/hnsw_faiss.py
'''

import os
import time
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import faiss
import config
from preprocessor import clean
from utils.tools import create_logger

logger = create_logger(config.root_path + '/logs/hnsw_faiss.log')

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

    def evaluate(self, vecs):
        '''
        验证模型
        '''
        logger.info('Evaluating hnsw model')
        nq, d = vecs.shape
        t0 = time.time
        
        # 找top1个相似的
        D, I = self.index.search(vecs, 1)
        t1 = time.time

        missing_rate = (I == -1).sum() / float(nq)
        recall_at_1 = (I == np.arange(nq)).sum() / float(nq)
        print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
            (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))
    
    def build_hnsw(self, to_file, ef=2000, m=64):
        """
        训练hnsw模型
        """
        logger.info('building hnsw index')

        # 所有的句向量拼接
        vecs = np.stack(self.data['custom_vec'].values).reshape(-1, config.embed_dim)
        vecs = vecs.astype('float32')

        dim = self.w2v_model.vector_size

         # 构建索引
        index = faiss.IndexHNSWFlat(dim, m)

        # 使用单个GPU资源
        res = faiss.StandardGpuResources()
        
        faiss.index_cpu_to_gpu(res, 0, index)
        index.hnsw.ef_construction = ef
        index.verbose = True
        index.add(vecs)

        # 保存hnsw模型
        faiss.write_index(index, to_file)

        return index

    def load_hnsw(self, model_path):
        logger.info(f"Loading hnsw from {model_path}")
        hnsw = faiss.read_index(model_path)
        return hnsw

    def search(self, text, k=5):
        """
        通过hnsw检索topk
        """
        print(f'Searching for {text}.')

        # 转换句向量
        test_vec = sentence_embedding(clean(text), self.w2v_model)
        test_vec = test_vec.astype('float32')

        # 搜索相似度最高的k个句向量
        D, I = self.index.search(test_vec, k)           

        df = pd.concat((self.data.iloc[I[0]]['custom'].reset_index(),
                        self.data.iloc[I[0]]['assistance'].reset_index(drop=True),
                        pd.DataFrame(D.reshape(-1, 1), columns=['q_distance'])),
                        axis=1) 

        return df

if __name__ == '__main__':
    hnsw = HNSW(config.w2v_path,
            config.ef_construction,
            config.M,
            config.fassi_path,
            config.train_path)

    text = '我要转人工'
    print(hnsw.search(text, k=10))

    # 验证模型
    eval_vecs = np.stack(hnsw.data['custom_vec'].values).reshape(-1, config.embed_dim)
    eval_vecs.astype('float32')
    hnsw.evaluate(eval_vecs[:1000])