'''
@Author: dengzaiyong
@Date: 2021-09-21 15:16:08
@LastEditTime: 2021-09-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: 集成意图识别模块、召回模块，排序模块
@FilePath: /JDQA/main.py
'''

import os
import pandas as pd
from intention.business import Intention
from retrieval.hnsw_faiss import HNSW
from ranking.ranker import RANK
import config

def retrieve(k):
    """
    调用意图识别和召回模块
    """
    intention = Intention(data_path=config.train_path,
                            sku_path=config.ware_path,
                            model_path=config.fasttext_path,
                            keywords_path=config.keyword_path)

    hnsw = HNSW(w2v_path=config.w2v_path,
                ef=config.ef_construction,
                M=config.M,
                model_path=config.fassi_path,
                data_path=config.train_path)

    # 验证证和测试集一起作为测试
    dev_set = pd.read_csv(config.root_path + '/data/dev.csv').dropna()
    test_set = pd.read_csv(config.root_path + '/data/test.csv').dropna()
    data_set = dev_set.append(test_set)

    result = pd.DataFrame()

    for query in data_set['custom']:
        query = query.strip()

        # 先进行意图识别
        classify = intention.predict(query)

        # 再进行召回
        if len(query) > 1 and str(classify[0][0]) == '__label__1':  

            # 查询的句子    
            df_query =  pd.DataFrame({'query': [query] * k})

            # 召回的k句相似的句子
            df_retrie =  pd.DataFrame({'retrieved': list(hnsw.search(query, k)['custom'].values)})

            df = pd.concat([df_query, df_retrie], axis=1)
            result = result.append(df, ignore_index=True)

    # 保存召回结果
    result.to_csv(config.root_path + '/result/retrieved.csv', index=False)


def rank():
    """
    调用排序模块
    """
    # 读取召回的结果
    retrieved = pd.read_csv(config.root_path + '/result/retrieved.csv')
    
    ranker = RANK(do_train=False)
    ranked = pd.DataFrame()

    # generate_feature的columns特征格式
    ranked['question1'] = retrieved['query']
    ranked['question2'] = retrieved['retrieved']

    # 得到排序分数
    rank_scores = ranker.predict(ranker.generate_feature(ranked))
    ranked['rank_score'] = rank_scores

    # 保存排序结果
    ranked.to_csv(config.root_path + '/result/ranked.csv', index=False)    

if __name__ == "__main__":
    retrieve(5)
    rank()