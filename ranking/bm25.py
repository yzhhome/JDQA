'''
@Author: dengzaiyong
@Date: 2021-08-21 15:16:08
@LastEditTime: 2021-08-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: bm25算法实现
@FilePath: /JDQA/ranking/bm25.py
'''

import math
from collections import Counter
import csv
import jieba
import jieba.posseg as pseg
import numpy as np
import pandas as pd
import joblib
import config

class BM25(object):
    def __init__(self, 
                do_train=True,
                save_path=config.root_path + '/model/ranking/'):

        # 计算所有词的idf并保存
        if do_train:
            self.data = pd.read_csv(config.rank_train_file, 
                sep='\t', header=None, quoting=csv.QUOTE_NONE, 
                names=['question1', 'question2', 'target'])  
                              
            self.idf, self.avgdl = self.get_idf()
            self.saver(save_path)
        # 加载之前保存的idf和avgdl
        else:
            self.stopwords = self.load_stop_word()
            self.load(save_path)

    def load_stop_word(self):
        """
        加载停用词
        """
        stopwords = open(config.stopwords_path, 'r', encoding='utf-8').readlines()
        stopwords = [w.strip() for w in stopwords]
        return stopwords

    def tf(self, word, count):
        """
        计算tf, 词频 / 文本长度
        """
        return count[word] / sum(count.values())

    def n_containing(self, word, count_list):
        """
        统计词在所有文档出现的次数
        """
        return sum(1 for count in count_list if word in count)

    def cal_idf(self, word, count_list):
        """
        计算idf, log(文档总数 / 词在所有文档出现的次数)
        """
        return math.log(len(count_list)) / (1 + self.n_containing(word, count_list))

    def get_idf(self):
        """
        计算每个词的idf, 返回idf和文档平均长度
        """

        # 对question2每行进行分词
        self.data['question2'] = self.data['question2'].apply(lambda x: " ".join(jieba.cut(x)))

        # 统率每个词在question2出现的次数
        idf = Counter([y for x in self.data['question2'].tolist() for y in x.split()])

        # 计算每个词的idf
        idf = {k: self.cal_idf(k, self.data['question2'].tolist()) for k, v in idf.items()}

        # 求文档平均长度
        avgdl = np.array([len(x.split()) for x in self.data['question2'].tolist()]).mean()

        return idf, avgdl

    def saver(self, save_path):
        joblib.dump(self.idf, save_path + 'bm25_idf.bin')
        joblib.dump(self.avgdl, save_path + 'bm25_avgdl.bin')

    def load(self, save_path):
        self.idf = joblib.load(save_path + 'bm25_idf.bin')
        self.avgdl = joblib.load(save_path + 'bm25_avgdl.bin')

    def bm_25(self, q, d, k1=1.2, k2=200, b=0.75):
        """
        按bm25公式计算query和doucument的相似度分数
        """
        stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']

        # 切分查询式
        words = pseg.cut(q)  
        fi = {}
        qfi = {}
        for word, flag in words:
            if flag not in stop_flag and word not in self.stopwords:
                # 词在文档出现次数
                fi[word] = d.count(word)
                
                # 词在Query中出现次数
                qfi[word] = q.count(word)

        # 计算K值
        K = k1 * (1 - b + b * (len(d) / self.avgdl))  
        ri = {}
        for key in fi:
            # 计算R
            ri[key] = fi[key] * (k1+1) * qfi[key] * (k2+1) / ((fi[key] + K) * (qfi[key] + k2))  

        score = 0
        for key in ri:
            score += self.idf.get(key, 20.0) * ri[key]
        return score

if __name__ =='__main__':
    bm25 = BM25(do_train=True)