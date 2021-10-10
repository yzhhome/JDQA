'''
@Author: dengzaiyong
@Date: 2021-08-21 15:16:08
@LastEditTime: 2021-08-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: 训练tfidf, word2vec, fasttext语言模型
@FilePath: /JDQA/ranking/train_LM.py
'''

import os
from collections import defaultdict
from gensim import models, corpora
import config 
import pandas as pd
import jieba
from utils.tools import create_logger

logger = create_logger(config.root_path + '/logs/train_LM.log')

class Trainer(object):
    def __init__(self):
        self.data = self.data_reader(config.rank_train_file) + \
                    self.data_reader(config.rank_test_file) + \
                    self.data_reader(config.rank_dev_file)
        self.stopwords = open(config.stopwords_path).readlines()
        self.preprocessor()      
        self.train()
        self.saver()

    def data_reader(self, path):
        """
        读取数据集，返回question1和question2所有的句子
        """
        sentences = []
        df = pd.read_csv(path, sep='\t', encoding='utf-8')

        question1 = df['question1'].values
        question2 = df['question2'].values

        sentences.extend(list(question1))
        sentences.extend(list(question2))

        return sentences

    def preprocessor(self):
        """
        分词，并生成计算tfidf需要的数据
        """
        logger.info('loading data...')
        # 对所有句子进行分词
        self.data = [[word for word in jieba.cut(sentence)] for sentence in self.data]

        # 计算每个词出现的次数
        self.freq = defaultdict(int)
        for sentence in self.data:
            for word in sentence:
                self.freq[word] += 1

        # 过滤出现次数小于1的词
        self.data = [[word for word in sentence if self.freq[word] > 1] \
            for sentence in self.data]

        logger.info('building dictionary...')

        # 构建词典
        self.dictionary = corpora.Dictionary(self.data)

        # 保存词典
        self.dictionary.save(config.temp_path + '/model/ranking/ranking.dict')

        # 构建语料库
        self.corpus = [self.dictionary.doc2bow(text) for text in self.data]

        # 语料库序列化保存
        corpora.MmCorpus.serialize(config.temp_path + '/model/ranking/ranking.mm', self.corpus)

    def train(self):
        logger.info('train tfidf model...')
        self.tfidf = models.TfidfModel(self.corpus, normalize=True)

        logger.info('train word2vec model...')
        self.w2v = models.Word2Vec(self.data, 
                                    vector_size=config.embed_dim,
                                    window=2,
                                    min_count=2,
                                    sample=6e-5,
                                    min_alpha=0.0007,
                                    alpha=0.03,
                                    workers=4,
                                    negative=15,
                                    epochs=10)
        self.w2v.build_vocab(self.data)
        self.w2v.train(self.data,
                        total_examples=self.w2v.corpus_count,
                        epochs=15,
                        report_delay=1)

        logger.info('train fasttext model...')
        self.fast = models.FastText(self.data,
                            vector_size=config.embed_dim,
                            window=3,
                            min_count=1,
                            epochs=10,
                            min_n=3,
                            max_n=6,
                            word_ngrams=1)

    def saver(self):
        logger.info(' save tfidf model ...')
        self.tfidf.save(os.path.join(config.temp_path, 'model/ranking/tfidf.model'))
 
        logger.info(' save word2vec model ...')
        self.w2v.save(os.path.join(config.temp_path, 'model/ranking/w2v.model'))
        
        logger.info(' save fasttext model ...')
        self.fast.save(os.path.join(config.temp_path, 'model/ranking/fast.model')) 
        

if __name__ == "__main__":
    Trainer()