'''
@Author: dengzaiyong
@Date: 2021-08-21 15:16:08
@LastEditTime: 2021-08-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: 训练word2vec词向量
@FilePath: /JDQA/retrieval/word2vec.py
'''

import multiprocessing
from time import time
import os
from jieba.posseg import cut
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from preprocessor import clean, read_file
from utils.tools import create_logger
import config

logger = create_logger(config.root_path + '/logs/word2vec.log')

def read_data(file_path):
    """
    读取数据并清洗
    """
    train = pd.DataFrame(read_file(file_path, True),
                         columns=['session_id', 'role', 'content'])
    train['clean_content'] = train['content'].apply(clean)
    return train

def train_w2v(train, to_file):

    # 所有有句子
    sent = [row.split() for row in train['clean_content']]

    phrases = Phrases(sent, min_count=5, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[sent]

    # cpu的核数
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=2,
                         window=2,
                         vector_size=config.embed_dim,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=15,
                         workers=cores - 1,
                         epochs=10)

    t = time()
    w2v_model.build_vocab(sentences)
    logger.info('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()
    w2v_model.train(sentences,
                    total_examples=w2v_model.corpus_count,
                    epochs=15,
                    report_delay=1)
    logger.info('Time to train vocab: {} mins'.format(round((time() - t) / 60, 2)))

    if not os.path.exists(os.path.dirname(to_file)):
        os.mkdir(os.path.dirname(to_file))

    w2v_model.save(to_file)

    logger.info('train word2vec finished.')
    

if __name__ == "__main__":
    train = read_data(config.train_raw)
    train_w2v(train, config.w2v_path)