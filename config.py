'''
@Author: dengzaiyong
@Date: 2021-08-21 15:16:08
@LastEditTime: 2021-08-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: 项目配置文件
@FilePath: /JDQA/config.py
'''

import sys
import torch
import os

# 项目所在根目录
root_path = os.path.abspath(os.path.dirname(__file__))
temp_path = '/home/opendata/temp/JDQA/'
sys.path.append(root_path)

# 未经处理的原始数据
train_raw = os.path.join(root_path, 'data/chat.txt')
dev_raw = os.path.join(root_path, 'data/开发集.txt')
test_raw = os.path.join(root_path, 'data/测试集.txt')
ware_path = os.path.join(root_path, 'data/ware.txt')
stopwords_path = os.path.join(root_path, 'data/stopwords.txt')

SEP = '[SEP]'

# 处理后的问答数据集
train_path = os.path.join(root_path, 'data/train_no_blank.csv')
dev_path = os.path.join(root_path, 'data/dev.csv')
test_path = os.path.join(root_path, 'data/test.csv')

# 排序模型数据集
rank_train_file =  os.path.join(root_path, 'data/ranking/train.tsv')
rank_dev_file =  os.path.join(root_path, 'data/ranking/dev.tsv')
rank_test_file =  os.path.join(root_path, 'data/ranking/test.tsv')

# 生成模型数据集
generative_train_file =  os.path.join(temp_path, 'data/generative/train.tsv')
generative_dev_file =  os.path.join(temp_path, 'data/generative/dev.tsv')
generative_test_file =  os.path.join(temp_path, 'data/generative/test.tsv')

# 意图识别模型数据集
business_train = os.path.join(root_path, 'data/intention/business.train')
business_test = os.path.join(root_path, 'data/intention/business.test')
keyword_path = os.path.join(root_path, 'data/intention/key_word.txt')

# fasttext模型路径
fasttext_path = os.path.join(root_path, "model/intention/fastext.model")

# lightgbm排序模型数据集
lightgbm_train_file = os.path.join(root_path, "data/ranking/lightgbm_train.csv")
lightgbm_test_file = os.path.join(root_path, "data/ranking/lightgbm_test.csv")

lightgbm_rows_limit = 10000   # lightgbm训练时最大的行数

# Embedding
w2v_path = os.path.join(root_path, "model/retrieval/word2vec.model")
embed_dim = 300  # 词向量维度

# HNSW参数
ef_construction = 3000  # 搜索时保存最近邻的动态列表大小
M = 64                  # 图中最大节点的近邻结点数
fassi_path = os.path.join(temp_path, 'model/retrieval/hnsw_fassi.bin')
hnswlib_path = os.path.join(temp_path, 'model/retrieval/hnsw_hnswlib.bin')

# 通用配置
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

max_seq_len = 103   # 最大序列长度
max_length = 100

base_chinese_bert_vocab = '/home/opendata/temp/JDQA/lib/bert/vocab.txt'

bert_chinese_model_path = '/home/opendata/temp/JDQA/lib/bert/pytorch_model.bin'

batch_size = 32

lr = 1e-5

max_grad_norm = 10

epochs = 30
