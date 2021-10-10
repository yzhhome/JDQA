'''
@Author: dengzaiyong
@Date: 2021-08-21 15:16:08
@LastEditTime: 2021-08-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: 将数据处理成bert模型可用的数据
@FilePath: /JDQA/ranking/dataprocess.py
'''

import pandas as pd
import torch
from torch.utils.data import Dataset
import config

class DataProcessForSentence(Dataset):
    def __init__(self, 
                tokenizer,         # bert tokenizer
                file_path,         # 数据集文件路径
                max_seq_len=103):  # 最大序列长度 

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # 将数据集进行bert tokenize，以便输入到bert模型
        self.seqs, self.seq_masks, self.seq_segments, self.labels = \
            self.tokenize(file_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx], self.labels[idx]

    def tokenize(self, file_path):
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')

        # 去除句子中的空格
        df['question1'] = df['question1'].apply(lambda x: "".join(x.split()))
        df['question2'] = df['question2'].apply(lambda x: "".join(x.split()))
        labels = df['label'].astype('int8').values

        # 用bert tokenizer进行分词
        tokens_seq_1 = list(map(self.tokenizer.tokenize, df['question1'].values))
        tokens_seq_2 = list(map(self.tokenizer.tokenize, df['question2'].values))

        result = list(map(self.truncate_and_pad, tokens_seq_1, tokens_seq_2))

        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]

        return torch.Tensor(seqs).type(torch.long),        \
                torch.Tensor(seq_masks).type(torch.long),   \
                torch.Tensor(seq_segments).type(torch.long), \
                torch.Tensor(labels).type(torch.long)

    def truncate_and_pad(self, tokens_seq_1, tokens_seq_2):
        """
        对两个句子的token相加，超过max_seq_len的部分进行截断，不足的部分进行pad初0
        """ 
        # 对超过max_seq_len的进行截断
        # (max_seq_len - [CLS] - [SEP] - [SEP]) // 2
        if len(tokens_seq_1) > ((self.max_seq_len - 3) // 2):
            tokens_seq_1 = tokens_seq_1[0:(self.max_seq_len -3) // 2]

        if len(tokens_seq_2) > ((self.max_seq_len - 3) // 2):
            tokens_seq_2 = tokens_seq_2[0:(self.max_seq_len -3) // 2]        

        # 拼接[CLS]和[SEP]
        seq = ['[CLS]'] + tokens_seq_1 + ['[SEP]'] + tokens_seq_2 + ['[SEP]']

        # 用0和1分别代表前后两个句子
        # 第一个句子的长度为 len(tokens_seq_1) + [CLS] + [SEP]
        # 第二个句子的长度为 len(tokens_seq_1) + [SEP]
        seq_segment = [0] * (len(tokens_seq_1) + 2) + [1] * (len(tokens_seq_2) + 1)

        # ID化
        seq = self.tokenizer.convert_tokens_to_ids(seq)

        # 计算需要padding的数量
        padding = [0] * (self.max_seq_len - len(seq))

        # seq_mask有效的seq为1, 无效的补0
        seq_mask = [1] * len(seq) + padding

        # seq_segment进行padding
        seq_segment = seq_segment + padding

        # seq 进行padding
        seq = seq + padding

        # 确保序列长度正确
        assert len(seq) == self.max_seq_len
        assert len(seq_mask) == self.max_seq_len
        assert len(seq_segment) == self.max_seq_len

        return seq, seq_mask, seq_segment