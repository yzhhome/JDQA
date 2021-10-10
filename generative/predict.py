'''
@Author: dengzaiyong
@Date: 2021-09-21 15:16:08
@LastEditTime: 2021-09-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: BERT seq2seq模型预测    
@FilePath: /JDQA/generative/predict.py
'''

import os
import torch
from config import is_cuda, temp_path
from generative.bert_model import BertConfig
from generative.seq2seq import Seq2SeqModel
from generative.tokenizer import load_chinese_base_vocab

class BertSeq2Seq(object):
    def __init__(self, model_path, is_cuda):
        self.word2id = load_chinese_base_vocab()
        self.bertconfig = BertConfig(len(self.word2id))
        self.bert_seq2seq = Seq2SeqModel(self.bertconfig)
        self.is_cuda = is_cuda

        # 加载state dict参数
        if is_cuda:
            device = torch.device("cuda")
            self.bert_seq2seq.load_state_dict(torch.load(model_path))
            self.bert_seq2seq.to(device)
        else:
            device = torch.device("cpu")
            checkpoint = torch.load(model_path, map_location=device)
            self.bert_seq2seq.load_state_dict(checkpoint)

        self.bert_seq2seq.eval()

    def generate(self, text, k=5):
        result = self.bert_seq2seq.generate(text, beam_size=k, is_cuda=self.is_cuda)
        return result

if __name__ == '__main__':
    for i in range(29, -1, -1):
        model_path = temp_path + 'model/generative/bert.model.epoch.' + str(i)
        if os.path.exists(model_path):
            break

    if not os.path.exists(model_path):
        print("Not exist trained bert seqseq model")
    else:
        seq2seq = BertSeq2Seq(model_path, is_cuda)
        text = '吃饭 了 吗'
        print(seq2seq.generate(text, k=5))