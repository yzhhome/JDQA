'''
@Author: dengzaiyong
@Date: 2021-08-21 15:16:08
@LastEditTime: 2021-08-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: Definition of matching network using BERT.
@FilePath: /JDQA/ranking/matchnn.py
'''

import torch
from torch import nn
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import BertConfig, BertForSequenceClassification
from ranking.dataprocess import DataProcessForSentence
import config

tqdm.pandas()


class BertModelTrain(nn.Module):
    """
    Bert训练类
    """
    def __init__(self):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            config.root_path + '/lib/bert/', num_labels=2)

        self.device = config.device
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, seqs, seq_masks, seq_segments, labels):

        # 送到bert模型输出loss和logits
        output = self.bert(input_ids=seqs,
                                attention_mask=seq_masks,
                                token_type_ids=seq_segments,
                                labels=labels)
        loss, logits = output.loss, output.logits

        # logits归一化，输出概率
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities

class BertModelPredict(nn.Module):
    """
    Bert预测类
    """
    def __init__(self):
        super().__init__()
        bert_config = BertConfig.from_pretrained(
            config.root_path + '/lib/bert/config.json')
            
        self.bert = BertForSequenceClassification(bert_config)
        self.device = config.device

    def forward(self, seqs, seq_masks, seq_segments):
        logits = self.bert(input_ids=seqs,
                            attention_mask=seq_masks,
                            token_type_ids=seq_segments)[0]
                            
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities

class MatchingNN(object):
    """
    使用BertModelPredict类进行预测
    """
    def __init__(self,
                model_path=config.root_path + '/model/ranking/best_bert.model',
                vocab_path=config.root_path + '/lib/bert/vocab.txt',
                data_path=config.rank_train_file,
                is_cuda=config.is_cuda,
                max_seq_len=config.max_seq_len):

        self.model_path = model_path
        self.vocab_path = vocab_path
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.is_cuda = is_cuda
        self.device = config.device
        self.load_model()

    def load_model(self):
        self.model = BertModelPredict().to(self.device)

        # 加载保存的state_dict
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint, strict=False)

        # 预测时使用评估模式
        self.model.eval()

        self.bert_tokenizer = BertTokenizer.from_pretrained(
            config.root_path + '/lib/bert/', do_lower_case=True)

        self.dataprocess = DataProcessForSentence(self.bert_tokenizer, 
                                                    self.data_path, 
                                                    self.max_seq_len)

    def predict(self, q1, q2):

        # 先进行tokenize，再截断和padding
        result = [self.dataprocess.truncate_and_pad(
            self.bert_tokenizer.tokenize(q1), self.bert_tokenizer.tokenize(q2))]

        # input_ids
        seqs = [i[0] for i in result]
        # attention_mask
        seq_masks = [i[1] for i in result]
        # token_type_ids
        seq_segments = [i[2] for i in result]

        # 转换为Tensor
        seqs = torch.Tensor(seqs).type(torch.long)
        seq_masks = torch.Tensor(seq_masks).type(torch.long)
        seq_segments = torch.Tensor(seq_segments).type(torch.long)

        if self.is_cuda:
            seqs = seqs.to(self.device)
            seq_masks = seq_masks.to(self.device)
            seq_segments = seq_segments.to(self.device)
        
        with torch.no_grad():
            output = self.model(seqs, seq_masks, seq_segments)

            # 取probabilities
            res = output[-1].cpu().detach().numpy()
            
            label = res.argmax()
            score = res.tolist()[0][label]
            return label, score