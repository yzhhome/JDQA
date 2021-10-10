'''
@Author: dengzaiyong
@Date: 2021-09-21 15:16:08
@LastEditTime: 2021-09-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: BERT seq2seq模型训练     
@FilePath: /JDQA/generative/train.py
'''

import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
import pandas as pd
import numpy as np
import os
import sys
from generative.bert_model import BertConfig
from generative.seq2seq import Seq2SeqModel
import time
from torch.utils.data import Dataset, DataLoader
from generative.tokenizer import Tokenizer, load_chinese_base_vocab
from utils.tools import create_logger, collate_fn, read_corpus
from config import root_path, max_length, max_grad_norm, temp_path
from config import lr, bert_chinese_model_path, batch_size
from sklearn.metrics import accuracy_score
from torch.nn.utils import clip_grad_norm_
import config

logger = create_logger(root_path + '/logs/generative_train.log')

class SelfDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.sent_src, self.sent_tgt = read_corpus(path)
        self.word2id = load_chinese_base_vocab()
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.tokenizer = Tokenizer(self.word2id)

    def __getitem__(self, i):
        src = self.sent_src[i] if len(self.sent_src[i]) < max_length else \
                self.sent_src[i][:max_length]
        tgt = self.sent_tgt[i] if len(self.sent_tgt[i]) < max_length else \
                self.sent_tgt[i][:max_length]

        token_ids, token_type_ids = self.tokenizer.encode(src, tgt)
        output = {"token_ids": token_ids, "token_type_ids": token_type_ids}
        return output

    def __len__(self):
        return len(self.sent_src)

class Trainer(object):
    def __init__(self):
        self.pretrain_model_path = bert_chinese_model_path
        self.batch_size = batch_size
        self.lr = lr
    
        logger.info("加载字典")
        self.word2id = load_chinese_base_vocab()

        self.device = config.device
        logger.info('using device:{}'.format(self.device))

        bertconfig = BertConfig(vocab_size=len(self.word2id))

        logger.info("初始化Bert模型")
        self.bert_model = Seq2SeqModel(config=bertconfig)

        logger.info("加载预训练模型")
        self.load_model(self.bert_model, self.pretrain_model_path)   

        self.bert_model.to(self.device)

        logger.info("初始化模型参数")
        self.optim_parameters = list(self.bert_model.parameters())
        self.init_optimizer(lr=lr)

        logger.info("加载训练数据集")
        train_data = SelfDataset(config.generative_train_file)
        self.trainloader = DataLoader(train_data, batch_size=self.batch_size, 
            shuffle=True, collate_fn=collate_fn)

        logger.info("加载验证数据集")
        dev_data = SelfDataset(config.generative_dev_file)    
        self.devloader = DataLoader(dev_data, batch_size=self.batch_size, 
            shuffle=True, collate_fn=collate_fn)

    def init_optimizer(self, lr):
        self.optimizer = Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)

    def load_model(self, model, pretrain_model_path):
        checkpoint = torch.load(self.pretrain_model_path)
        checkpoint = {k[5:]: v for k, v in checkpoint.items()
                if k[:4] == "bert" and "pooler" not in k}
        model.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        logger.info("{} loaded!".format(pretrain_model_path))                

    def train(self, epoch):
        logger.info("Start training epoch: {}".format(epoch))
        # 训练模式
        self.bert_model.train()
        self.iteration(epoch, self.trainloader)
        logger.info("Epoch {} Train finished".format(epoch))

    def iteration(self, epoch, dataloader):
        total_loss = 0.0
        batch_count = 0

        # 训练开始时间
        start_time = time.time()
        for batch_idx, data in enumerate(tqdm(dataloader, position=0, leave=True)):
            self.optimizer.zero_grad()           

            token_ids, token_type_ids, target_ids = data
            batch_count += len(token_ids)

            token_ids = token_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            enc_layers, logits, loss, attention_layers = self.bert_model(token_ids,
                token_type_ids, labels=target_ids)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()

            if batch_idx % 10 == 0: 
                epoch_loss = total_loss / (batch_idx + 1)
                logger.info(f"epoch: {epoch}, batch: {batch_idx}, train loss: {round(epoch_loss, 3)}")  

            # 梯度裁剪
            clip_grad_norm_(self.bert_model.parameters(), max_grad_norm)

        end_time = time.time()
        spend_time = end_time - start_time
        logger.info(f"epoch: {epoch}, train spend time: {spend_time}")
        self.save_state_dict(self.bert_model, epoch)

    def evaluate(self, epoch):
        logger.info("start evaluating model")
        self.bert_model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.devloader,position=0, leave=True)):
                token_ids, token_type_ids, target_ids = data

                token_ids = token_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                target_ids = target_ids.to(self.device)  

                enc_layers, logits, loss, attention_layers = self.bert_model(token_ids,
                    token_type_ids, labels=target_ids)
                total_loss += loss.item()

                if batch_idx % 10 == 0: 
                    epoch_loss = total_loss / (batch_idx + 1)
                    logger.info(f"epoch: {epoch}, batch: {batch_idx}, valid loss: {round(epoch_loss, 3)}")                     

    def save_state_dict(self, model, epoch, 
        file_path = temp_path + '/model/generative/bert.model'):
        save_path = file_path + '.epoch.{}'.format(str(epoch))
        torch.save(model.state_dict(), save_path)
        logger.info("{} saved!".format(save_path))

if __name__ == '__main__':
    trainer = Trainer()

    for epoch in range(config.epochs):
        trainer.train(epoch)
        trainer.evaluate(epoch)