'''
@Author: dengzaiyong
@Date: 2021-09-21 15:16:08
@LastEditTime: 2021-09-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: Hyper parameters for knowledge distillation    
@FilePath: /JDQA/generative/distill_train.py
'''

import random
from functools import partial
import numpy as np
import torch
import warnings
from textbrewer import DistillationConfig, GeneralDistiller, TrainingConfig
from torch.utils.data import DataLoader, Dataset, dataloader
from tqdm import tqdm
from config import root_path
from generative.bert_model import BertConfig
from generative.optimizer import BERTAdam
from generative.seq2seq import Seq2SeqModel
from generative.tokenizer import Tokenizer, load_chinese_base_vocab
from utils.tools import create_logger, collate_fn, read_corpus
from generative.distill_matches import matches
from generative.distill_config import parse

warnings.filterwarnings('ignore')

logger = create_logger(root_path + '/logs/distill_train.log')

def divide_parameters(named_parameters, lr=None, args=None):
    """
    对参数按 weight decay 进行分组
    """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # 进行weight decay的参数
    decay_parameters_names = list(zip(*[(p,n) for n,p in named_parameters 
        if not any((di in n) for di in no_decay)]))

    # 不进行weight decay的参数
    no_decay_parameters_names = list(zip(*[(p,n) for n,p in named_parameters 
        if any((di in n) for di in no_decay)]))

    param_group = []
    if len(decay_parameters_names)>0:
        decay_parameters, decay_names = decay_parameters_names
        if lr is not None:
            decay_group = {'params':decay_parameters,   
                            'weight_decay_rate': args.weight_decay_rate, 
                            'lr':lr}
        else:
            decay_group = {'params': decay_parameters, 
                            'weight_decay_rate': args.weight_decay_rate}
        param_group.append(decay_group)

    if len(no_decay_parameters_names)>0:
        no_decay_parameters, no_decay_names = no_decay_parameters_names
        if lr is not None:
            no_decay_group = {'params': no_decay_parameters, 
                                'weight_decay_rate': 0.0, 
                                'lr': lr}
        else:
            no_decay_group = {'params': no_decay_parameters, 
                                'weight_decay_rate': 0.0}
        param_group.append(no_decay_group)

    assert len(param_group)>0
    return param_group

class SelfDataset(Dataset):
    def __init__(self, path, max_length) -> None:
        super().__init__()
        self.sent_src, self.sent_tgt = read_corpus(path)
        self.word2id = load_chinese_base_vocab()
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.tokenizer = Tokenizer(self.word2id)
        self.max_length = max_length

    def __getitem__(self, i):
        src = self.sent_src[i] if len(self.sent_src[i]) < self.max_length else \
                self.sent_src[i][:self.max_length]
        tgt = self.sent_tgt[i] if len(self.sent_tgt[i]) < self.max_length else \
                self.sent_tgt[i][:self.max_length]

        token_ids, token_type_ids = self.tokenizer.encode(src, tgt)
        output = {"token_ids": token_ids, "token_type_ids": token_type_ids}
        return output

    def __len__(self):
        return len(self.sent_src)

def main():
    args = parse()
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    logger.info('加载字典')
    word2idx = load_chinese_base_vocab()

    # 判断是否有可用GPU
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.is_cuda else "cpu")

    logger.info('using device:{}'.format(args.device))

    # 定义模型超参数
    bertconfig_T = BertConfig(vocab_size=len(word2idx))
    bertconfig_S = BertConfig(vocab_size=len(word2idx), num_hidden_layers=3)   

    logger.info('初始化Bert模型')
    bert_model_T = Seq2SeqModel(config=bertconfig_T)
    bert_model_S = Seq2SeqModel(config=bertconfig_S)   

    logger.info('加载训练好的Teacher模型')
    load_model(bert_model_T, args.tuned_checkpoint_T) 

    bert_model_T.to(args.device)
    bert_model_T.eval()

    logger.info('加载Student预训练模型')
    if args.load_model_type == 'bert':
        load_model(bert_model_S, args.init_checkpoint_S)
    else:
        logger.info(" Student Model is randomly initialized.")   
    bert_model_S.to(args.device)

    logger.info('加载训练数据')
    train = SelfDataset(args.train_path, args.max_length)
    trainloader = DataLoader(train,
                             batch_size=args.train_batch_size,
                             shuffle=True,
                             collate_fn=collate_fn)    

    if args.do_train:
        logger.info('声明需要优化的参数')

        # 总计训练的batch数量
        num_train_steps = int(len(trainloader) / args.train_batch_size) * args.num_train_epochs

        # student模型需要优化的参数
        optim_parameters = list(bert_model_S.named_parameters())

        # 按weight decay进行分组后的参数
        all_trainable_params = divide_parameters(optim_parameters,
                                                 lr=args.learning_rate,
                                                 args=args)      

        optimizer = BERTAdam(params=all_trainable_params,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps,
                             schedule=args.schedule,
                             s_opt1=args.s_opt1,
                             s_opt2=args.s_opt2,
                             s_opt3=args.s_opt3)

        train_config = TrainingConfig(gradient_accumulation_steps=args.gradient_accumulation_steps,
                                    ckpt_frequency=args.ckpt_frequency,
                                    log_dir=args.output_dir,
                                    output_dir=args.output_dir,
                                    device=args.device)

        intermediate_matches = []
        for match in args.matches:
            intermediate_matches += matches[match]

        distill_config = DistillationConfig(temperature=args.temperature,
                                            intermediate_matches=intermediate_matches)

        def Seq2SeqSimpleAdaptor(batch, model_outputs):
            return {'hidden': model_outputs[0], 'logits': model_outputs[1], 
                    'loss': model_outputs[2], 'attention': model_outputs[3]}

        adaptor_T = partial(Seq2SeqSimpleAdaptor)
        adaptor_S = partial(Seq2SeqSimpleAdaptor)

        distiller = GeneralDistiller(train_config=train_config,
                                     distill_config=distill_config,
                                     model_T=bert_model_T,
                                     model_S=bert_model_S,
                                     adaptor_T=adaptor_T,
                                     adaptor_S=adaptor_S)    

        # 回调函数在每个checkpoint评测模型效果 
        callback_func = partial(predict, model=bert_model_S, data_path=args.dev_path, step=0, args=args)   

        logger.info('Start distillation')
        distiller.train(optimizer,
                        scheduler=None,
                        dataloader=trainloader,
                        num_epochs=args.num_train_epochs,
                        callback=None)
        logger.info('Distill finished')

    # 预测模式
    if not args.do_train and args.do_predict:
        predict(bert_model_S, args.test_path, step=0, args=args)   

def load_model(model, pretrain_model_path):
    """
    加载预训练的模型
    """
    checkpoint = torch.load(pretrain_model_path, map_location=torch.device('cpu'))
    checkpoint = {k[5:]: v for k,v in checkpoint.items() if k[:4] =='bert' and 'pooler' not in k}

    model.load_state_dict(checkpoint, strict=False)
    torch.cuda.empty_cache()
    logger.info("{} loaded!".format(pretrain_model_path))

def predict(model, data_path, step, args):
    logger.info('加载测试数据')
    dev = SelfDataset(data_path, args.max_length)

    logger.info("Start evaluating")
    devloader = DataLoader(dev,
                           batch_size=args.predict_batch_size,
                           shuffle=True,
                           collate_fn=collate_fn)    
    model.eval()
    total_loss = 0.0

    for batch_idx, data in enumerate(tqdm(devloader, desc="Evaluating", disable=None)):        
        token_ids, token_type_ids, target_ids = data
        token_ids = token_ids.to(args.device)
        token_type_ids = token_type_ids.to(args.device)
        target_ids = target_ids.to(args.device)

        with torch.no_grad():
            predictions, loss = model(token_ids,
                                      token_type_ids,
                                      labels=target_ids)
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                epoch_loss = total_loss / (batch_idx + 1)
                logger.info(f"evaluate batch {batch_idx} , loss {round(epoch_loss, 3)}")
        
    logger.info("Evaluate finished")


if __name__ == '__main__':
    main()