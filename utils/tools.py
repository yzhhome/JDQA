import logging
import os
import re
import torch
import numpy as np
import pandas as pd

def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(levelname)s - %(asctime)s : %(message)s')    

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

def clean_str(string):
    """
    清除一些无用字符
    """
    # 空白字符, 包括空格、制表符、换页符
    string = re.sub(r"\s+", "", string)

    # 非汉字字符
    string = re.sub(r"[^\u4e00-\u9fff]", "", string)
    return string.strip()

def collate_fn(batch):
    """
    对batch数据进行padding
    """
    def padding(indice, max_length, pad_idx=0):
        """
        对token_id进行padding
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return pad_indice

    token_ids = [data["token_ids"] for data in batch]
    max_lengths = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    
    token_ids_padded = padding(token_ids, max_lengths)
    token_type_ids_padded = padding(token_type_ids, max_lengths)

    token_ids_padded = np.array(token_ids_padded)
    token_type_ids_padded = np.array(token_type_ids_padded)

    # 生成的target_ids从第1列开始
    target_ids_padded = np.array(token_ids_padded[:, 1:])

    token_ids_padded = torch.tensor(token_ids_padded)
    token_type_ids_padded = torch.tensor(token_type_ids_padded)  
    target_ids_padded = torch.tensor(target_ids_padded)   

    return token_ids_padded, token_type_ids_padded, target_ids_padded

def read_corpus(data_path):
    """
    读取数据集，返回问答对列表
    """
    df = pd.read_csv(data_path, sep='\t', encoding='utf-8').dropna()
    sent_src = []
    target_src = []
    for index, row in df.iterrows():
        query = row['src']
        answer = row['tgt']
        sent_src.append(query)
        target_src.append(answer)
    return sent_src, target_src