'''
@Author: dengzaiyong
@Date: 2021-08-21 15:16:08
@LastEditTime: 2021-08-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: 数据进行预处理，清洗，转换为问答pair，并保存
@FilePath: /JDQA/preprocessor.py
'''

import re
import os
import numpy as np
import pandas as pd
import config
from utils.tools import create_logger
from sklearn.model_selection import train_test_split

logger = create_logger(config.root_path + '/logs/preprocessor.log')

def filter_content(sentence):
    """
    对句子内容进行过滤
    特殊字段有：
    1. #E-s[数字x] #E-2[数字x] 等一系列数字—— 表情
    2. [ORDERID_10187709] —— 订单号
    3. [数字x] —— 数字
    4. https://item.jd.com/5898522.html —— 网址
    5. [地址x] —— 地址
    6. [链接x] —— 链接
    7. [金额x] —— 金额
    8. [日期x] —— 日期
    9. [时间x] —— 时间
    10. [站点x] —— 站点
    11. [组织机构x] ——组织机构
    12. [电话x] —— 电话
    13. [姓名x] —— 人名
    对于表情，做法是直接删除。其他用希腊符号替换。
    """
    sentence = str(sentence)
    sentence = re.sub(
        r"#E\-[\w]*(抱拳|傲慢|得意|蛋糕|呕吐|闭嘴|礼物|yaoping|柠檬|流泪|怒火|撇嘴|太阳|咒骂|糗|猪猪|足球|磕头|大兵|电话|灯泡|飞鸟|奋斗|高兴|击打|饥饿|咖啡|口罩|骷髅|可乐|疯狂|白眼|阴险|叹气|奸笑|发呆|害羞|飞吻|怒火|悲伤|胜利|生病|弱|可怜|咖啡|酷酷|眩晕|流泪|发抖|难过|右哼哼|惊恐|悲伤|犯困|愤怒|凋谢|哈欠|拥抱|抓狂|鄙视|时间|啤酒|勾引|左哼哼|月亮|偷笑|震惊|惊讶|跳跳|瞌睡|可爱|衰样|好|憨笑|水果|色色|黑线|微笑|流汗|握手|心碎|问号|大哭|亲亲|抠鼻|拜拜|鬼脸|香吻|米饭|花朵|尴尬|擦汗|安慰|委屈|调皮|爱心|我一定尽力为您解答的哦|很棒|鼓掌)+",
        "α", sentence)
    sentence = re.sub(r"#E\-[\w]+\[数字x]", "α", sentence)
    sentence = re.sub(r"\[ORDERID_[\d]+]", "[订单x]", sentence)
    sentence = re.sub(r"\[数字x]", "γ", sentence)
    sentence = re.sub(r"\[链接x]", "ε", sentence)
    sentence = re.sub(r"\[表情]", "α", sentence)
    sentence = re.sub("<sep>", config.SEP, sentence)
    sentence = re.sub("<SEP>", config.SEP, sentence)
    sentence = re.sub(
        r"(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?",
        "ε", sentence)
    sentence = re.sub(r"(http|ftp|https):\/\/ε", "ε", sentence)
    sentence = re.sub(r"[\d]+.*[\d]+", "γ", sentence)
    sentence = re.sub(r"【收到不支持的消息类型，暂无法显示】", " ", sentence)

    sentence = re.sub(r"#E\-[s]*(ν|γ|π|ζ|ρ|α|ε)*", "α", sentence)
    sentence = re.sub("α", " ", sentence)
    sentence = re.sub("ε", "[链接x]", sentence)
    sentence = re.sub("γ", "[数字x]", sentence)

    return sentence

def read_file(path, is_train=False):
    '''
    读取文件，并将原始数据中多次输入合并为一句
    path: 数据文件所在目录
    is_train：是否为训练数据集
    return: list 包含session_id, role, content
    '''
    chat = []

    with open(path, 'r') as f:

        tmp = []
        sessions = set()
        session_id, custom_id, is_assistance, content = '', '', '', []
        for lines in f:
            if len(sessions) > 300000:  # 0.3 million sessions at most.
                break
            line = lines.strip().replace(' ', '').split('\t')
            if len(line) < 5:  # Filtering short samples.
                continue
            if is_train:
                session_id_in_doc, custom_id_in_doc, is_assistance_in_doc = \
                    line[0], line[1], line[2]
            else:
                session_id_in_doc, custom_id_in_doc, is_assistance_in_doc = \
                    line[2], line[1], line[3]
            sessions.add(session_id_in_doc)
            if session_id != session_id_in_doc and session_id != '':
                fc = filter_content(content)
                if fc != '':
                    tmp.append([
                        session_id, 'custom'
                        if str(is_assistance) == '0' else 'assistance', fc
                    ])
                    content = []
                chat.extend(tmp)
                tmp = []
                session_id, custom_id = session_id_in_doc, custom_id_in_doc
            else:
                if is_assistance != is_assistance_in_doc and \
                        is_assistance != '':
                    content = filter_content(content)
                    is_assistance = 'custom' if str(
                        is_assistance) == '0' else 'assistance'
                    if content != '':
                        tmp.append([session_id, is_assistance, content])
                    is_assistance = is_assistance_in_doc
                    content = [line[-1]]
                else:
                    content.append(line[-1])
                    is_assistance = is_assistance_in_doc
                    session_id, _ = session_id_in_doc, custom_id_in_doc
    if content != '':
        tmp.append([
            session_id,
            'custom' if str(is_assistance) == '0' else 'assistance',
            filter_content(content)
        ])
    chat.extend(tmp)
    return chat

def clean(sent, sep='<'):
    """
    过滤句子中一些无用符号，过滤<sep>或[SEP]分隔符
    """
    sent = re.sub(r"[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+",
                  "", sent)
    i = 0
    tmp = []
    while i < len(sent):
        # 正常字符
        if sent[i] != sep:
            tmp.append(sent[i])
            i += 1
        # 碰到分隔符
        else:
            tmp.append(sent[i:i + 5])
            i += 5
            
    # 返回以空格分隔词的句子
    return " ".join(tmp)

def generate_data(filepath, save=True, to_file=None, pair=False):
    '''
    将read_file的结果进行转化成问答pair, 或者将一次会话内容合并一起
    file_path: 原始数据路径
    save: 是否保存
    to_file: 保存的文件名，会根据文件名判断是否为训练集
    pair: 是否生成问答pair的结果
    return: 处理后可以作为模型输入的数据集
    '''
    data = read_file(filepath, 'train' in to_file)
    data = pd.DataFrame(data, columns=['session_id', 'role', 'content'])
    if 'train' in to_file:
        data = data[(data['content'].str.len() <= 128)
                    & (data['content'].str.len() > 1)].reset_index(drop=True)

    data = data.reset_index()
    data['index'] = data['index'].apply(lambda x: x - 1
                                        if x % 2 == 1 else x)
    data = data.pivot_table(index=['index', 'session_id'],
                            columns='role',
                            values='content',
                            aggfunc='first').reset_index()
                            
    data = data[['session_id', 'custom',
                'assistance']].dropna().reset_index(drop=True)

    if save:
        data.to_csv('{}.csv'.format(to_file), index=False)
    return data

def process_ranking_data():
    """
    处理ranking原始数据，划分为训练，验证，测试数据集
    """
    with open(config.root_path + '/data/ranking/rank_data.csv', 'w') as fw:
        fw.write('question1' + '\t' + 'question2' + '\t' + 'label' + '\n')

        count = 0        
        with open(config.root_path + '/data/ranking/atec_nlp_sim_train_add.csv', 'r') as f1:
            lines = f1.readlines()
            total = len(lines) 

            for line in lines:
                pair = line.strip().split('\t')
                count += 1
                if count < total:
                    fw.write(pair[1] + '\t' + pair[2] + '\t' + pair[3] + '\n')
                else:
                    fw.write(pair[1] + '\t' + pair[2] + '\t' + pair[3])

        fw.write('\n')

        count = 0
        with open(config.root_path + '/data/ranking/atec_nlp_sim_train.csv', 'r') as f2:
            lines = f2.readlines()
            total = len(lines) 

            for line in lines:
                pair = line.strip().split('\t')
                count += 1

                if count < total:
                    fw.write(pair[1] + '\t' + pair[2] + '\t' + pair[3] + '\n')
                else:
                    fw.write(pair[1] + '\t' + pair[2] + '\t' + pair[3])
                

    df_rank = pd.read_csv(config.root_path + '/data/ranking/rank_data.csv', sep='\t')
    X = df_rank.drop(['label'], axis=1)
    y = df_rank['label']

    # 先划分训练数据集和测试数据集
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, 
        random_state=42, shuffle=True)

    # 再从训练数据集中划分一部分作为验证数据集
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, 
        random_state=42, shuffle=True)

    # 写入ranking训练数据集
    with open(config.rank_train_file, 'w') as ft:
        ft.write('question1' + '\t' + 'question2' + '\t' + 'label' + '\n')
        count = 0
        total = len(y_train)

        for x, y in zip(X_train, y_train):
            count += 1
            if count < total:
                ft.write(x[0] + '\t' + x[1] + '\t' + str(y) + '\n')
            else:
                ft.write(x[0] + '\t' + x[1] + '\t' + str(y))

    # 写入ranking测试数据集
    with open(config.rank_test_file, 'w') as ft:
        ft.write('question1' + '\t' + 'question2' + '\t' + 'label' + '\n')
        count = 0
        total = len(y_test)

        for x, y in zip(X_test, y_test):
            count += 1
            if count < total:
                ft.write(x[0] + '\t' + x[1] + '\t' + str(y) + '\n')
            else:
                ft.write(x[0] + '\t' + x[1] + '\t' + str(y))   

    # 写入ranking验证数据集
    with open(config.rank_dev_file, 'w') as ft:
        ft.write('question1' + '\t' + 'question2' + '\t' + 'label' + '\n')
        count = 0
        total = len(y_dev)

        for x, y in zip(X_dev, y_dev):
            count += 1
            if count < total:
                ft.write(x[0] + '\t' + x[1] + '\t' + str(y) + '\n')
            else:
                ft.write(x[0] + '\t' + x[1] + '\t' + str(y))       



if __name__ == "__main__":   

    # 根据原始数据生成模型可用的训练数据集
    data = generate_data(config.train_raw,
                         save=True,
                         to_file=os.path.join(
                             config.root_path, 'data/train_no_blank'),
                         pair=True)
    logger.info('training set created.')

    # 根据原始数据生成模型可用的验证数据集
    dev = generate_data(config.dev_raw,
                        save=True,
                        to_file=os.path.join(config.root_path, 'data/dev'),
                        pair=True)
    logger.info('Dev set created.')
    
    # 根据原始数据生成模型可用的测试数据集
    test = generate_data(config.test_raw,
                         save=True,
                         to_file=os.path.join(config.root_path, 'data/test'),
                         pair=True)
    logger.info('test set created.')

    # 根据ranking原始数据生成训练、验证、测试数据集
    process_ranking_data()