'''
@Author: dengzaiyong
@Date: 2021-09-21 15:16:08
@LastEditTime: 2021-09-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: 生成模块的数据处理
@FilePath: /JDQA/generative/dataprocess.py
'''

import os
import json
from config import temp_path

# 生成训练数据集
with open(os.path.join(temp_path, 'data/generative/train.tsv'), 'w') as train:
    train.write('src' + '\t' + 'tgt' + '\n')
    count = 0
    with open (os.path.join(temp_path, 'data/generative/LCCC-base_train.json'), 'r') as r:
        js_file = json.load(r)
        for item in js_file:
            # 只取前10万个
            if count >= 100000:
                break
            # 只取两句对话的
            if len(item) == 2:
                line = '\t'.join(item)
                train.write(line)
                train.write('\n')
                count += 1

# 生成验证数据集
with open(os.path.join(temp_path, 'data/generative/dev.tsv'), 'w') as dev:
    with open (os.path.join(temp_path, 'data/generative/LCCC-base_valid.json'), 'r') as r:
        dev.write('src' + '\t' + 'tgt' + '\n')
        js_file = json.load(r)
        for item in js_file:
            if len(item) == 2:
                line = '\t'.join(item)
                dev.write(line)
                dev.write('\n')

# 生成测试数据集
with open(os.path.join(temp_path, 'data/generative/test.tsv'), 'w') as test:
    with open (os.path.join(temp_path, 'data/generative/LCCC-base_test.json'), 'r') as r:
        test.write('src' + '\t' + 'tgt' + '\n')
        js_file = json.load(r)
        for item in js_file:
            if len(item) == 2:
                line = '\t'.join(item)
                test.write(line)
                test.write('\n')