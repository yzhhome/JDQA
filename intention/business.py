'''
@Author: dengzaiyong
@Date: 2021-08-21 15:16:08
@LastEditTime: 2021-08-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: 训练fasttext文本分类模型，用fasttext模型进行意图识别
@FilePath: /JDQA/intention/business.py
'''

import os
import config
import fastText
from jieba import posseg
import pandas as pd
from tqdm import tqdm
from utils.tools import create_logger
from preprocessor import clean, filter_content

tqdm.pandas()

logger = create_logger(config.root_path + '/logs/intention.log')

class Intention(object):
    def __init__(self,
                data_path=config.train_path,             # 训练数据集文件
                sku_path=config.ware_path,               # sku文件
                model_path=None,                         # fasttext模型保存路径
                keywords_path=None,                      # 提取的关键词保存路径
                model_train_file=config.business_train,  # 意图识别的训练数据集
                model_test_file=config.business_test):   # 意图识别的测试数据集

        self.model_path = model_path
        self.data = pd.read_csv(data_path)

        if model_path and  os.path.exists(model_path):
            self.fast_model = fastText.load_model(model_path)
        else:
            if os.path.exists(keywords_path):
                self.keywords = open(keywords_path, 'r').read().split()
            else:
                # 提取意图识别的关键词
                self.keywords = self.build_keyword(sku_path, to_file=keywords_path)

            if not os.path.exists(model_train_file):
                self.data_process(model_train_file)            

            self.fast_model = self.train(model_train_file, model_test_file)

        
    def build_keyword(self, sku_path, to_file):
        """
        提取用于意图识别的关键词
        从训练数据集和SKU文件中提取
        """
        logger.info("Building keywords...")
        tokens = []

        # 提取名词，名动词，其他专名
        tokens = self.data['custom'].dropna().apply(
            lambda x: [token for token, pos in posseg.cut(x) if pos in  ['n', 'vn', 'nz']]
        )

        key_words = set([tk for idx, sample in tokens.iteritems() 
                        for tk in sample if len(tk) > 1])

        logger.info("Building keywords finished")

        sku = []

        with open(sku_path, 'r') as f:
            next(f)
            for lines in f:
                line = lines.strip().split('\t')
                sku.extend(line[-1].split('/'))

        key_words |= set(sku)

        logger.info('Sku words merged.')

        if not os.path.exists(os.path.dirname(to_file)):
            os.mkdir(os.path.dirname(to_file))

        if to_file is not None:
            with open(to_file, 'w') as f:
                for word in key_words:
                    f.write(word + '\n')
        return key_words


    def data_process(self, model_data_file):
        """
        判断咨询中是否包含业务关键词，如果包含label为1否则为0
        并处理成fasttext模型需要的数据格式
        """ 
        logger.info('Processing data.')

        # 在提取的关键词中为0不在则为0
        self.data['is_business'] = self.data['custom'].progress_apply(
            lambda x: 1 if any(kw in x for kw in self.keywords) else 0)

        if not os.path.exists(os.path.dirname(model_data_file)):
            os.mkdir(os.path.dirname(model_data_file))            

        with open(model_data_file, 'w') as f:
            # 显示tqdm进度条
            for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
                # 保存为fasttext模型的数据格式
                outline = clean(row['custom']) + "\t__label__" + str(
                    int(row['is_business'])) + "\n"
                f.write(outline)

    def train(self, model_train_file, model_test_file):
        """
        训练和测试fasttext模型，并保存
        """
        logger.info('Training fasttext classifier')

        classifier = fastText.train_supervised(model_train_file, 
                                                label='__label__',
                                                dim=100,      
                                                epoch=10,
                                                lr=0.1,
                                                wordNgrams=2,
                                                loss='softmax',
                                                thread=5,
                                                verbose=True)

        classifier.save_model(self.model_path) 
        logger.info('fasttext model saved.')

        self.test(classifier, model_test_file)
        return classifier   

    def test(self, classifier, model_test_file):
        """
        测试fasttest模型分类的精度
        """          
        logger.info('Testing fasttext model')

        test = pd.read_csv(config.test_path).fillna('')

        # 在提取的关键词中为0不在则为0
        test['is_business'] = test['custom'].progress_apply(
            lambda x: 1 if any(kw in x for kw in self.keywords) else 0)

        if not os.path.exists(os.path.dirname(model_test_file)):
            os.mkdir(os.path.dirname(model_test_file))                  

        with open(model_test_file, 'w') as f:
            for index, row in tqdm(test.iterrows(), total=test.shape[0]):
                outline = clean(row['custom']) + "\t__label__" + str(
                    int(row['is_business'])) + "\n"
                f.write(outline)
        
        result = classifier.test(model_test_file)

        # F1 score
        # precesion * recall * 2/ (precesion + recall)
        print('F1 score:', result[1] * result[2] * 2 / (result[2] + result[1]))


    def predict(self, text):
        """
        使用fasttext模型预测分类
        """
        logger.info('Predicting classification.')
        label, score = self.fast_model.predict(clean(filter_content(text)))
        return label, score


if __name__ == "__main__":

    # 构建fasttext意图识别模型
    intention = Intention(data_path=config.train_path,      # 训练数据集文件
                        sku_path=config.ware_path,          # sku文件
                        model_path=config.fasttext_path,    # fasttext模型保存路径
                        keywords_path=config.keyword_path)  # 提取的关键词保存路径

    print(intention.predict('户外服装怎么卖呢？'))
    print(intention.predict('你好'))