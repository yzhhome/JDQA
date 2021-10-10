'''
@Author: dengzaiyong
@Date: 2021-08-21 15:16:08
@LastEditTime: 2021-08-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: Generating features and train a LightGBM ranker.
@FilePath: /JDQA/ranking/ranker.py
'''

import os
import pandas as pd
import lightgbm as lgb
import joblib
from tqdm import tqdm
from ranking.matchnn import MatchingNN
from ranking.similarity import TextSimilarity
from sklearn.model_selection import train_test_split
from utils.tools import create_logger
from config import root_path, temp_path, lightgbm_rows_limit
import config

tqdm.pandas()
logger = create_logger(root_path + '/logs/ranker.log')

# lightgbm参数
params = {'boosting_type': 'gbdt',
          'max_depth': 5,
          'objective': 'binary',
          'nthread': 3,  # Updated from nthread
          'num_leaves': 64,
          'learning_rate': 0.05,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 0.5,
          'subsample_freq': 5,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'max_position': 20,
          'group': 'name:groupId',
          'metric': 'auc'}

class RANK(object):
    def __init__(self,
                do_train=True,
                model_path=root_path + '/model/ranking/lightgbm.bin'):

        self.similarity = TextSimilarity()
        self.matchNN = MatchingNN()

        if do_train:
            logger.info("Training lightgbm model")
            self.train = pd.read_csv(config.rank_train_file, sep='\t', encoding='utf-8')

            if not os.path.exists(config.lightgbm_train_file):            

                # 生成人工特征和bert特征
                self.data = self.generate_feature(self.train)

                # 保存生成的训练特征文件
                self.data.to_csv(config.lightgbm_train_file, sep='\t', index=False)
            else:
                self.data = pd.read_csv(config.lightgbm_train_file, sep='\t', encoding='utf-8')

            self.train()
            self.save(model_path)
        else:
            logger.info('Predicting lightgbm model')
            self.gbm = joblib.load(model_path)   

    def generate_feature(self, data):
        """
        生成模型训练所需要的特征
        """
        logger.info('Generating manual features.')

        # 将原始特征和人工特征拼接
        data = pd.concat([data, pd.DataFrame.from_records(
            data.apply(lambda row: self.similarity.generate_all(
                row['question1'], row['question2']), axis=1))], axis=1)

        logger.info('Generating deeep-matching features.')

        # 生成深度学习模型特征
        data['matching_score'] = data.apply(lambda row: self.matchNN.predict(
            row['question1'], row['question2'])[-1], axis=1)

        return data

    def train(self):
        """
        lightgbm模型训练
        """
        logger.info('Training lightgbm model.')
        self.gbm = lgb.LGBMRanker(metrics='auc')

        # 排除原始特征,只保留生成的特征
        columns = [i for i in self.data.columns if i not in 
            ['question1', 'question2', 'label']]

        X_train, X_test, y_train, y_test = train_test_split(
            self.data[columns], self.data['label'],
            test_size=0.3, random_state=42)

        if X_train.shape[0] > lightgbm_rows_limit:
            X_train = X_train[:lightgbm_rows_limit]
            y_train = y_train[:lightgbm_rows_limit]

        if X_test.shape[0] > lightgbm_rows_limit:
            X_test = X_test[:lightgbm_rows_limit]
            y_test = y_test[:lightgbm_rows_limit]            

        query_train = [X_train.shape[0]]
        query_val = [X_test.shape[0]]

        # 训练lightgbm
        self.gbm.fit(X_train, y_train, group=query_train,
                     eval_set=[(X_test, y_test)], eval_group=[query_val],
                     eval_at=[5, 10, 20], early_stopping_rounds=100, verbose=True)

    def test(self, 
            model_path=root_path + '/model/ranking/lightgbm.bin'):

        logger.info('Testing lightgbm model')
        self.gbm = joblib.load(model_path)

        self.test = pd.read_csv(config.rank_test_file, sep='\t', encoding='utf-8')

            # 生成人工特征和bert特征
        self.testdata = self.generate_feature(self.test)

        # 保存生成的测试特征文件
        self.testdata.to_csv(config.lightgbm_tes_file, sep='\t', index=False)

        self.predict(self.testdata)

    def save(self, model_path):
        """
        保存lightgbm模型
        """
        logger.info('Saving lightgbm model.')
        joblib.dump(self.gbm, model_path)

    def predict(self, data):
        """
        lightgbm模型预测
        """
        columns = [i for i in data.columns if i not in 
            ['question1', 'question2','label']]

        result = self.gbm.predict(data[columns])

        return result

if __name__ == "__main__":
    rank = RANK(do_train=True)
    rank.test()