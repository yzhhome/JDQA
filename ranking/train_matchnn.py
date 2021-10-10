'''
@Author: dengzaiyong
@Date: 2021-08-21 15:16:08
@LastEditTime: 2021-08-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: Train a matching network.
@FilePath: /JDQA/ranking/train_matchnn.py
'''

import os
import torch
import config
from torch.utils.data import DataLoader
from dataprocess import DataProcessForSentence
from matchnn_utils import train, validate
from transformers import BertTokenizer
from matchnn import BertModelTrain
from transformers.optimization import AdamW
from config import is_cuda, root_path, device
from utils.tools import create_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = create_logger(root_path + '/logs/train_matchnn.log')

torch.manual_seed(42)
if is_cuda:
    torch.cuda.manual_seed_all(42)


def train_matchnn(train_file,            # 训练数据集
                    dev_file,            # 验证数据集
                    target_dir,          # 模型保存目录
                    epochs=10,           # 训练的epoch轮数
                    batch_size=32,       # batch size
                    lr=2e-05,            # 学习率 
                    patience=3,          # early stop的条件
                    max_grad_norm=10.0,  # 最大梯度
                    checkpoint=None):    # 之前保存的模型参数
    
    # 加载预训练的bert模型
    bert_tokenizer = BertTokenizer.from_pretrained(root_path + '/lib/bert/', do_lower_case=True)

    logger.info("Preparing for training")

    # 创建保存模型的目录
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    logger.info("Loading training data")
    train_data = DataProcessForSentence(bert_tokenizer, train_file)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    logger.info("Loading validation data")
    dev_data = DataProcessForSentence(bert_tokenizer, dev_file)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=True)    

    logger.info("Building model")

    model = BertModelTrain().to(device)

    # 待优化的参数
    param_optimizer = list(model.named_parameters())

    # 不需要weight decay的参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {   
            'params':
            [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        }, 
        {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]    

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    scheduler = ReduceLROnPlateau(optimizer, patience=0, 
        verbose=True, mode='max', factor=0.85)

    best_score = 0.0
    start_epoch = 1

    epochs_count = []
    train_losses = []
    valid_losses = []

    # 继续之前保存的模型参数学习
    if checkpoint:
        check_point = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        logger.info("Training will continue on existing model from epoch {}"
            .format(start_epoch))

        # 读取之前保存的参数
        model.load_state_dict(check_point["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]        

    # 开始或者继续训练之前计算模型损失，准确率，auc
    _, valid_loss, valid_accuracy, auc = validate(model, dev_loader)

    logger.info("Validation loss before training: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}"
        .format(valid_loss, (valid_accuracy * 100), auc))

    logger.info("Training bert model on device: {}".format(device))

    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        logger.info("Training epoch {}:".format(epoch))

        epoch_time, epoch_loss, epoch_accuracy = train(model, 
                                                    train_loader, 
                                                    optimizer, 
                                                    max_grad_norm)

        train_losses.append(epoch_loss)

        logger.info("Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%".
            format(epoch_time, epoch_loss, (epoch_accuracy * 100)))

        logger.info("Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, epoch_auc = validate(
            model, dev_loader)
        valid_losses.append(epoch_loss)

        logger.info("Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n"
            .format(epoch_time, epoch_loss, (epoch_accuracy * 100), epoch_auc))

        # 更新学习率
        scheduler.step(epoch_accuracy)     

        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            logger.info("Save model on best epoch validation accuracy: {:.4f}%".
                format((epoch_accuracy * 100)))

            best_score = epoch_accuracy
            patience_counter = 0
            # 保存模型参数
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses
                }, os.path.join(target_dir, "best_bert.model"))  

        # 达到early stop条件结束训练
        if patience_counter >= patience:
            logger.info("Early stopping: patience limit reached, stopping...")
            break

if __name__ == '__main__':
    train_matchnn(config.rank_train_file, 
                    config.rank_dev_file,
                    root_path + "/model/ranking/")