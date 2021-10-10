'''
@Author: dengzaiyong
@Date: 2021-08-21 15:16:08
@LastEditTime: 2021-08-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: Helper functions for training the matching network.
@FilePath: /JDQA/ranking/train_matchnn.py
'''

import torch
import torch.nn as nn
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def generate_sent_masks(enc_hiddens, source_lengths):
    """ 
    Generate sentence masks for encoder hidden states.
    Args:
        enc_hiddens: encodings of shape (b, src_len, h)
            b = batch size,src_len = max source length, h = hidden size. 
            
        source_lengths: List of actual lengths for each of the sentences in the batch.
            len = batch size
    Returns: 
        Tensor of sentence masks of shape (b, src_len)
        src_len = max source length, b = batch size.
    """
    # 生成全是0的mask矩阵
    enc_masks = torch.zeros(enc_hiddens.size(0),
                            enc_hiddens.size(1),
                            dtype=torch.float)

    # mask矩阵每行0:src_len置为1
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, :src_len] = 1
    return enc_masks

def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)

    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask

    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)

def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask

def correct_predictions(probabilities, targets):
    """
    计算正确预测的数量
    """
    _, out_classes = probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()

def validate(model, dataloader):
    """
    验证模型
    """
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []

    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in dataloader:

            seqs = batch_seqs.to(device)
            masks = batch_seq_masks.to(device)
            segments = batch_seq_segments.to(device)
            labels = batch_labels.to(device)

            loss, logits, probabilities = model(seqs, masks, segments, labels)

            # 损失累加
            running_loss += loss.item()

            # 准确率累加
            running_accuracy += correct_predictions(probabilities, labels)

            # 所有预测的概率
            all_prob.extend(probabilities[:, 1].cpu().numpy())

            # 所有的标签
            all_labels.extend(batch_labels)

    # epoch的运行时间
    epoch_time = time.time() - epoch_start

    # epoch的loss
    epoch_loss = running_loss / len(dataloader)

    # epoch的准确率
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    return epoch_time, epoch_loss, epoch_accuracy, \
            roc_auc_score(all_labels, all_prob)

def test(model, dataloader):
    """
    测试模型
    """
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob = []
    all_labels = []

    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in dataloader:
            # batch的开始时间
            batch_start = time.time()

            seqs, masks, segments, labels =    \
                batch_seqs.to(device),         \
                batch_seq_masks.to(device),    \
                batch_seq_segments.to(device), \
                batch_labels.to(device)

            # 送到bert模型输出
            loss, logits, probabilities = model(seqs, masks, segments, labels)

            # 预测正确数量累加
            accuracy += correct_predictions(probabilities, labels)

            # 总的时间累加
            batch_time += time.time() - batch_start

            # 所有预测的概率
            all_prob.extend(probabilities[:, 1].cpu().numpy())

            # 所有的标签
            all_labels.extend(batch_labels)

    # batch的平均时间
    batch_time /= len(dataloader)

    # 总的运行时间
    total_time = time.time() - time_start

    # 准确率求平均
    accuracy /= (len(dataloader.dataset))

    return batch_time, total_time, accuracy, \
            roc_auc_score(all_labels, all_prob)


def train(model, dataloader, optimizer, max_gradient_norm):
    """
    训练模型
    """
    model.train()
    device = model.device
    epoch_start = time.time() # epoch开始运行时间
    batch_time_total = 0.0    # 所有batch运行时间
    running_loss = 0.0        # 每个epoch的损失
    correct_preds = 0         # 每个epoch正确预测的数量

    tqdm_batch_iterator = tqdm(dataloader)

    for index, (seqs, seq_masks, seq_segments, labels) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # 搬到GPU上
        seqs = seqs.to(device)
        seq_masks = seq_masks.to(device)
        seq_segments = seq_segments.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # 获取模型输出
        loss, logits, probabilities = model(seqs, seq_masks, seq_segments, labels)
        loss.backward()

        # 进行梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        # batch运行时间累加
        batch_time_total += time.time() - batch_start

        # 损失累加
        running_loss += loss.item()

        # 预测正确数量累加
        correct_preds += correct_predictions(probabilities, labels)

        # 更新进度条上面的运行时间和loss
        description = "Average batch run time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_total / (index+1), running_loss / (index + 1))
        tqdm_batch_iterator.set_description(description)

    # epoch运行时间
    epoch_time = epoch_start - time.time()

    # epoch损失
    epoch_loss= running_loss / len(dataloader)

    # epoch准确率
    epoch_accuracy  = correct_preds / len(dataloader.dataset)

    return epoch_time, epoch_loss, epoch_accuracy