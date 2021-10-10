'''
@Author: dengzaiyong
@Date: 2021-09-21 15:16:08
@LastEditTime: 2021-09-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: Hyper parameters for knowledge distillation    
@FilePath: /JDQA/generative/distill_config.py
'''

import argparse
import os
from config import root_path, temp_path

def parse(opt=None):
    parser = argparse.ArgumentParser()

    # 学生模型保存路径
    parser.add_argument(
        "--output_dir",
        default=os.path.join(temp_path, 'model/generative'),
        type=str,
        help="The output directory where the model checkpoints will be written."
    )
    # 日志文件路径
    parser.add_argument(
        "--log_file",
        default=os.path.join(root_path, 'logs/distil.log'),
        type=str,
        help="The output directory where the model checkpoints will be written."
    )
    # 训练数据集
    parser.add_argument("--train_path",
                        default=os.path.join(temp_path, 'data/generative/train.tsv'),
                        type=str)
    parser.add_argument("--dev_path",
                        default=os.path.join(temp_path, 'data/generative/dev.tsv'),
                        type=str)
    parser.add_argument("--test_path",
                        default=os.path.join(temp_path, 'data/generative/test.tsv'),
                        type=str)
    # 文本转换为小写
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Whether to lower case the input text. Should be True for uncased "
            "models and False for cased models.")
    # 文本最大长度
    parser.add_argument(
        "--max_length",
        default=100,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization."
            "Sequences longer than this will be truncated, and sequences shorter than this will be padded."
    )
    # 
    # 长文档分成块的滑窗大小
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks."
    )
    # 是否为训练模式
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    # 是否为预测模式
    parser.add_argument("--do_predict",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    # 训练的batch_ize
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    # 预测的batch_sizes
    parser.add_argument("--predict_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for predictions.")
    # 学习率
    parser.add_argument("--learning_rate",
                        default=0.01,
                        type=float,
                        help="The initial learning rate for Adam.")
    # 总计训练的epochs
    parser.add_argument("--num_train_epochs",
                        default=50,
                        type=int,
                        help="Total number of training epochs to perform.")
    # 学习率warmup比例
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start \
             and end predictions are not conditioned on one another.")
    parser.add_argument(
        "--verbose_logging",
        default=True,
        action='store_true',
        help="If true, all of the warnings related to data processing will be printed. \
             A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--is_cuda",
                        default=True,
                        type=bool,
                        help="Whether to use CUDA when available")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass."
    )
    parser.add_argument('--random_seed', type=int, default=10236797)
    parser.add_argument('--load_model_type',
                        type=str,
                        default='bert',
                        choices=['bert', 'none'])
    parser.add_argument('--weight_decay_rate', type=float, default=0.01)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--PRINT_EVERY', type=int, default=200)
    parser.add_argument('--weight', type=float, default=1.0)
    parser.add_argument('--ckpt_frequency', type=int, default=1)

    for i in range(29, -1, -1):
        model_path = temp_path + 'model/generative/bert.model.epoch.' + str(i)
        if os.path.exists(model_path):
            break    

    # 教师模型保存路径
    parser.add_argument('--tuned_checkpoint_T',
                        type=str,
                        default=model_path)
    # fine tuen学生模型
    parser.add_argument('--tuned_checkpoint_S', type=str, default=None)

    # 预训练学生模型路径
    parser.add_argument("--init_checkpoint_S",
                        default=os.path.join(temp_path,
                        'lib/rbt3/pytorch_model.bin'),
                        type=str)
    # 蒸馏温度
    parser.add_argument("--temperature", default=1, type=float, required=False)
    parser.add_argument("--teacher_cached", action='store_true')

    parser.add_argument('--s_opt1',
                        type=float,
                        default=1.0,
                        help="release_start / step1 / ratio")
    parser.add_argument('--s_opt2',
                        type=float,
                        default=0.0,
                        help="release_level / step2")
    parser.add_argument('--s_opt3',
                        type=float,
                        default=1.0,
                        help="not used / decay rate")
    parser.add_argument('--schedule',
                        type=str,
                        default='warmup_linear_release')
    # 匹配的层
    parser.add_argument('--matches', 
                        nargs='*', 
                        type=list, 
                        default=[
                                'L3_hidden_smmd',
                                'L3_hidden_mse',
                                'L3_attention_mse',
                                'L3_attention_ce',
                                'L3_attention_mse_sum',
                                'L3_attention_ce_mean',
                                ])
    if opt is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(opt)
    return args


if __name__ == '__main__':
    args = parse()
    print(args)