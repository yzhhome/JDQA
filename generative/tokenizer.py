'''
@Author: dengzaiyong
@Date: 2021-09-21 15:16:08
@LastEditTime: 2021-09-27 19:37:08
@LastEditors: dengzaiyong
@Desciption: 手动实现Bert Tokenizer
@FilePath: /JDQA/generative/tokenizer.py
'''

import sys
import unicodedata
from typing import Dict, List
from config import base_chinese_bert_vocab

def load_chinese_base_vocab():
    """
    加载bert预训练模型字典
    """
    word2idx = {}
    with open(base_chinese_bert_vocab, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            word2idx[line.strip('\n')] = index
    return word2idx


class BasicTokenizer(object):
    """
    Bert Tokenizer的父类
    """
    def __init__(self):
        self._token_pad = '[PAD]'
        self._token_cls = '[CLS]'
        self._token_sep = '[SEP]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'

    def tokenize(self, text: str, add_cls=True, add_sep=True, max_length=None):
        """
        分词函数
        """
        tokens = self._tokenize(text)
        if add_cls:
            tokens.insert(0, self._token_cls)
        if add_sep:
            tokens.append(self._token_sep)
        if max_length is not None:
            self.truncate_sequence(max_length, tokens, None, -2)

        return tokens

    def token_to_id(self, token):
        """
        token转换为对应的id
        """
        raise NotImplementedError    

    def tokens_to_ids(self, tokens):
        """
        token序列转换为对应的id序列
        """
        return [self.token_to_id(token) for token in tokens]

    def truncate_sequence(self, 
                        max_length,
                        firt_sequence: List[str],
                        seconde_sequence=None,
                        pop_index=-1):
        """
        对firt_sequence + seconde_sequence的长度进行截断        
        """

        # 只有一个句子
        if seconde_sequence is None:
            seconde_sequence = []

        while True:
            total_length = len(firt_sequence) + len(seconde_sequence)

            # 两个句子长度达到max_length不需要再截断
            if total_length <= max_length:
                break
            # 第一个句子比第二个句子长则截断第一个句子
            elif len(firt_sequence) > len(seconde_sequence):
                firt_sequence.pop(pop_index)
            # 第一个句子比第一个句子长则截断第二个句子
            else:
                seconde_sequence.pop(pop_index)
    
    def encode(self,
               first_text,
               second_text=None,
               max_length=None,
               first_length=None,
               second_length=None):
        """输出文本对应token id和segment id
        如果传入first_length，则强行padding第一个句子到指定长度；
        同理，如果传入second_length，则强行padding第二个句子到指定长度。
        """
        first_tokens = self.tokenize(first_text)

        if second_text is None:
            second_tokens = None
        else:
            second_tokens = self.tokenize(second_text)

        if max_length is not None:
            self.truncate_sequence(max_length, first_tokens, second_tokens, -2)

        first_token_ids = self.tokens_to_ids(first_tokens)

        if first_length is not None:
            # 确定第一个句子的长度为first_length
            first_token_ids = first_token_ids[0:first_length]

            # 不足first_length的补0
            first_token_ids.extend([self._token_pad] * 
                                    (first_length - len(first_token_ids)))

        # 第一个句子标识
        first_segment_ids = [0] * len(first_token_ids)

        if second_text is not None:
            second_token_ids = self.tokens_to_ids(second_tokens)
            
            if second_length is not None:
                # 确定第二个句子的长度为second_length
                second_token_ids = second_token_ids[:second_length]

                # 不足second_length的补0
                second_token_ids.extend([self._token_pad_id] *
                                        (second_length - len(second_token_ids)))

            # 第一个句子标识
            second_segment_ids = [1] * len(second_token_ids)

            # 添加到第一个句子后面
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)

        return first_token_ids, first_segment_ids        

    def id_to_token(self, i):
        """
        id序列为对应的token
        """
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        """
        id序列转换为对应的token序列
        """
        return [self.id_to_token(i) for i in ids]

    def decode(self, ids):
        """
        转为可读文本
        """
        raise NotImplementedError

    def _tokenize(self, text):
        """
        基本分词函数
        """
        raise NotImplementedError


class Tokenizer(BasicTokenizer):
    def __init__(self, token_dict):
        super(Tokenizer, self).__init__()

        self._token_dict = token_dict

        self._token_dict_inv = {v: k for k, v in token_dict.items()}
        for token in ['pad', 'cls', 'sep', 'unk', 'mask']:
            try:
                # 取出特殊字符的tokenid
                _token_id = token_dict[getattr(self, f'_token_{token}')]

                # 设置属性
                setattr(self, f'_token_{token}_id', _token_id)
            except Exception as e:
                print(e)
        self._vocab_size = len(token_dict)

    def token_to_id(self, token):
        """
        token转换为对应的id
        """
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, id):
        """
        id转换为对应的token
        """
        return self._token_dict_inv[id]

    def decode(self, ids):
        """
        转为可读文本
        """
        tokens = self.ids_to_tokens(ids)
        return "".join(tokens).strip()     

    def _tokenize(self, text):
        """基本分词函数, 以每个字切分
        """
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch    
        
        return spaced.strip().split()   

    @staticmethod
    def _is_space(ch):
        """空格类字符判断
        """
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
            unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_punctuation(ch):
        """标点符号类字符判断（全/半角均在此内）
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
            58 <= code <= 64 or \
            91 <= code <= 96 or \
            123 <= code <= 126 or \
            unicodedata.category(ch).startswith('P')

    @staticmethod
    def _cjk_punctuation():
        return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008 \
                \u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54 \
                \xb7\uff01\uff1f\uff61\u3002'

    @staticmethod
    def _is_cjk_character(ch):
        """CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']') 

if __name__ == "__main__":    
    # 测试分词并通过id还原句子
    word2idx = load_chinese_base_vocab()
    tokenizer = Tokenizer(word2idx)
    input_ids, segment_ids = tokenizer.encode("你好啊，今天过的怎么样？", "我很好，谢谢你啦")
    text = tokenizer.decode(input_ids)
    print(text)