import torch
from transformers import AutoTokenizer

#加载HF分词器
class Tokenizer:
    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # Fix padding token issue

    # 编码方法
    # 返回值: tensor of token ids
    def encode(self, text: str):
        return self.tokenizer.encode(text, return_tensors='pt')

    # 解码方法
    # 返回值: 解码后的字符串
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)