from torch.utils.data import Dataset
import torch
import os
from datasets import load_dataset
import json

# 取消tokenizer并行加速
os.environ["TOKENIZERS_PARALLELISM"] = "false"



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        # 编码器
        encoding = self.tokenizer(
            # 取出text部分,并转换为str
            str(sample['text']),
            max_length=self.max_length,
            # 不足长度的padding到最大长度
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        # (max_length,)
        input_ids = encoding['input_ids'].squeeze()
        # mask用于标记哪些是真实token , 哪些是padding
        loss_mask = (input_ids != self.tokenizer.pad_token_id).long()
        # 自回归 [1,2,3,4,5,6] -> x=[1,2,3,4,5] y=[2,3,4,5,6] ,保证x,y长度一致
        x = torch.tensor(input_ids[:-1], dtype=torch.LongTensor)
        y = torch.tensor(input_ids[1:], dtype=torch.LongTensor)
        loss_mask = torch.Tensor(loss_mask[1:], dtype=torch.LongTensor)
        return x, y, loss_mask
