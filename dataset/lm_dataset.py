from torch.utils.data import Dataset
import torch
import os
from datasets import load_dataset
import json

# 取消tokenizer并行加速
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                # 提取每一行内容放到sample
                data = json.loads(line.strip())
                samples.append(data)
        return samples

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
        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return x, y, loss_mask
