import torch
from torch.utils.data import Dataset

class BioLayDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_input=512, max_output=128):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_output = max_output

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        inputs = self.tokenizer(
            item['report'],
            truncation=True,
            padding='max_length',
            max_length=self.max_input,
            return_tensors='pt'
        )
        targets = self.tokenizer(
            item['summary'],
            truncation=True,
            padding='max_length',
            max_length=self.max_output,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': targets.input_ids.squeeze()
        }
