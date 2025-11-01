"""
dataset.py

Wraps a Hugging Face dataset for radiology-to-layman text generation tasks.

This dataset class:
- Reads radiology reports and their simplified summaries.
- Tokenizes both inputs and targets for model training.
- Handles empty summaries by inserting a minimal placeholder.
- Ensures padding tokens are ignored in loss computation.

"""

import torch
from torch.utils.data import Dataset

class BioLayDataset(Dataset):
    def __init__(self, hf_dataset, tokeniser, max_input=512, max_output=128):
        """ Wrap a Hugging Face dataset for radiology report summarisation."""
        self.dataset = hf_dataset
        self.tokeniser = tokeniser
        self.max_input = max_input
        self.max_output = max_output

    def __len__(self):
        """Return length of dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get tokenised input and target tensors for index idx."""
        item = self.dataset[idx]
        report = item.get("radiology_report", "")
        summary = item.get("layman_report", "")

        # If summary is empty/whitespace, give a tiny placeholder so labels aren't all -100
        if not isinstance(summary, str) or summary.strip() == "":
            # use tokeniser.eos_token if available, else a small placeholder
            summary = (self.tokeniser.eos_token or "</s>") if getattr(self.tokeniser, "eos_token", None) else "No summary."

        # Tokenise input 
        inputs = self.tokeniser(
            report,
            truncation=True,
            padding="max_length",
            max_length=self.max_input,
            return_tensors="pt"
        )

        # Tokenise target but keep at most max_output tokens; use max_length padding so we have stable label length
        targets = self.tokeniser(
            summary,
            truncation=True,
            padding="max_length",
            max_length=self.max_output,
            return_tensors="pt"
        )

        # Remove batch dim safely
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        labels = targets.input_ids.squeeze(0).long()

        # Replace pad tokens with -100 so loss ignores them
        labels[labels == self.tokeniser.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
