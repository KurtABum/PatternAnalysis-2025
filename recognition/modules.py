"""
modules.py

Wrapper for Flan-T5 models to summarise radiology reports.

This module provides:
- FlanT5Summariser: initialises model + tokenizer on CPU/GPU.
- Methods to generate single or batch summaries.
- Beam search and repetition control for higher-quality outputs.
- Easy save/load of model and tokeniser.

"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class FlanT5Summariser:
    """
    Flan-T5 Summariser for radiology reports.
    """
    def __init__(self, model_name="google/flan-t5-base", device="cpu"):
        """
        Initialise tokeniser + model.
        device: 'cpu' or 'cuda' for GPU.
        """
        self.device = device
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def generate_summary(self, text, max_length=1024, num_beams=4):
        """
        Summarise a single radiology report.
        text: full report string
        max_length: max tokens in summary
        num_beams: beam search width for quality
        """
        inputs = self.tokeniser(
            text,
            truncation=True,
            padding=True,
            max_length=2048,  # allows full report to fit
            return_tensors="pt"
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=num_beams,
            early_stopping=True,
            eos_token_id=self.tokeniser.eos_token_id,
            no_repeat_ngram_size=3  # prevents repetitive phrases
        )

        return self.tokeniser.decode(output_ids[0], skip_special_tokens=True)

    def generate_batch(self, input_ids, attention_mask, max_length=1024, num_beams=4):
        """
        Summarise a batch of reports.
        input_ids, attention_mask: pre-tokenised batch tensors
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            num_beams=num_beams,
            early_stopping=True,
            eos_token_id=self.tokeniser.eos_token_id,
            no_repeat_ngram_size=3
        )

        return [self.tokeniser.decode(ids, skip_special_tokens=True) for ids in output_ids]

    def save_pretrained(self, save_dir):
        """
        Save model and tokeniser for later use.
        """
        self.model.save_pretrained(save_dir)
        self.tokeniser.save_pretrained(save_dir)
