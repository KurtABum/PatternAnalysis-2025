from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class FlanT5Summarizer:
    def __init__(self, model_name="google/flan-t5-base", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def generate_summary(self, text, max_length=1024, num_beams=4):
        """Generate a lay summary from a single radiology report"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=2048,           # let full report through
            return_tensors="pt"
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_length, # output-only length
            num_beams=num_beams,
            early_stopping=True,
            eos_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_batch(self, input_ids, attention_mask, max_length=1024, num_beams=4):
        """Generate summaries for a batch of reports"""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,  # same fix here
            num_beams=num_beams,
            early_stopping=True,
            eos_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )

        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

    def save_pretrained(self, save_dir):
        """Save model + tokenizer for reuse"""
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
