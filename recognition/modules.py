from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class FlanT5Summarizer:
    def __init__(self, model_name="google/flan-t5-base", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def generate_summary(self, text, max_length=128, num_beams=4):
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        output_ids = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            num_beams=num_beams
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_batch(self, input_ids, attention_mask, max_length=128, num_beams=4):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams
        )
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

    def save_pretrained(self, save_dir):
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
