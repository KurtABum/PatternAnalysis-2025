from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class FlanT5Summarizer:
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_summary(self, text, max_length=128):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        output_ids = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
