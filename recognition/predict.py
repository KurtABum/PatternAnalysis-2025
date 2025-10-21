import torch
from modules import FlanT5Summarizer

model_wrapper = FlanT5Summarizer()
model_wrapper.model.load_state_dict(torch.load("best_model.pt"))
model_wrapper.model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model_wrapper.model.to(device)

# Example usage
examples = [
    "Patient has a 3cm mass in the upper lobe of the right lung. Recommend follow-up CT.",
    "MRI shows mild disc bulge at L4-L5 without nerve compression."
]

for ex in examples:
    summary = model_wrapper.generate_summary(ex)
    print(f"Report: {ex}\nSummary: {summary}\n")
