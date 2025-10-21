import torch
from datasets import load_dataset
from modules import FlanT5Summarizer

MODEL_DIR = "best_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

wrapper = FlanT5Summarizer(model_name=MODEL_DIR, device=DEVICE)
test_ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="test")

examples = test_ds.shuffle(seed=42)[:3]

for i, item in enumerate(examples):
    expert_report = item["radiology_report"]
    generated_summary = wrapper.generate_summary(expert_report, max_length=128)

    print(f"\n--- Example {i+1} ---")
    print("Expert Report:")
    print(expert_report[:500] + ("..." if len(expert_report) > 500 else ""))
    print("\nGenerated Lay Summary:")
    print(generated_summary)
    print("\nTarget Lay Summary:")
    print(item["layman_report"])
    print("-" * 50)
