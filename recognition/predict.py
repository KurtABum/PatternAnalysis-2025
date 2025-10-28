import torch
from datasets import load_dataset
from modules import FlanT5Summariser

MODEL_DIR = "best_model_test"  #directory of trained model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_OUTPUT_LEN = 512  #max length for generated summaries
NUM_EXAMPLES = 5     

model = FlanT5Summariser(model_name=MODEL_DIR, device=DEVICE)

test_ds = load_dataset(
    "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track",
    split="test"
)

# Select a few examples safely using `.select()`
examples = test_ds.shuffle(seed=42).select(range(NUM_EXAMPLES))

#generate and print summaries
for i, item in enumerate(examples):
    expert_report = item["radiology_report"]
    generated_summary = model.generate_summary(expert_report, max_length=MAX_OUTPUT_LEN)

    print(f"\n--- Example {i+1} ---")
    print("Expert Report:")
    print(expert_report[:500] + ("..." if len(expert_report) > 500 else ""))
    print("\nGenerated Lay Summary:")
    print(generated_summary)
    print("\nTarget Lay Summary:")
    print(item["layman_report"])
    print("-" * 50)
