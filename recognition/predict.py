import torch
from datasets import load_dataset
from modules import FlanT5Summarizer

# ---------------------------
# Config
# ---------------------------
MODEL_DIR = "best_model_test"  # local folder with your trained model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_OUTPUT_LEN = 128  # max length for generated summaries
NUM_EXAMPLES = 5     # number of examples to preview

# ---------------------------
# Load model
# ---------------------------
wrapper = FlanT5Summarizer(model_name=MODEL_DIR, device=DEVICE)

# ---------------------------
# Load test dataset
# ---------------------------
test_ds = load_dataset(
    "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track",
    split="test"
)

# Select a few examples safely using `.select()`
examples = test_ds.shuffle(seed=42).select(range(NUM_EXAMPLES))

# ---------------------------
# Generate summaries
# ---------------------------
for i, item in enumerate(examples):
    expert_report = item["radiology_report"]
    generated_summary = wrapper.generate_summary(expert_report, max_length=MAX_OUTPUT_LEN)

    print(f"\n--- Example {i+1} ---")
    print("Expert Report:")
    print(expert_report[:500] + ("..." if len(expert_report) > 500 else ""))
    print("\nGenerated Lay Summary:")
    print(generated_summary)
    print("\nTarget Lay Summary:")
    print(item["layman_report"])
    print("-" * 50)
