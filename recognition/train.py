import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from rouge_score import rouge_scorer
from tqdm import tqdm

from modules import FlanT5Summarizer
from dataset import BioLayDataset
from datasets import load_dataset

# ---------------------------
# Config (quick full-length training)
# ---------------------------
MODEL_NAME = "google/flan-t5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 5
LR = 5e-6
MAX_INPUT_LEN = 2048   # full report
MAX_OUTPUT_LEN = 512   # full summary
SAVE_DIR = "best_model"
DIAG_BATCHES = 5

# ---------------------------
# Load small subset of datasets
# ---------------------------
train_ds = load_dataset(
    "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track",
    split="train"
)
val_ds = load_dataset(
    "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track",
    split="validation"
)
test_ds = load_dataset(
    "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track",
    split="test"
)
print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

# ---------------------------
# Model + tokenizer
# ---------------------------
wrapper = FlanT5Summarizer(model_name=MODEL_NAME, device=DEVICE)
tokenizer = wrapper.tokenizer
model = wrapper.model

# Quick NaN check
def params_have_nan(model):
    for n, p in model.named_parameters():
        if p is None:
            continue
        if not torch.isfinite(p).all().item():
            return True, n
    return False, None

has_nan, param_name = params_have_nan(model)
if has_nan:
    raise RuntimeError(f"Model parameter {param_name} contains NaNs — aborting.")

# ---------------------------
# Dataset + DataLoader
# ---------------------------
train_dataset = BioLayDataset(train_ds, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
val_dataset   = BioLayDataset(val_ds, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
test_dataset  = BioLayDataset(test_ds, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)

# Fast collate function to avoid slow tensor warning
def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ---------------------------
# Optimizer, scorer
# ---------------------------
optimizer = AdamW(model.parameters(), lr=LR)
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)

# ---------------------------
# Utilities
# ---------------------------
def decode_labels(label_ids, tokenizer):
    """Decode labels to clean text, ignoring -100 and pad tokens"""
    targets = []
    for row in label_ids:
        cleaned = [x for x in row if x not in (-100, tokenizer.pad_token_id)]
        text = tokenizer.decode(cleaned, skip_special_tokens=True)
        targets.append(text.lower().strip())  # normalize for ROUGE
    return targets

def evaluate(model_wrapper, data_loader, tokenizer, scorer, device, max_output_len):
    """Compute ROUGE scores on a dataset"""
    model_wrapper.model.eval()
    total_f1 = {k: 0.0 for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]}
    n_examples = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Generate summaries
            preds = model_wrapper.generate_batch(
                input_ids, attention_mask, max_length=max_output_len, num_beams=2
            )
            targets = decode_labels(labels.cpu().numpy(), tokenizer)

            # Normalize predictions
            preds = [p.lower().strip() for p in preds]

            # Compute ROUGE
            for pred, tgt in zip(preds, targets):
                for key in total_f1.keys():
                    total_f1[key] += scorer.score(tgt, pred)[key].fmeasure
                n_examples += 1

    # guard against division by zero
    if n_examples == 0:
        return {k: 0.0 for k in total_f1}
    avg_rouge = {k: v / n_examples for k, v in total_f1.items()}
    return avg_rouge

# ---------------------------
# Training loop
# ---------------------------
best_rouge = -1.0
print("🚀 Starting quick full-length training...")

global_step = 0
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch"):
        global_step += 1
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"✅ Epoch {epoch} — Avg train loss: {avg_train_loss:.4f}")

    # Validation — print full set of ROUGE scores
    avg_val_rouge = evaluate(wrapper, val_loader, tokenizer, scorer, DEVICE, MAX_OUTPUT_LEN)
    # formatted print of all metrics
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_val_rouge.items()])
    print(f"📊 Validation ROUGE — {metrics_str}")

    # keep same saving criterion (based on rougeLsum)
    if avg_val_rouge.get("rougeLsum", 0.0) > best_rouge:
        best_rouge = avg_val_rouge["rougeLsum"]
        os.makedirs(SAVE_DIR, exist_ok=True)
        wrapper.save_pretrained(SAVE_DIR)
        print(f"💾 Saved best model to {SAVE_DIR}")

# ---------------------------
# Test evaluation
# ---------------------------
print("\n🧪 Evaluating best model on test set...")
avg_test_rouge = evaluate(wrapper, test_loader, tokenizer, scorer, DEVICE, MAX_OUTPUT_LEN)
metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_test_rouge.items()])
print(f"🏁 Test ROUGE — {metrics_str}")
