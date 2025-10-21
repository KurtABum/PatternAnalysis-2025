import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
from tqdm import tqdm

from modules import FlanT5Summarizer
from dataset import BioLayDataset
from datasets import load_dataset

def decode_labels(label_ids, tokenizer):
    """Replace -100 with pad_token_id and decode to text"""
    targets = []
    for row in label_ids:
        cleaned = [x if x != -100 else tokenizer.pad_token_id for x in row]
        targets.append(tokenizer.decode(cleaned, skip_special_tokens=True))
    return targets

def evaluate(model_wrapper, data_loader, tokenizer, scorer, device, max_output_len):
    model_wrapper.model.eval()
    total_f1 = {k: 0.0 for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]}
    n_examples = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            preds = model_wrapper.generate_batch(input_ids, attention_mask, max_length=max_output_len, num_beams=2)
            targets = decode_labels(labels.cpu().numpy(), tokenizer)

            for pred, tgt in zip(preds, targets):
                for key in total_f1.keys():
                    total_f1[key] += scorer.score(tgt, pred)[key].fmeasure
                n_examples += 1

    avg_rouge = {k: v / n_examples for k, v in total_f1.items()}
    return avg_rouge

def main():
    # ---------------------------
    # Config (small test version)
    # ---------------------------
    MODEL_NAME = "google/flan-t5-small"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 2          # small for quick testing
    NUM_EPOCHS = 1          # just 1 epoch for smoke test
    LR = 5e-5
    MAX_INPUT_LEN = 128     # shorten input
    MAX_OUTPUT_LEN = 64     # shorten output
    PATIENCE = 1
    SAVE_DIR = "best_model_test"

    # ---------------------------
    # Load datasets
    # ---------------------------
    train_ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="train[:2%]")
    val_ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="validation[:2%]")
    test_ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="test[:2%]")

    print(f"Train/Val/Test lengths: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    # ---------------------------
    # Model + tokenizer
    # ---------------------------
    wrapper = FlanT5Summarizer(model_name=MODEL_NAME, device=DEVICE)
    tokenizer = wrapper.tokenizer
    model = wrapper.model

    # ---------------------------
    # Dataloaders
    # ---------------------------
    train_dataset = BioLayDataset(train_ds, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    val_dataset = BioLayDataset(val_ds, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    test_dataset = BioLayDataset(test_ds, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---------------------------
    # Optimizer, scaler, scorer
    # ---------------------------
    optimizer = AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)

    best_rouge = -1.0

    print("Starting quick training test...")

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc="Training", unit="batch"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            with autocast(enabled=(DEVICE == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} completed — Avg training loss: {avg_train_loss:.4f}")

        # ---------------------------
        # Validation
        # ---------------------------
        avg_val_rouge = evaluate(wrapper, val_loader, tokenizer, scorer, DEVICE, MAX_OUTPUT_LEN)
        current_rouge = avg_val_rouge["rougeLsum"]
        print(f"Validation ROUGE-Lsum: {current_rouge:.4f}")

        if current_rouge > best_rouge:
            best_rouge = current_rouge
            os.makedirs(SAVE_DIR, exist_ok=True)
            wrapper.save_pretrained(SAVE_DIR)
            print(f"Saved best model to {SAVE_DIR}")

    # ---------------------------
    # Quick Test Evaluation
    # ---------------------------
    print("\nQuick test evaluation...")
    avg_test_rouge = evaluate(wrapper, test_loader, tokenizer, scorer, DEVICE, MAX_OUTPUT_LEN)
    print(f"Test ROUGE scores: {avg_test_rouge}")

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
