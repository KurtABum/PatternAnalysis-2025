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
    """Evaluate model and return average ROUGE scores"""
    model_wrapper.model.eval()
    total_f1 = {k: 0.0 for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]}
    n_examples = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            preds = model_wrapper.generate_batch(input_ids, attention_mask, max_length=max_output_len, num_beams=4)
            targets = decode_labels(labels.cpu().numpy(), tokenizer)

            for pred, tgt in zip(preds, targets):
                for key in total_f1.keys():
                    total_f1[key] += scorer.score(tgt, pred)[key].fmeasure
                n_examples += 1

    avg_rouge = {k: v / n_examples for k, v in total_f1.items()}
    return avg_rouge

def main():
    # ---------------------------
    # Config
    # ---------------------------
    MODEL_NAME = "google/flan-t5-base"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8
    NUM_EPOCHS = 5
    LR = 5e-5
    MAX_INPUT_LEN = 512
    MAX_OUTPUT_LEN = 128
    PATIENCE = 2
    SAVE_DIR = "best_model"

    # ---------------------------
    # Load datasets
    # ---------------------------
    train_ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="train")
    val_ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="validation")
    test_ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="test")

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
    patience_counter = 0
    train_losses = []
    val_rouges = []

    print("Starting training...")

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch {epoch}/{NUM_EPOCHS} — {len(train_dataset)} samples")

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training", unit="batch")):
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
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch} completed — Avg training loss: {avg_train_loss:.4f}")

        # ---------------------------
        # Validation
        # ---------------------------
        avg_val_rouge = evaluate(wrapper, val_loader, tokenizer, scorer, DEVICE, MAX_OUTPUT_LEN)
        val_rouges.append(avg_val_rouge)
        current_rouge = avg_val_rouge["rougeLsum"]
        print(f"Validation ROUGE-Lsum: {current_rouge:.4f}")

        # ---------------------------
        # Early stopping + save best
        # ---------------------------
        if current_rouge > best_rouge:
            best_rouge = current_rouge
            patience_counter = 0
            os.makedirs(SAVE_DIR, exist_ok=True)
            wrapper.save_pretrained(SAVE_DIR)
            print(f"Saved best model (ROUGE-Lsum={best_rouge:.4f}) to {SAVE_DIR}")
        else:
            patience_counter += 1
            if patience_counter > PATIENCE:
                print("Early stopping — no improvement")
                break

    # ---------------------------
    # Plot metrics
    # ---------------------------
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='train_loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('Training loss')
    plt.grid(True); plt.legend(); plt.show()

    plt.figure()
    rougeLsum_vals = [d['rougeLsum'] for d in val_rouges]
    plt.plot(range(1, len(rougeLsum_vals)+1), rougeLsum_vals, marker='o', label='val_rougeLsum')
    plt.xlabel('epoch'); plt.ylabel('rougeLsum'); plt.title('Validation ROUGE-Lsum')
    plt.grid(True); plt.legend(); plt.show()

    # ---------------------------
    # Final Test Evaluation
    # ---------------------------
    print("\nStarting final test evaluation...")
    avg_test_rouge = evaluate(wrapper, test_loader, tokenizer, scorer, DEVICE, MAX_OUTPUT_LEN)
    print(f"\nFinal Test ROUGE scores: {avg_test_rouge}")

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
