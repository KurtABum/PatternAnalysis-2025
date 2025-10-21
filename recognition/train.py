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

    print(f"Raw train dataset length: {len(train_ds)}")
    print(f"Raw validation dataset length: {len(val_ds)}")
    print(f"Raw test dataset length: {len(test_ds)}")

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

    print(f"Wrapped train dataset length: {len(train_dataset)}")
    print(f"Wrapped validation dataset length: {len(val_dataset)}")
    print(f"Wrapped test dataset length: {len(test_dataset)}")
    print(f"Number of train batches per epoch: {len(train_dataset)//BATCH_SIZE} (approx)")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---------------------------
    # Optimizer and scaler
    # ---------------------------
    optimizer = AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL","rougeLsum"], use_stemmer=True)

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
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"Rows this epoch: {len(train_dataset)}")

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training", unit="batch")):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            with autocast(enabled=(DEVICE=="cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = running_loss / (batch_idx + 1)
                print(f"Batch {batch_idx+1}/{len(train_loader)} — Avg loss: {avg_loss:.4f}")

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch} completed — Avg training loss: {avg_train_loss:.4f}")

        # ---------------------------
        # Validation
        # ---------------------------
        model.eval()
        total_f1 = {k: 0.0 for k in ["rouge1","rouge2","rougeL","rougeLsum"]}
        n_examples = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                preds = wrapper.generate_batch(input_ids, attention_mask, max_length=MAX_OUTPUT_LEN, num_beams=4)

                # decode labels
                label_ids = labels.cpu().numpy()
                targets = []
                for row in label_ids:
                    targets.append(tokenizer.decode([x if x != tokenizer.pad_token_id else tokenizer.pad_token_id for x in row], skip_special_tokens=True))

                for pred, tgt in zip(preds, targets):
                    for key in total_f1.keys():
                        score = scorer.score(tgt, pred)[key].fmeasure
                        total_f1[key] += score
                    n_examples += 1

        avg_rouge_scores = {k: v/n_examples for k,v in total_f1.items()}
        val_rouges.append(avg_rouge_scores)
        current_rouge = avg_rouge_scores["rougeLsum"]

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
    # Plot training metrics
    # ---------------------------
    plt.figure()
    plt.plot(range(1,len(train_losses)+1), train_losses, marker='o', label='train_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    rougeLsum_vals = [d['rougeLsum'] for d in val_rouges]
    plt.plot(range(1,len(rougeLsum_vals)+1), rougeLsum_vals, marker='o', label='val_rougeLsum')
    plt.xlabel('epoch')
    plt.ylabel('rougeLsum')
    plt.title('Validation ROUGE-Lsum')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------------------
    # Final Test Evaluation
    # ---------------------------
    print("\nStarting final test evaluation...")
    model.eval()
    total_f1 = {k: 0.0 for k in ["rouge1","rouge2","rougeL","rougeLsum"]}
    n_examples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            preds = wrapper.generate_batch(input_ids, attention_mask, max_length=MAX_OUTPUT_LEN, num_beams=4)

            label_ids = labels.cpu().numpy()
            targets = []
            for row in label_ids:
                targets.append(tokenizer.decode([x if x != tokenizer.pad_token_id else tokenizer.pad_token_id for x in row], skip_special_tokens=True))

            for pred, tgt in zip(preds, targets):
                for key in total_f1.keys():
                    score = scorer.score(tgt, pred)[key].fmeasure
                    total_f1[key] += score
                n_examples += 1

    avg_test_rouge = {k: v/n_examples for k,v in total_f1.items()}
    print(f"\nFinal Test ROUGE scores: {avg_test_rouge}")

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
