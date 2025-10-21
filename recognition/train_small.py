import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import DataCollatorForSeq2Seq
from rouge_score import rouge_scorer
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

def params_have_nan(model):
    for n, p in model.named_parameters():
        if p is None:
            continue
        if not torch.isfinite(p).all().item():
            return True, n
    return False, None

def evaluate(model_wrapper, data_loader, tokenizer, scorer, device, max_output_len):
    model_wrapper.model.eval()
    total_f1 = {k: 0.0 for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]}
    n_examples = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            preds = model_wrapper.generate_batch(
                input_ids, attention_mask, max_length=max_output_len, num_beams=2
            )
            targets = decode_labels(labels.cpu().numpy(), tokenizer)

            for pred, tgt in zip(preds, targets):
                for key in total_f1.keys():
                    total_f1[key] += scorer.score(tgt, pred)[key].fmeasure
                n_examples += 1

    avg_rouge = {k: v / n_examples for k, v in total_f1.items()}
    return avg_rouge

def main():
    # ---------------------------
    # Config (stable FP32 run)
    # ---------------------------
    MODEL_NAME = "google/flan-t5-small"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 2
    NUM_EPOCHS = 10
    LR = 5e-6               # lowered LR for stability
    MAX_INPUT_LEN = 128
    MAX_OUTPUT_LEN = 64
    SAVE_DIR = "best_model_test"
    DIAG_BATCHES = 5        # number of batches to print diagnostics for
    USE_AMP = False         # AMP disabled for stability (fp32 path)

    # ---------------------------
    # Load dataset (tiny split)
    # ---------------------------
    train_ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="train[:2%]")
    val_ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="validation[:2%]")
    test_ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="test[:2%]")
    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    # ---------------------------
    # Model + tokenizer
    # ---------------------------
    wrapper = FlanT5Summarizer(model_name=MODEL_NAME, device=DEVICE)
    tokenizer = wrapper.tokenizer
    model = wrapper.model

    # quick param NaN check before training
    has_nan, param_name = params_have_nan(model)
    if has_nan:
        raise RuntimeError(f"Model parameter {param_name} contains NaNs before training — aborting.")

    # ---------------------------
    # Dataset + collator
    # ---------------------------
    train_dataset = BioLayDataset(train_ds, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    val_dataset = BioLayDataset(val_ds, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    test_dataset = BioLayDataset(test_ds, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        return_tensors="pt"
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    # ---------------------------
    # Optimizer, scorer
    # ---------------------------
    optimizer = AdamW(model.parameters(), lr=LR)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)

    best_rouge = -1.0
    print("🚀 Starting training (FP32, AMP OFF)...")

    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch"):
            global_step += 1
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # ---------- diagnostics for first few batches ----------
            if global_step <= DIAG_BATCHES:
                try:
                    unique = torch.unique(labels)
                    num_ignored = (labels == -100).sum().item()
                    total = labels.numel()
                    pct_ignored = 100.0 * num_ignored / total if total > 0 else 0.0
                    print(f"\n[DIAG] step={global_step} labels.shape={labels.shape} unique_sample={unique[:20]}")
                    print(f"[DIAG] ignored -100 tokens: {num_ignored}/{total} ({pct_ignored:.1f}%)")

                    decoded_input = tokenizer.decode(input_ids[0].cpu().tolist(), skip_special_tokens=True)
                    lab = labels[0].cpu().numpy()
                    lab_clean = [x if x != -100 else tokenizer.pad_token_id for x in lab]
                    decoded_target = tokenizer.decode(lab_clean, skip_special_tokens=True)
                    print(f"[DIAG] decoded input (trunc): {decoded_input[:300]!s}")
                    print(f"[DIAG] decoded target (trunc): {decoded_target[:200]!s}")
                except Exception as e:
                    print(f"[DIAG] diagnostic exception: {e}")

            optimizer.zero_grad()

            # ---------- FP32 forward/backward (stable) ----------
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                # extended diagnostics then skip the batch
                print("\n⚠️ NaN/Inf loss in FP32 — extended diagnostics and skipping batch")
                try:
                    logits = outputs.logits
                    finite = torch.isfinite(logits).all().item()
                    print(f"[DIAG-EXT] logits finite: {finite}")
                    if finite:
                        print(f"[DIAG-EXT] logits mean/std: {logits.mean().item():.4e}/{logits.std().item():.4e}")
                except Exception as e:
                    print(f"[DIAG-EXT] couldn't access logits: {e}")

                unique = torch.unique(labels)
                num_ignored = (labels == -100).sum().item()
                total = labels.numel()
                pct_ignored = 100.0 * num_ignored / total if total > 0 else 0.0
                print(f"[DIAG-EXT] labels.shape={labels.shape} ignored -100: {num_ignored}/{total} ({pct_ignored:.1f}%)")
                lab = labels[0].cpu().numpy()
                lab_clean = [x if x != -100 else tokenizer.pad_token_id for x in lab]
                print("[DIAG-EXT] decoded target (again):", tokenizer.decode(lab_clean, skip_special_tokens=True))

                # skip this batch to keep training moving
                continue

            # backward + step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

        # avoid division-by-zero if all batches skipped
        num_batches = len(train_loader)
        avg_train_loss = running_loss / num_batches if num_batches > 0 else float("nan")
        print(f"✅ Epoch {epoch} — Avg train loss: {avg_train_loss:.4f}")

        # ---------------------------
        # Validation
        # ---------------------------
        avg_val_rouge = evaluate(wrapper, val_loader, tokenizer, scorer, DEVICE, MAX_OUTPUT_LEN)
        current_rouge = avg_val_rouge["rougeLsum"]
        print(f"📊 Validation ROUGE-Lsum: {current_rouge:.4f}")

        if current_rouge > best_rouge:
            best_rouge = current_rouge
            os.makedirs(SAVE_DIR, exist_ok=True)
            wrapper.save_pretrained(SAVE_DIR)
            print(f"💾 Saved best model to {SAVE_DIR}")

    # ---------------------------
    # Test Evaluation
    # ---------------------------
    print("\n🧪 Evaluating best model on test set...")
    avg_test_rouge = evaluate(wrapper, test_loader, tokenizer, scorer, DEVICE, MAX_OUTPUT_LEN)
    print(f"🏁 Test ROUGE scores: {avg_test_rouge}")


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
