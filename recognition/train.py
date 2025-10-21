import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import matplotlib.pyplot as plt
from modules import FlanT5Summarizer
from dataset import BioLayDataset
from datasets import load_dataset
from rouge_score import rouge_scorer

# Load datasets
train_ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="train")
val_ds   = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="validation")

# Initialize model & tokenizer
model_wrapper = FlanT5Summarizer()
tokenizer = model_wrapper.tokenizer
model = model_wrapper.model
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Create dataloaders
train_loader = DataLoader(BioLayDataset(train_ds, tokenizer), batch_size=8, shuffle=True)
val_loader   = DataLoader(BioLayDataset(val_ds, tokenizer), batch_size=8)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop with early stopping
best_rouge = 0
patience = 2
counter = 0
train_losses, val_rouges = [], []

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = {k:v.to(device) for k,v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Validation ROUGE
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rougeLsum'])
    rouge_score_epoch = 0
    with torch.no_grad():
        for batch in val_loader:
            input_texts = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            target_texts = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            for inp, tgt in zip(input_texts, target_texts):
                pred = model_wrapper.generate_summary(inp)
                score = scorer.score(tgt, pred)['rougeLsum'].fmeasure
                rouge_score_epoch += score
    avg_rouge = rouge_score_epoch / len(val_loader.dataset)
    val_rouges.append(avg_rouge)
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val ROUGE-Lsum={avg_rouge:.4f}")

    # Early stopping
    if avg_rouge > best_rouge:
        best_rouge = avg_rouge
        torch.save(model.state_dict(), "best_model.pt")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

# Plot loss and ROUGE
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_rouges, label="Val ROUGE-Lsum")
plt.legend()
plt.show()
