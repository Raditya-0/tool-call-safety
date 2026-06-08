"""
Training loop dan evaluasi TCSSC
"""
import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, classification_report,
    confusion_matrix, accuracy_score
)
from collections import Counter

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *


# Hitung class weights untuk imbalanced data
def compute_class_weights(samples: list, device: str = DEVICE) -> torch.Tensor:
    labels = [LABEL2ID.get(s["label"], s["label"]) if isinstance(s["label"], str) else s["label"]
              for s in samples]
    counts = Counter(labels)
    total  = sum(counts.values())
    weights = [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)]
    return torch.tensor(weights, dtype=torch.float).to(device)


# Satu epoch training
def train_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     str,
) -> tuple:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        # pindahkan semua tensor ke device
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("label")

        optimizer.zero_grad()
        logits = model(batch)
        loss   = criterion(logits, labels)
        loss.backward()

        # gradient clipping untuk stabilitas training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, f1


# Evaluasi pada satu dataloader
def evaluate(
    model:   nn.Module,
    loader:  DataLoader,
    criterion: nn.Module,
    device:  str,
) -> tuple:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")

            logits = model(batch)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, f1, all_preds, all_labels


# Training loop utama
def train(
    model:         nn.Module,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    train_samples: list,
    aggregator_type: str = "lstm",
    num_epochs:    int   = NUM_EPOCHS,
    lr:            float = LEARNING_RATE,
    device:        str   = DEVICE,
    checkpoint_dir: str  = CHECKPOINT_DIR,
) -> dict:

    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)

    # class weights untuk imbalanced data
    weights   = compute_class_weights(train_samples, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_f1   = 0.0
    patience_cnt  = 0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    for epoch in range(1, num_epochs + 1):
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f}")

        # simpan model terbaik
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_cnt = 0
            ckpt_path = os.path.join(checkpoint_dir, f"tcssc_{aggregator_type}_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Model terbaik disimpan (val F1: {best_val_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOP:
                print(f"Early stopping di epoch {epoch}")
                break

    return history


# Evaluasi final pada test set
def evaluate_final(
    model:       nn.Module,
    test_loader: DataLoader,
    device:      str = DEVICE,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    _, _, preds, labels = evaluate(model, test_loader, criterion, device)

    report = classification_report(
        labels, preds,
        target_names=list(LABEL2ID.keys()),
        output_dict=True,
        zero_division=0,
    )
    cm  = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=list(LABEL2ID.keys()), zero_division=0))
    print("Confusion Matrix:")
    print(cm)

    return {"report": report, "confusion_matrix": cm, "accuracy": acc}


# Hitung Attack Success Rate (ASR)
def compute_asr(
    model:       nn.Module,
    test_loader: DataLoader,
    device:      str = DEVICE,
) -> float:
    """
    ASR = proporsi serangan yang TIDAK terdeteksi oleh TCSSC.
    Kelas 0 (benign) dianggap "lolos", kelas 1-3 adalah serangan.
    """
    model.eval()
    attack_total    = 0
    attack_detected = 0

    with torch.no_grad():
        for batch in test_loader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label").cpu().numpy()
            logits = model(batch)
            preds  = logits.argmax(dim=-1).cpu().numpy()

            for true, pred in zip(labels, preds):
                if true != 0:  # sample ini adalah serangan
                    attack_total += 1
                    if pred == 0:  # model salah klasifikasi jadi benign
                        attack_detected += 1

    asr = (attack_detected / attack_total * 100) if attack_total > 0 else 0.0
    print(f"Attack Success Rate (ASR): {asr:.2f}% ({attack_detected}/{attack_total} serangan lolos)")
    return asr
