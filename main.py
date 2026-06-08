"""
Script utama — jalankan semua eksperimen TCSSC
Urutan: load data → train LSTM → train Transformer → bandingkan → print hasil
"""
import os
import json
import torch
import random
import argparse
import numpy as np
from transformers import AutoTokenizer

import sys
sys.path.append(os.path.dirname(__file__))
from config import *
from data.dataset import load_all_datasets, build_dataloaders
from models.tcssc import TCSSC
from experiments.trainer import train, evaluate_final, compute_asr


# Reproducibility
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Jalankan satu eksperimen
def run_experiment(
    aggregator_type: str,
    train_loader,
    val_loader,
    test_loader,
    train_samples: list,
    device: str,
    num_epochs: int = NUM_EPOCHS,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Eksperimen: TCSSC-{aggregator_type.upper()}")
    print(f"{'='*60}")

    model = TCSSC(aggregator_type=aggregator_type, freeze_bert=True)

    # hitung jumlah parameter trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameter trainable: {trainable:,}")

    # training
    history = train(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        train_samples   = train_samples,
        aggregator_type = aggregator_type,
        device          = device,
        num_epochs      = num_epochs,
    )

    # load model terbaik untuk evaluasi final
    ckpt = os.path.join(CHECKPOINT_DIR, f"tcssc_{aggregator_type}_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)

    # evaluasi final
    print(f"\nEvaluasi final TCSSC-{aggregator_type.upper()} pada test set:")
    results = evaluate_final(model, test_loader, device)

    # hitung ASR
    asr = compute_asr(model, test_loader, device)
    results["asr"] = asr

    return results, history


# Bandingkan semua hasil
def print_comparison(all_results: dict):
    print(f"\n{'='*60}")
    print("PERBANDINGAN SEMUA MODEL")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Accuracy':>10} {'F1 Weighted':>12} {'ASR (%)':>10}")
    print("-" * 60)

    for model_name, (results, _) in all_results.items():
        acc = results.get("accuracy", 0) * 100
        f1  = results["report"]["weighted avg"]["f1-score"] * 100
        asr = results.get("asr", 0)
        print(f"{model_name:<25} {acc:>9.2f}% {f1:>11.2f}% {asr:>9.2f}%")


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 2 epochs, batch_size 4, 1 aggregator only")
    args = parser.parse_args()

    num_epochs = 2          if args.debug else NUM_EPOCHS
    batch_size = 4          if args.debug else BATCH_SIZE
    agg_types  = ["lstm"]   if args.debug else ["lstm", "transformer"]

    set_seed()

    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if args.debug:
        print("DEBUG MODE: epochs=2, batch_size=4, aggregator=lstm only")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # load tokenizer
    print(f"\nMemuat tokenizer: {ENCODER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)

    # load semua dataset
    print("\nMemuat dataset...")
    train_samples, val_samples, test_samples = load_all_datasets(PROCESSED_DIR)

    # buat dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        train_samples, val_samples, test_samples, tokenizer, batch_size=batch_size
    )

    print(f"\nTrain: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    # jalankan semua eksperimen
    all_results = {}

    for agg_type in agg_types:
        results, history = run_experiment(
            aggregator_type = agg_type,
            train_loader    = train_loader,
            val_loader      = val_loader,
            test_loader     = test_loader,
            train_samples   = train_samples,
            device          = device,
            num_epochs      = num_epochs,
        )
        all_results[f"TCSSC-{agg_type.upper()}"] = (results, history)

        # simpan history
        history_path = os.path.join(OUTPUT_DIR, f"history_{agg_type}.json")
        with open(history_path, "w") as f:
            # konversi numpy ke float untuk JSON serialization
            hist_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
            json.dump(hist_serializable, f, indent=2)

    # tampilkan perbandingan
    print_comparison(all_results)

    # simpan semua hasil
    summary = {}
    for model_name, (results, _) in all_results.items():
        summary[model_name] = {
            "accuracy":    float(results.get("accuracy", 0)),
            "f1_weighted": float(results["report"]["weighted avg"]["f1-score"]),
            "asr":         float(results.get("asr", 0)),
            "per_class":   {
                k: {"f1": float(v["f1-score"]), "precision": float(v["precision"]), "recall": float(v["recall"])}
                for k, v in results["report"].items()
                if k in LABEL2ID
            }
        }

    summary_path = os.path.join(OUTPUT_DIR, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nHasil disimpan di {summary_path}")


if __name__ == "__main__":
    main()
