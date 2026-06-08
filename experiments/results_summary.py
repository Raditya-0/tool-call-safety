"""Cetak ringkasan hasil eksperimen dari outputs/results_summary.json sebagai tabel."""
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR


def main():
    summary_path = os.path.join(OUTPUT_DIR, "results_summary.json")
    with open(summary_path) as f:
        summary = json.load(f)

    print(f"{'Model':<20} {'Accuracy':>10} {'F1 Weighted':>12} {'ASR (%)':>10}")
    print("-" * 56)
    for model_name, results in summary.items():
        acc = results["accuracy"] * 100
        f1 = results["f1_weighted"] * 100
        asr = results["asr"] * 100
        print(f"{model_name:<20} {acc:>9.2f}% {f1:>11.2f}% {asr:>9.2f}%")

    for model_name, results in summary.items():
        print(f"\n{model_name} — per kelas:")
        print(f"  {'Kelas':<22} {'F1':>8} {'Precision':>10} {'Recall':>8}")
        for cls, scores in results["per_class"].items():
            print(f"  {cls:<22} {scores['f1']:>8.3f} {scores['precision']:>10.3f} {scores['recall']:>8.3f}")


if __name__ == "__main__":
    main()
