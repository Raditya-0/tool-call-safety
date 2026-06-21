"""
Verifikasi dataset — jalankan sebelum training.
Usage: python data/verify_dataset.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PROCESSED_DIR, STAC_PATH,
    TOOLSAFE_AGENTHARM_DIR, TOOLSAFE_SAFETYBENCH_PATH,
    WILDJAILBREAK_PATH, SCRAPED_LABELED_PATH,
    RJUDGE_DIR,
    ID2LABEL, LABEL2ID,
)
from data.dataset import (
    load_stac, load_safetoolbench_dir,
    load_toolsafe_agentharm, load_toolsafe_safetybench,
    load_wildjailbreak, load_scraped_labeled,
    load_rjudge_dir,
    generate_benign_samples,
)
from utils.preprocessing import validate_sample
from collections import Counter


SOURCES = [
    ("SafeToolBench",          lambda: load_safetoolbench_dir(os.path.join(PROCESSED_DIR, "safetoolbench"))
                                       if os.path.isdir(os.path.join(PROCESSED_DIR, "safetoolbench")) else []),
    ("STAC",                   lambda: load_stac(STAC_PATH) if os.path.exists(STAC_PATH) else []),
    ("ToolSafe AgentHarm",     lambda: load_toolsafe_agentharm(TOOLSAFE_AGENTHARM_DIR)
                                       if os.path.isdir(TOOLSAFE_AGENTHARM_DIR) else []),
    ("ToolSafe SafetyBench",   lambda: load_toolsafe_safetybench(TOOLSAFE_SAFETYBENCH_PATH)
                                       if os.path.exists(TOOLSAFE_SAFETYBENCH_PATH) else []),
    ("WildJailbreak",          lambda: load_wildjailbreak(WILDJAILBREAK_PATH)
                                       if os.path.exists(WILDJAILBREAK_PATH) else []),
    ("Scraped labeled",        lambda: load_scraped_labeled(SCRAPED_LABELED_PATH)
                                       if os.path.exists(SCRAPED_LABELED_PATH) else []),
    ("R-Judge",                lambda: load_rjudge_dir(RJUDGE_DIR)
                                       if os.path.isdir(RJUDGE_DIR) else []),
]


def main():
    all_samples = []
    source_counts = {}

    print("=" * 60)
    print("LOADING SEMUA SUMBER")
    print("=" * 60)

    for name, loader in SOURCES:
        try:
            samples = loader()
            all_samples.extend(samples)
            source_counts[name] = len(samples)
            status = f"{len(samples)} samples" if samples else "SKIP (tidak ditemukan)"
            print(f"  {name:<28}: {status}")
        except Exception as e:
            print(f"  {name:<28}: ERROR — {e}")
            source_counts[name] = 0

    # Synthetic benign untuk balance
    benign_count  = sum(1 for s in all_samples if s["label"] == "benign")
    attack_count  = len(all_samples) - benign_count
    n_synthetic   = max(100, min(attack_count // 3 - benign_count, 500))
    benign = generate_benign_samples(n_synthetic)
    all_samples.extend(benign)
    source_counts["Synthetic benign"] = len(benign)
    print(f"  {'Synthetic benign':<28}: {len(benign)} samples")

    print(f"\n{'TOTAL':<28}: {len(all_samples)} samples")

    # Distribusi per label
    dist = Counter(s["label"] for s in all_samples)
    print("\nDistribusi label:")
    for label in sorted(dist):
        pct = dist[label] / len(all_samples) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<25}: {dist[label]:>5}  ({pct:5.1f}%)  {bar}")

    # Cek target 10k
    if len(all_samples) >= 10000:
        print(f"\n✓ Target 10.000 tercapai: {len(all_samples)} samples")
    else:
        needed = 10000 - len(all_samples)
        print(f"\n✗ Belum 10.000: kurang {needed} samples")
        if not os.path.exists(WILDJAILBREAK_PATH):
            print("  → Download WildJailbreak: python data/download_wildjailbreak.py")
        if not os.path.exists(SCRAPED_LABELED_PATH):
            print("  → Jalankan scraping: python data/scraper.py")
            print("  → Lalu pseudolabeling: python data/pseudolabeler.py")

    # Validasi format
    invalid = [i for i, s in enumerate(all_samples) if not validate_sample(s)]
    if invalid:
        print(f"\nWARN: {len(invalid)} sample tidak valid di index: {invalid[:10]}")
    else:
        print(f"\nSemua {len(all_samples)} sample valid.")

    # 2 contoh per kelas
    print("\n" + "=" * 60)
    print("CONTOH PER KELAS (2 per label)")
    print("=" * 60)
    shown = Counter()
    for s in all_samples:
        label = s["label"]
        if shown[label] >= 2:
            continue
        shown[label] += 1
        seq   = s["tool_call_sequence"]
        names = [tc.get("name", tc.get("tool_name", "?")) for tc in seq[:4]]
        ctx   = s["conversation"][0]["content"][:80] if s.get("conversation") else ""
        print(f"\n[{label}] source={s['source']}")
        print(f"  seq_len : {len(seq)}")
        print(f"  tools   : {names}")
        print(f"  context : {ctx!r}")
        if all(shown[l] >= 2 for l in dist):
            break

    print("\n" + "=" * 60)
    print("Verifikasi selesai.")


if __name__ == "__main__":
    main()
