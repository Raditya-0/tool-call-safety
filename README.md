# TCSSC (Tool-Call Sequence Safety Classifier)

Deteksi jailbreak pada LLM agentic melalui analisis urutan parameter tool call secara sequential.

Paper: *"Tool-Call Sequence Safety Classifier: Detecting Jailbreak in LLM Agents through Sequential Tool Call Parameter Analysis"*

## Latar Belakang

LLM agent modern dapat dimanipulasi melalui rangkaian *tool call* yang masing-masing terlihat aman secara individual, tetapi secara kumulatif menghasilkan aksi yang destruktif (lihat serangan STAC — Sequential Tool Call Attack). Pendekatan moderasi yang ada umumnya hanya membaca teks permintaan pengguna atau output model, tanpa menganalisis isi dan urutan parameter tool call yang sebenarnya dieksekusi.

TCSSC mengisi celah ini dengan mengklasifikasikan keamanan sebuah sesi agent berdasarkan *seluruh urutan* tool call beserta parameternya — bukan hanya satu permintaan tunggal — sehingga pola serangan sekuensial dan injeksi parameter tersembunyi dapat terdeteksi sebelum tereksekusi.

## Arsitektur

```
Input (tool call sequence)
        ↓
Tool Call Encoder (mBERT + MLP)    — encode setiap tool call
        ↓
Sequence Aggregator                 — baca urutan
    ├── TCSSC-LSTM  (Bidirectional LSTM 2 layer)
    └── TCSSC-Transformer (4 layer Transformer)
        ↓
Harm Classifier (MLP 2 layer)
        ↓
Output: benign / direct_attack / sequential_attack / parameter_injection
```

## Hasil Eksperimen

| Model             | Accuracy | F1 Weighted | ASR   |
| ----------------- | -------- | ----------- | ----- |
| Baseline (STAC)   | —       | —          | 58.6% |
| TCSSC-LSTM        | 85.4%    | 85.3%       | 5.1%  |
| TCSSC-Transformer | 82.9%    | 82.7%       | 5.7%  |

ASR turun 91.3% relatif terhadap baseline.

## Dataset

| Sumber               | Jumlah           | Label       |
| -------------------- | ---------------- | ----------- |
| SafeToolBench        | 1.000            | Multi-kelas |
| STAC Benchmark       | 483              | sequential  |
| ToolSafe AgentHarm   | 416              | Multi-kelas |
| ToolSafe SafetyBench | 1.586            | Multi-kelas |
| GitHub Scraping      | 134              | Pseudolabel |
| Augmentasi           | ~6.817           | Augmented   |
| **Total**      | **10.436** |             |

## Kelas Label

| ID | Label               | Deskripsi                                     |
| -- | ------------------- | --------------------------------------------- |
| 0  | benign              | Tool call normal dan aman                     |
| 1  | direct_attack       | Tool call tunggal langsung berbahaya          |
| 2  | sequential_attack   | Rangkaian tool call kumulatif destruktif      |
| 3  | parameter_injection | Parameter tersembunyi berbahaya dalam argumen |

## Struktur Folder

```
FP/
├── README.md
├── .gitignore
├── requirements.txt
├── config.py
├── main.py
├── TCSSC_Kaggle.ipynb
├── data/
│   ├── dataset.py
│   ├── scraper.py
│   ├── augmentor.py
│   ├── verify_dataset.py
│   ├── manual_label_template.py
│   ├── raw/
│   │   ├── scraped_github.csv
│   │   ├── STAC_benchmark_data.json
│   │   ├── ToolSafe/             ← clone repo ToolSafe (gitignored, ~658MB)
│   │   └── labels_batch_*.json
│   └── processed/
│       ├── safetoolbench/        ← 8 file query_*.json
│       ├── augmented.csv
│       ├── scraped_labeled.csv
│       └── wildjailbreak.parquet (gitignored)
├── models/
│   └── tcssc.py
├── experiments/
│   ├── trainer.py
│   └── results_summary.py
├── utils/
│   └── preprocessing.py
├── outputs/                      ← hasil training (gitignored)
└── checkpoints/                  ← model weights (gitignored, ~1.4GB)
```

## Cara Menjalankan

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Siapkan dataset (download dari Kaggle dataset: tcssc-dataset)
# Taruh file di data/processed/ dan data/raw/ sesuai struktur folder di atas

# 3. Verifikasi dataset
python data/verify_dataset.py

# 4. Training semua model (TCSSC-LSTM dan TCSSC-Transformer)
python main.py

# 5. Lihat ringkasan hasil
python experiments/results_summary.py
```

## Cara Scraping Data Baru

```bash
# 1. Scraping GitHub Issues + Reddit (butuh token GitHub & Reddit di data/scraper.py)
python data/scraper.py

# 2. Manual pseudolabeling — buka template, copy prompt ke Gemini,
#    paste response JSON ke data/raw/labels_batch_N.json
python data/manual_label_template.py

# 3. Gabungkan seluruh batch label menjadi data/processed/scraped_labeled.csv
#    (dilakukan satu kali; lihat data/dataset.py::load_scraped_labeled untuk format yang diharapkan)
```

## Referensi

```
[1] Cartagena & Teixeira. Mind the GAP. arXiv:2602.16943, 2026.
[2] Li et al. STAC. arXiv:2509.25624, 2025.
[3] ToolSafety. EMNLP 2025.
[4] Mazeika et al. AgentHarm. ICLR 2025.
[5] Inan et al. LLaMA Guard. Meta AI, 2023.
[6] Hu et al. LoRA. ICLR 2022.
[7] Dettmers et al. QLoRA. NeurIPS 2023.
```

## Lisensi

MIT License — untuk keperluan penelitian akademik.
