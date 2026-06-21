# TCSSC (Tool-Call Sequence Safety Classifier)

Deteksi jailbreak pada LLM agentic melalui analisis urutan parameter tool call secara sequential.

Paper: *"Tool-Call Sequence Safety Classifier: Detecting Jailbreak in LLM Agents through Sequential Tool Call Parameter Analysis"*

## Latar Belakang

LLM agent modern dapat dimanipulasi melalui rangkaian *tool call* yang masing-masing terlihat aman secara individual, tetapi secara kumulatif menghasilkan aksi yang destruktif (lihat serangan STAC (Sequential Tool Call Attack)). Pendekatan moderasi yang ada umumnya hanya membaca teks permintaan pengguna atau output model, tanpa menganalisis isi dan urutan parameter tool call yang sebenarnya dieksekusi.

TCSSC mengisi celah ini dengan mengklasifikasikan keamanan sebuah sesi agent berdasarkan *seluruh urutan* tool call beserta parameternya bukan hanya satu permintaan tunggal tapi pola serangan sekuensial dan injeksi parameter tersembunyi dapat terdeteksi sebelum tereksekusi.

## Arsitektur

```
Input (tool call sequence)
        ↓
Tool Call Encoder (mBERT + MLP)    — encode setiap tool call
        ↓
Sequence Aggregator                
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
| TCSSC-LSTM        | 93.95%  | 93.95%       | 1.3%  |
| TCSSC-Transformer | 93.89%  | 93.86%       |  1.04%  |

ASR turun 97.78% relatif terhadap baseline.

## Dataset

Dataset final (`data/tcssc_dataset.csv`) 100% data real dari sumber publik terverifikasi — tanpa augmentasi sintetis, tanpa pseudo-label. Sudah dedup exact (`tool_calls_json` lintas semua source) dan di-balance (undersampling ke target real kelas minoritas, toleransi rasio 1.15).

| Sumber                                       | Jumlah final     | Label                                                                                 |
| -------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------- |
| WildJailbreak                                | 2.121            | benign / sequential_attack                                                            |
| APIGen-MT-5k (Liu et al., NeurIPS 2024)      | 2.120            | benign — function-calling dialog terverifikasi otomatis, pipeline APIGen, Salesforce |
| BIPIA                                        | 1.747            | parameter_injection                                                                   |
| ToolSafe SafetyBench                         | 1.004            | Multi-kelas                                                                           |
| InjecAgent                                   | 829              | direct_attack / parameter_injection                                                   |
| SafeToolBench                                | 691              | direct_attack / parameter_injection                                                   |
| RedCode-Exec (Guo et al., NeurIPS 2024)      | 656              | direct_attack — eksekusi code/command risky terverifikasi manual                     |
| STAC Benchmark                               | 439              | sequential_attack                                                                     |
| R-Judge                                      | 332              | Multi-kelas                                                                           |
| AgentDojo (Debenedetti et al., NeurIPS 2024) | 250              | parameter_injection — agent hijacking via indirect prompt injection, ETH Zurich      |
| ToolSafe AgentHarm                           | 53               | Multi-kelas                                                                           |
| **Total**                              | **10.242** |                                                                                       |

Distribusi label: benign 2.654, parameter_injection 2.654, sequential_attack 2.626, direct_attack 2.308 (imbalance ratio 1.15).

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
├── notebooks/
│   ├── eda/
│   │   └── eda.ipynb
│   ├── core/
│   │   ├── tcssc_lstm.ipynb
│   │   ├── tcssc_transformer.ipynb
│   │   └── tcssc_reasoner.ipynb
│   └── ablations/                <- model perbandingan (Model 1, 4, 5, 6) 
├── archive/
│   └── legacy_scripts/           <- scraper.py, augmentor.py, verify_dataset.py, dst 
├── data/
│   ├── dataset.py
│   ├── split_utils.py
│   ├── export_for_kaggle.py
│   ├── tcssc_dataset.csv
│   ├── splits/
│   │   └── tcssc_split.json
│   ├── raw/                      
│   │   ├── scraped_github.csv
│   │   ├── STAC_benchmark_data.json
│   │   ├── ToolSafe/              <- clone repo ToolSafe
│   │   ├── RedCode/               <- clone repo RedCode
│   │   ├── agentdojo/, InjecAgent/, R-Judge/, ToolSafe/
│   │   └── labels_batch_*.json
│   └── processed/
│       ├── safetoolbench/        ← 8 file query_*.json
│       └── wildjailbreak.parquet
├── outputs/
│   ├── eda/                      
│   ├── model1_textonly_bert/
│   ├── tcssc_lstm/
│   ├── tcssc_transformer/
│   ├── model4_perturn/
│   ├── model5_fusion_text_sequence/
│   ├── model6_cnn_lstm/
│   └── tcssc_reasoner/         
├── checkpoints/                 
└── LICENSE
```

## Struktur Notebook

- `notebooks/eda/` — eksplorasi/analisis dataset
- `notebooks/core/` — 3 notebook model inti TCSSC (LSTM, Transformer, Reasoner)
- `notebooks/ablations/` — notebook model perbandingan dari Kaggle (Model 1, 4, 5, 6).
- `archive/` — hasil eksperimen dan script dari iterasi dataset versi lama, disimpan untuk referensi, bukan dipakai lagi 

## Cara Menjalankan

`data/tcssc_dataset.csv` (dataset final, sudah jadi) ikut di-commit di repo ini. buat training/eval, cukup install dependency dan lanjut ke langkah 3. Langkah 1-2 cuma perlu kalau mau re-generate dataset dari sumber mentah.

**Catatan untuk Langkah 2:** Repositori yang di-clone di bawah ini hanya sebagian dari total sumber data. Sisa sumber data lainnya (seperti WildJailbreak, APIGen-MT-5k, BIPIA, STAC Benchmark, dll.) akan diunduh secara otomatis melalui API (seperti HuggingFace) ketika skrip `dataset.py` dijalankan.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Opsional) Re-generate dataset dari sumber mentah
git clone https://github.com/MurrayTom/ToolSafe data/raw/ToolSafe
git clone https://github.com/AI-secure/RedCode.git data/raw/RedCode
git clone https://github.com/ethz-spylab/agentdojo data/raw/agentdojo
git clone https://github.com/uiuc-kang-lab/InjecAgent.git data/raw/InjecAgent
git clone https://github.com/Lordog/R-Judge data/raw/R-Judge
#    lalu jalankan:
python data/dataset.py
# output: data/tcssc_dataset.csv

# 3. Training — jalankan notebook di Kaggle (GPU), upload tcssc_dataset.csv + tcssc_split.json
#    sebagai Kaggle Dataset, lalu run salah satu:
#      notebooks/core/tcssc_lstm.ipynb
#      notebooks/core/tcssc_transformer.ipynb
#      notebooks/core/tcssc_reasoner.ipynb
#      notebooks/ablations/*.ipynb
```

## Scraping Data Baru (Archive)

Pipeline scraping GitHub/Reddit + pseudolabeling sudah diarsipkan ke
`archive` (lihat README di dalam folder itu untuk alasannya).
Tidak dipakai lagi di Fungsi pengumpulan data karena semua source di dataset final
sekarang dari benchmark publik terverifikasi, bukan hasil scraping/pseudolabel.

## Referensi

```
[1]  Cartagena & Teixeira. Mind the GAP. arXiv:2602.16943, 2026.
[2]  Li et al. STAC. arXiv:2509.25624, 2025.
[3]  ToolSafety. EMNLP 2025.
[4]  Mazeika et al. AgentHarm. ICLR 2025.
[5]  Inan et al. LLaMA Guard. Meta AI, 2023.
[6]  Hu et al. LoRA. ICLR 2022.
[7]  Dettmers et al. QLoRA. NeurIPS 2023.
[8]  Jiang et al. WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially)
     Safer Language Models. arXiv:2406.18510, 2024. (sumber dataset WildJailbreak)
[9]  Yi et al. Benchmarking and Defending Against Indirect Prompt Injection Attacks
     on Large Language Models. arXiv:2312.14197, 2023. (sumber dataset BIPIA)
[10] Zhan et al. InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated
     Large Language Model Agents. ACL Findings 2024, arXiv:2403.02691.
[11] Xia et al. SafeToolBench: Pioneering a Prospective Benchmark to Evaluating Tool
     Utilization Safety in LLMs. EMNLP Findings 2025, arXiv:2509.07315.
[12] Yuan et al. R-Judge: Benchmarking Safety Risk Awareness for LLM Agents.
     ICLR 2024, arXiv:2401.10019.
```

## Lisensi

MIT License untuk keperluan penelitian akademik.
