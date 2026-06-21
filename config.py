"""Konfigurasi global project TCSSC — semua hyperparameter dan path ada di sini."""
import os

# Path dataset
DATA_DIR        = "data/"
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
STAC_PATH       = os.path.join(RAW_DIR, "STAC_benchmark_data.json")
TOOLSAFE_DIR    = os.path.join(RAW_DIR, "ToolSafe")
TOOLSAFE_AGENTHARM_DIR    = os.path.join(TOOLSAFE_DIR, "benchmark", "agentharm", "dataset")
TOOLSAFE_SAFETYBENCH_PATH = os.path.join(TOOLSAFE_DIR, "benchmark", "agent_safetybench", "data", "released_data.json")
WILDJAILBREAK_PATH        = os.path.join(DATA_DIR, "processed", "wildjailbreak.parquet")
SCRAPED_LABELED_PATH      = os.path.join(DATA_DIR, "processed", "scraped_labeled.csv")
AUGMENTED_PATH            = os.path.join(DATA_DIR, "processed", "augmented.csv")
RJUDGE_DIR                = os.path.join(RAW_DIR, "R-Judge", "data")
INJECAGENT_DIR            = os.path.join(RAW_DIR, "InjecAgent", "data")
AGENTDOJO_RUNS_DIR        = os.path.join(RAW_DIR, "agentdojo", "runs")
AGENTDOJO_PIPELINE        = "gpt-4o-2024-05-13"
REDCODE_EXEC_DIR          = os.path.join(RAW_DIR, "RedCode", "dataset", "RedCode-Exec")
OUTPUT_DIR      = "outputs/"
CHECKPOINT_DIR  = "checkpoints/"

# Label kelas
LABEL2ID = {
    "benign":              0,
    "direct_attack":       1,
    "sequential_attack":   2,
    "parameter_injection": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)

# Konfigurasi encoder
ENCODER_MODEL   = "bert-base-multilingual-cased"
MAX_SEQ_LEN     = 128   # panjang maksimal per tool call
MAX_SEQ_LEN_CTX = 64    # panjang konteks percakapan
EMBED_DIM       = 768   # output dim mBERT

# Konfigurasi sequence aggregator
MAX_TOOL_CALLS  = 20    # maksimal tool call per sesi
LSTM_HIDDEN     = 256
LSTM_LAYERS     = 2
LSTM_DROPOUT    = 0.3

TRANSFORMER_HEADS  = 8
TRANSFORMER_LAYERS = 4
TRANSFORMER_FF_DIM = 512
TRANSFORMER_DROPOUT = 0.1

# Konfigurasi harm classifier
CLASSIFIER_HIDDEN = 128
CLASSIFIER_DROPOUT = 0.3

# Training
BATCH_SIZE      = 32
LEARNING_RATE   = 2e-4
NUM_EPOCHS      = 20
WEIGHT_DECAY    = 1e-4
EARLY_STOP      = 5
SEED            = 42

# Device
DEVICE = "cuda"  # ganti "cpu" jika tidak ada GPU
