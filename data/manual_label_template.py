"""
Template semi-otomatis untuk manual labeling: buka Gemini di Chrome dan paste prompt otomatis.
User copy response JSON dari browser, simpan ke labels_batch_N.json, lalu Enter.
Usage: python data/manual_label_template.py
"""
import os
import sys
import json
import math
import time
import pandas as pd
import pyperclip
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_PATH = "data/raw/scraped_github.csv"
LABELS_DIR = "data/raw"
BATCH_SIZE = 20
GEMINI_URL = "https://gemini.google.com/app?hl=id"

PROMPT_HEADER = """\
Role: AI Safety Expert.
Task: Labeli tool call sequences berikut.
Kelas:
0 = irrelevant
1 = benign
2 = direct_attack
3 = sequential_attack
4 = parameter_injection

Format output: JSON array HANYA berisi "index" dan "label".
Contoh: [{"index": 0, "label": 2}, {"index": 1, "label": 0}]
Hanya JSON valid, tanpa markdown, tanpa penjelasan.

Data:
"""


def load_csv() -> pd.DataFrame:
    if not os.path.exists(INPUT_PATH) or os.path.getsize(INPUT_PATH) < 10:
        raise FileNotFoundError(f"File tidak ada atau kosong: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows dari {INPUT_PATH}")
    return df


def build_batch_items(df: pd.DataFrame, start: int, end: int) -> list:
    items = []
    for i, (_, row) in enumerate(df.iloc[start:end].iterrows()):
        items.append({
            "index":      i,
            "title":      str(row.get("title", ""))[:120],
            "body":       str(row.get("body",  ""))[:100],
            "tool_calls": str(row.get("tool_calls", "[]"))[:200],
        })
    return items


def build_prompt(items: list) -> str:
    return PROMPT_HEADER + json.dumps(items, ensure_ascii=False, indent=2)


def label_path(batch_idx: int) -> str:
    return os.path.join(LABELS_DIR, f"labels_batch_{batch_idx}.json")


def already_filled(batch_idx: int) -> bool:
    p = label_path(batch_idx)
    if not os.path.exists(p) or os.path.getsize(p) < 5:
        return False
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, list) and len(data) > 0
    except Exception:
        return False


def create_empty_json_files(n_batches: int):
    os.makedirs(LABELS_DIR, exist_ok=True)
    for i in range(n_batches):
        p = label_path(i)
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as f:
                f.write("[]")


def paste_prompt(driver, wait: WebDriverWait, prompt: str):
    pyperclip.copy(prompt)
    input_box = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div[contenteditable='true']"))
    )
    time.sleep(2)
    input_box.click()
    time.sleep(1)
    input_box.send_keys(Keys.CONTROL, "v")
    time.sleep(2)
    input_box.send_keys(Keys.ENTER)


def wait_for_user_json(batch_idx: int):
    out_path = label_path(batch_idx)
    abs_path = os.path.abspath(out_path)

    print(f"\n{'='*57}")
    print(f" Prompt batch {batch_idx} sudah dikirim ke Gemini.")
    print(f" Gemini sedang memproses... Copy response JSON dari browser.")
    print(f" Paste ke file:")
    print(f"   {abs_path}")
    print(f"{'='*57}")

    while True:
        ans = input("Setelah file JSON terisi, tekan Enter untuk lanjut (atau 'skip'): ").strip().lower()

        if ans == "skip":
            print(f"  Batch {batch_idx} di-skip.")
            return

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 5:
            print(f"  WARN: file masih kosong. Isi dulu lalu Enter lagi.")
            continue

        try:
            with open(out_path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list) or len(data) == 0:
                print(f"  WARN: JSON kosong atau bukan array. Paste response Gemini dulu.")
                continue
            print(f"  OK: {len(data)} label tersimpan untuk batch {batch_idx}.")
            return
        except json.JSONDecodeError as e:
            print(f"  WARN: JSON tidak valid — {e}. Perbaiki file dulu.")


def main():
    df        = load_csv()
    total     = len(df)
    n_batches = math.ceil(total / BATCH_SIZE)

    create_empty_json_files(n_batches)
    print(f"Template JSON: {n_batches} file siap di {os.path.abspath(LABELS_DIR)}/")

    done = [i for i in range(n_batches) if already_filled(i)]
    todo = [i for i in range(n_batches) if not already_filled(i)]

    if done:
        print(f"Sudah terisi: batch {done}")
    if not todo:
        print("Semua batch sudah dilabeli. Jalankan: python data/json_to_csv.py")
        return

    print(f"Akan diproses : batch {todo}\n")

    # Buka Chrome
    options = uc.ChromeOptions()
    driver  = uc.Chrome(version_main=148, options=options)
    wait    = WebDriverWait(driver, 30)

    # Batch pertama — buka Gemini dan tunggu login
    driver.get(GEMINI_URL)
    input("Silakan login Google jika diminta. Tekan Enter setelah siap di Gemini...\n")

    try:
        for i, batch_idx in enumerate(todo):
            start = batch_idx * BATCH_SIZE
            end   = min(start + BATCH_SIZE, total)
            items = build_batch_items(df, start, end)
            prompt = build_prompt(items)

            # Buka tab baru untuk setiap batch kecuali yang pertama
            if i > 0:
                driver.switch_to.new_window("tab")
                driver.get(GEMINI_URL)
                time.sleep(3)

            print(f"\nMengirim prompt batch {batch_idx}/{n_batches - 1} ke Gemini...")
            paste_prompt(driver, wait, prompt)

            wait_for_user_json(batch_idx)

    finally:
        try:
            driver.quit()
        except Exception:
            pass

    print(f"\nSemua batch selesai.")
    print(f"Jalankan: python data/json_to_csv.py")


if __name__ == "__main__":
    main()
