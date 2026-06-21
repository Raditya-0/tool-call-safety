# Archive

Cluster script dari strategi data collection/augmentasi awal yang sudah
ditinggalkan, digantikan oleh `data/dataset.py` (`collect_all_samples()` →
`dedup_records()` -> `balance_records()`).

- `scraper.py` —> scraping GitHub issues + Reddit, sumbernya pseudo-label
  (bukan benchmark terverifikasi)
- `manual_label_template.py` —> helper buat pseudolabeling manual hasil scraping
- `scraped_labeled.csv` —> hasil scraping + pseudolabeling di atas. Loader-nya
  (`load_scraped_labeled` di `data/dataset.py`) ada bug baca kolom
  `pseudo_label` yang sebenarnya tidak ada di CSV ini (kolom asli `label`),
  jadi semua baris selalu fallback ke `benign`. Sudah dilepas dari
  `collect_all_samples()`.
- `augmentor.py` —> strategi augmentasi sintetis (target fixed 2.500/kelas,
  substitusi sinonim nama API) yang ditinggalkan demi prinsip 100% data real
- `verify_dataset.py` —> sanity checker versi lama, masih pakai logic
  "target 10k + synthetic balance", tidak nyentuh pipeline dedup+balance
  ataupun `tcssc_dataset.csv`

Disimpan sebagai jejak, bukan untuk dipakai lagi. Kalau perlu sanity-check
dataset final, pakai langsung summary print di `data/dataset.py::__main__`.
