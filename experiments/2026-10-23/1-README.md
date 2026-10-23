# Experiment records, 2026-10-23 (Eschau Kurrent fine-tune)

Machine-readable record of the 2026-10-23 Kaggle experiment series on
`dh-unibe/trocr-kurrent` (MIT), reading 19th-century Kurrent register table
cells for the Eschau 1081 registers (JewishGen JUF archive; all persons
long deceased; no living-person data, no secrets, no model weights here).

## Files

- `run-log.json` - per-run config, GPU seconds, measured metrics, verdicts,
  and the metric definitions (which sets are validation vs reserved test).
- `target-cells-retained.jsonl` - the 148 clean verified target cells used in
  the v7/v8 experiments (IdNr, kind, file, bbox, text). Labels come from the
  JUF archive index / blind-quorum / clearly printed reads only; no model
  self-labeling.
- `target-cells-excluded.json` - the 22 of 170 candidate cells dropped after
  crop forensics (19 wrong-column s02/s03 date cells, 2 suspect s02
  occupation cells, 1 empty g06 name band), plus 1 corrected bbox (38796).
- `eval_names_v4.jsonl` - the frozen keep-best validation set (held-out;
  IdNrs 33739/35238 never trained on).
- `../../scripts/kaggle_kernel_v4.py` - v5 full mixed run source.
- `../../scripts/kaggle_kernel_v7_diag.py` - v6/v7 diagnostic source.
- `../../scripts/kaggle_kernel_v8.py` - v8 mix-oversample source (running).

## Regeneration pointers (hosted copies, not independent backups)

- Kaggle kernel: reihlinstinct/kurrent-timing-run-1 (versions 5-8).
- Kaggle datasets: kurrent-code v9 (all manifests + eval files +
  colab_train_v*.py), kurrent-train-v1/v1b/v1c/v1d (page scans + MM lines).
- Colab notebook (user-run): Google Drive file id
  100KwKIvcXiHFBJ0KOxLVcxLmQD1BuyCv.
- Source scans: https://files.juf.stadtarchiv-digital.de/quellen/eschau/standesregister/1081/{s,g,h}NN_22.jpg

## Known preservation gap

Model weights (v5/v7 best checkpoints) exist only in the Kaggle kernel
Output tab: the account's API tokens currently refuse writes (cause
unknown), so no independent checkpoint backup exists. Metrics and configs
are preserved here instead.
