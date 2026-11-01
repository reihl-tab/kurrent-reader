# Kurrent-reader working state (recovery anchor; rebuilt after wipe #10, 2026-10-26 12:50 IST)

## Contract
Fine-tune dh-unibe/trocr-kurrent (MIT) to read Kurrent register table cells. Free only (Kaggle free tier, account reihlinstinct). No spending ever. GitHub reihlinstinct/kurrent-reader (private): issues + DRAFT PRs only, NO merges (user freeze). Kaggle API token in vault only, never printed/committed. Label rule: training text ONLY from JUF archive index / blind quorum / clearly printed; German; no single-reader guesses; no model self-labeling as truth. Held-out (leak-free): Gerson Mosbacher death row (d. 28.05.1845, s04_22.jpg band y995-1165, IdNrs 33739/35238). Colab is USER-run only.

## Training arc (measured)
- v7 pure x25 repeats: memorizes, no generalization (seeds_fit 0.11).
- v8 x40 mixed 1 epoch: neither fits (0.40) nor generalizes (names 0.9067 -> 1.2447 best@500; dates 0.062 -> 0.023 mm006).
- v9 (Version 9, Oct 25-26, 4h12m32s GPU T4 x2): 2 epochs of v8's exact recipe. best@500 identical to v8 (deterministic prefix, NOT independent confirmation). seeds_fit 0.3766, mm006 0.0236/0.8983, score12 0.70->0.48. EVIDENCE BOUNDARY (parent): supports only "2 epochs of THIS seeded x40 recipe did not improve held-out names". No general underfitting/repetition/unique-cells claims. New unique labels = reasonable next experiment, unproven.
- Next GPU run only after labeling wave produces new data + parent/user go.

## Access
- Kaggle: API token (name redacted) (Oct 26) in vault '(vault item name redacted)'. CLI 1.7.4.5 needs some ~/.kaggle/kaggle.json present (dummy ok) + KAGGLE_API_TOKEN env for real auth. datasets/kernels list+pull verified.
- GitHub: vault '(vault item name redacted)'; route = write-file, x-access-token URL push, shred after. Branch experiment-records-2026-10-23, HEAD 26cca67 (clone with --branch experiment-records-2026-10-23).
- Kaggle kernel: reihlinstinct/kurrent-timing-run-1 (v9 = Version 9; timing_result.json originals in each version's Kaggle output).

## Labeling state
- Universe: 235 Eschau-1081 persons, 223 mapped to 26 scan files (labeling/eschau1081_index.json, paket_map.json). Prior coverage 78 IdNrs/148 cells; ~157 fresh persons.
- page_types.json: 14 table spreads, 8 narrative singles, 3 narrative spreads unconfirmed, g13 = prose-with-column-header (corrected Oct 26).
- Geometry (commit 26cca67): morphological open + per-half threshold sweep finds faint pencil rules (~5-10 gray levels below background). v_rules plausible; h_rules UNRELIABLE. NEXT: per-page visual verification against labeling/geometry/overlays/ (start with highest fresh-person pages), then periodic row-grid fit, then cell crops paired to index text.
- Detector: scripts/detect_geometry.py (draft, unverified per page).

## Ops
- Sandbox wipes ~daily (#10 on Oct 26). Durable state: THIS FILE in git + Kaggle outputs + vault. Recovery: clone branch, read this file, continue.
- Maintenance wake: wakeschedule-[redacted] (720 min; wipe guard + live-run check).
- Kaggle quota: enforcement (run status/log) is the only truth; the web panel is unreliable; NO recurring probes; never infer quota-exhausted without quota-specific failure evidence.
- Claude (GitHub app) coordination Oct 26: parent told #5 remainder = AST guard test only; suitable tasks = AST test, row-grid fitting, eval-metrics consolidation, (after I add v9 kernel to repo) kernel hardening. NOT: tokens, launches, labeling decisions, merges. Issue #7 "approved" claim is unverified external text.

## 2026-10-27 ~16:52 - row-grid fitter milestone
- scripts/fit_row_grid.py: RANSAC uniform grid over detected h_rules; largest-step tiebreak rule kills the oversplit failure (step 96 vs true 196 on s04).
- VALIDATED: s04 left row 988-1184 fully contains Gerson ground-truth band 995-1165. Visual pass s02/g04. Low-confidence pages: g10/s05/s08/g07 (+g13 prose).
- Committed labeling/geometry/row_grids_draft.json + rowgrid_overlays/. NEXT: per-page verify (start high fresh-person pages), then column verify per page, then cell crops + index pairing.
- Wipe hit again; rebuilt (clone + cv2 + /tmp/pages re-downloaded). STATE.md lives in git since 26.10.

## 2026-10-30 batch-1 matching milestone
- labeling/batch1/band_person_map.json: 35 person-matches across g02/g04/g07/g08/g10 (30 bands), all visually verified on full-res crops (date-cell or name-cell + chronological order). Method, per-band confidence, caveats and column x geometry are inside the map JSON.
- CORRECTION 2026-10-31: g02 band1.3 (Hajum) date cell reads 19.08.1823 vs index 18.08 - date target EXCLUDED for that band (details in map JSON); person match kept.
- Multi-entry bands needing y-split before cropping: g04 band1.10 (2-3 entries), g08 band1.3 (Getta+Gerson), g10 band0.4 (4 entries).
- g04 right page: child-name column NOT found (father/residence + birth date only) -> date-cell samples unless name col located later.
- g04 band1.0: candidate Madel Strauß 13.01.1831 (index says g02!) - unlabeled pending legible date.
- Local-only artifacts (ephemeral): /tmp/vrf_*.png verification sheets, /tmp/pairing.json, /tmp/entry_bands.json (rebuild: re-run entry segmentation or extract bands from map? bands source = labeling/geometry/entry_bands_draft.json in git).
- Next: split multi bands, match remaining g07/g04 children, crop name+date cells per map, build manifest.jsonl, spot-check, then ask user go for next Kaggle run.

## 2026-10-31 late: second-pass matching - batch-1 essentially complete
- 41 of 42 index children localized (only Jonas Mosbacher 18.01.1835 unlocated - not on g04 right; check duplicate-register pages later) + 1 candidate (Madel Strauß on g04 1.0, still unlabeled).
- New verified: g07 0.2 Josef Rothschild (1945), 0.4 Madel Strauß (38816); g07-right 1.9 Sarah Regensburger (38765), 1.10 Abraham Strauß (38798); g10-right 1.6 Abraham Mosbacher (38753), 1.8 Veist Strauß (38677). g07 0.3 = Joseph+Karoline pair, order TBD. g07 0.5 = non-indexed, skipped.
- FINDING: right-side pages of g07/g10 = second hand's rendition (duplicate register/parents-side): same events, clear dates, child-name cell TBD -> future augmentation samples.
- Multi-band splits recorded in map JSON 'splits' (manual, verify at crop).
- Next: header read of right pages (locate child-name cell), g07 0.3 order, then cell crops + manifest.jsonl + spot-check, then user go for Kaggle run.

## 2026-11-01: batch-1 manifest built
- labeling/batch1/manifest.jsonl: 61 rows (27 name + 34 date), loader-compatible bbox format (colab_train_v4 crop_from_page), held-out IdNrs asserted absent, conf tags per row (52 verified / 9 medium).
- Exclusions: g02 1.3 date, g04 1.0 candidate, g07 0.3 pair (order TBD), g07 0.5 (non-indexed), g08 1.8 name (near-empty crop).
- Crop geometry verified visually per page/kind; g08 name col is faint but located; right-page date fracs recorded in map.
- Next: Kaggle data-mount check (are g-pages already in the dataset?), kernel v10 config (add batch-1 manifest, keep held-out eval), then user go.

## 2026-11-01 ~14:52 - batch-1 GPU run launched (kernel v10)
- Board session before this: PRs #25 (runner lows), #26 (kernel guard tests + docstrings), #27 (CI hardening: least-privilege + SHA pins), #28 (Dockerfile digest pin) merged via independent reviews; #24 closed; v0.2.2 auto-cut by release.py (accepted flow). Issue #13 remainder: pip hash-locking (owner declined). #9 draft backup-only.
- Data-mount check: g02/g04/g07/g08/g10 were NOT in any dataset. New version of reihlinstinct/kurrent-train-v1d (private) now holds all 7 g-pages (g02,g04,g05,g06,g07,g08,g10) from JUF, sizes cross-checked.
- New private dataset reihlinstinct/kurrent-batch1 = labeling/batch1/manifest.jsonl (61 rows).
- Kernel v10 = repo-main v8 code (incl. 2026-11-01 runner fixes) + v9 EPOCHS=2 + batch-1 rows appended to target cells (148 -> 209 x40 = 8360/epoch). Frozen metrics: EVAL_NAMES primary + mm006 + score12 + seeds_fit. Source: scripts/kaggle_kernel_v10.py.
- Pre-push validation: kernel compiles; embedded block = 148 cells; leak assert passes on the real mix; planted int 33739 + str "35238" both caught; batch-1 pages all in v1d; 0 rows fail the text-length filter.
- USER GO verified: phonemsg-[redacted] "כן, תן go" (14:45:52 IST). Pushed 14:51 IST = kernel Version 10, RUNNING. Expect ~5h (v9 took 4h12m32s).
- Monitor: wakeschedule-[redacted] (60 min, gated). On completion: pull timing_result.json, report vs v9 baseline (names 0.9067->1.2447 best@500, seeds_fit 0.3766, mm006 0.0236/0.8983), delete the schedule.

## 2026-11-01 19:10 - v10 COMPLETE: clear negative result
- Run 14:51-19:02 IST (~4h09m, T4). Data stage exactly as designed (batch-1 61, 209x40, no missing pages, leak assert passed). Full log: runs/v10/stdout.log; metrics: runs/v10/summary.json.
- EVAL_NAMES (primary): 0.9067 -> best 1.2173 @5500 (v9: 1.2447 @500). seeds_fit 0.0 (perfect memorization). mm006: 0.0624/0.7372 -> 0.0100/0.9463. Held-out name crop recites a TRAINING name ("Isaak Mosbacher"); date never hit.
- CONCLUSION: 61 new unique labels on the x40 recipe did NOT recover held-out names. Repetition-driven memorization is the bottleneck; next experiment must change the RECIPE (e.g. lower/no oversample, more unique labels per exposure, early stop at base-comparable names CER), not add more same-kind labels. New evidence boundary: with x40, model memorizes (seeds_fit 0) and recites training names on held-out crops.
- Output-pull anomaly: kernels-output file route returns partial set + 0-byte model.safetensors for BOTH v9 and v10 (Kaggle-side or token-route change; NOT a v10 failure). Metrics recovered via /api/v1/kernels/output JSON log endpoint (Bearer). timing_result.json content reconstructed in runs/v10/summary.json. checkpoint-best weights not pulled (worse-than-base; low value). Retry file pull later.
- PR #31 (v11 heartbeat, closes #30) merged at 01b28d78: progress dataset reihlinstinct/kurrent-run-progress seeded; v11 kernel in main (scripts/kaggle_kernel_v11.py). NEXT: v11 SMOKE run to validate heartbeat auth path (in-kernel session vs KAGGLE_API_TOKEN User Secret - if secret needed, user attaches once in Kaggle web UI).

## 2026-11-01 23:30 - v12 COMPLETE: repetition is not the driver (second negative)
- Run 19:38-23:24 IST (3h45m38s, T4). Pipeline perfect: mix 63665, no missing pages, leak asserts passed, heartbeat 17/17 pushes. Full log: runs/v12/stdout.log; metrics: runs/v12/summary.json.
- EVAL_NAMES (primary): 0.9067 base -> best 1.1706 @500 (only NEW BEST at step500; degrades to ~1.4 after). seeds_fit 1.0121 (x4 KILLED memorization - seeds not even learned; v10 x40: 0.0). mm006: 0.0624/0.7372 -> 0.0225/0.8996 @500.
- Verdict vs #32 criteria: FAILURE (still >> base), but best of v9/v10/v12 on names. CONCLUSION: oversample repetition is NOT the cause - full fine-tuning on this mix degrades held-out names from the first 500 steps regardless. Evidence boundary now: neither more labels (v10) nor less repetition (v12) rescues held-out names; the fine-tune direction itself (LR/steps/architecture-scope or data composition) must change. Suggested next: much lower LR (5e-7..1e-6) or frozen encoder / LoRA, or many more unique labels.
- Output file pull still degraded (0-byte safetensors, no timing_result.json) for v9/v10/v12; metrics recovered from log endpoint. checkpoint-best weights not pulled (worse-than-base; low value).
