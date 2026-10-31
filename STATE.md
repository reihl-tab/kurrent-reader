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
