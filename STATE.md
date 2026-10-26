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
