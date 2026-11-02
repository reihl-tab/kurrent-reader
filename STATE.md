# Kurrent-reader working state (recovery anchor; rebuilt after wipe #10, 2026-10-26 12:50 IST)

## Contract
Fine-tune dh-unibe/trocr-kurrent (MIT) to read Kurrent register table cells. Free only (Kaggle free tier, account reihlinstinct). No spending ever. GitHub reihlinstinct/kurrent-reader: merges to main only under the standing review gate. Kaggle API token kept in a credential vault, never printed/committed. Label rule: training text ONLY from JUF archive index / blind quorum / clearly printed; German; no single-reader guesses; no model self-labeling as truth. Held-out (leak-free): Gerson Mosbacher death row (d. 28.05.1845, s04_22.jpg band y995-1165, IdNrs 33739/35238). Colab is USER-run only.

## Training arc (measured)
- v7 pure x25 repeats: memorizes, no generalization (seeds_fit 0.11).
- v8 x40 mixed 1 epoch: neither fits (0.40) nor generalizes (names 0.9067 -> 1.2447 best@500; dates 0.062 -> 0.023 mm006).
- v9 (Version 9, Oct 25-26, 4h12m32s GPU T4 x2): 2 epochs of v8's exact recipe. best@500 identical to v8 (deterministic prefix, NOT independent confirmation). seeds_fit 0.3766, mm006 0.0236/0.8983, score12 0.70->0.48. EVIDENCE BOUNDARY (parent): supports only "2 epochs of THIS seeded x40 recipe did not improve held-out names". No general underfitting/repetition/unique-cells claims. New unique labels = reasonable next experiment, unproven.
- Next GPU run only after labeling wave produces new data + parent/user go.

## Access
- Kaggle: API token (Oct 26) held in a credential vault (item name redacted). CLI 1.7.4.5 needs some ~/.kaggle/kaggle.json present (dummy ok) + KAGGLE_API_TOKEN env for real auth. datasets/kernels list+pull verified.
- GitHub: PAT held in a credential vault (item name redacted); route = write-file, token-URL push, shred after. Branch experiment-records-2026-10-23, HEAD 26cca67 (clone with --branch experiment-records-2026-10-23).
- Kaggle kernel: reihlinstinct/kurrent-timing-run-1 (v9 = Version 9; timing_result.json originals in each version's Kaggle output).

## Labeling state
- Universe: 235 Eschau-1081 persons, 223 mapped to 26 scan files (labeling/eschau1081_index.json, paket_map.json). Prior coverage 78 IdNrs/148 cells; ~157 fresh persons.
- page_types.json: 14 table spreads, 8 narrative singles, 3 narrative spreads unconfirmed, g13 = prose-with-column-header (corrected Oct 26).
- Geometry (commit 26cca67): morphological open + per-half threshold sweep finds faint pencil rules (~5-10 gray levels below background). v_rules plausible; h_rules UNRELIABLE. NEXT: per-page visual verification against labeling/geometry/overlays/ (start with highest fresh-person pages), then periodic row-grid fit, then cell crops paired to index text.
- Detector: scripts/detect_geometry.py (draft, unverified per page).

## Ops
- Sandbox wipes ~daily (#10 on Oct 26). Durable state: THIS FILE in git + Kaggle outputs + vault. Recovery: clone branch, read this file, continue.
- Maintenance wake: a 720-min recurring wake (wipe guard + live-run check); schedule ID tracked outside the repo.
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
- USER GO verified on the owner's authenticated chat channel 14:45:52 IST (message ID redacted). Pushed 14:51 IST = kernel Version 10, RUNNING. Expect ~5h (v9 took 4h12m32s).
- Monitor: a 60-min gated wake (schedule ID redacted). On completion: pull timing_result.json, report vs v9 baseline (names 0.9067->1.2447 best@500, seeds_fit 0.3766, mm006 0.0236/0.8983), delete the schedule.

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

## 2026-11-02 05:05 - EVIDENCE-BOUNDARY CORRECTION on the v12 write-up
- The 23:30 conclusion overclaimed. What v12 actually supports: this x4/full-fine-tune recipe is rejected on the 7 held-out name crops (best 1.1706 vs 0.9067 base, degradation from step500). What it does NOT support: 'repetition is not the driver' or 'full fine-tuning itself is the cause' - REPEAT 40->4 also changed target exposure 10x, and seeds_fit 1.0121 shows lack of fit, not cause isolation. LR / freezing / unique-label count remain competing hypotheses. Held-out signal = only 7 crops; carry that limit into any v13 design. (User was told the stronger claim 23:29; parent will correct on his reply.)

## 2026-11-02 09:40 - v13a (lowlr) COMPLETE: still worse than base, best of four runs
- Run 05:56-09:37 IST (~3h37m, T4; kernel Version 14 after Version-13 boot crash + hotfix #40). Pipeline clean: mix 63665, 209 cells, no missing pages, leak asserts passed. Full log: runs/v13a/stdout.log; metrics: runs/v13a/summary.json.
- Recipe: GENTLE=lowlr, LR 5e-6 -> 1e-6, no frozen params, v12's exact data mix otherwise.
- EVAL_NAMES (primary): 0.9067 base -> best 1.0005 @1000 (NEW BESTs at 500/1000, then ~1.16-1.21). word_acc 0.0; date 28/1845 never hit. seeds_fit 0.951. mm006: 0.0624/0.7372 -> 0.0281/0.8832 @1000.
- Verdict vs #37: low-LR did NOT recover held-out names (1.0005 vs base 0.9067) but is the best of v9/v10/v12/v13a, best point later (1000 vs 500). Interpretation: consistent with optimization pressure contributing; cause NOT isolated (LR changes update magnitude on same exposure). 7-crop limit stands. Remaining designed lever: v13b freeze-encoder.

## 2026-11-02 12:15 - v13b (freeze-encoder) COMPLETE: gentle pair finished, both worse than base
- Run 09:56-12:09 IST (~2h09m, T4; kernel Version 15). Pipeline clean: mix 63665, 209 cells, no missing pages, leak asserts passed. Full log: runs/v13b/stdout.log; metrics: runs/v13b/summary.json.
- Recipe: GENTLE=freeze-encoder (164 encoder tensors frozen, 247.3M/333.9M trainable), LR 5e-6, same data mix as v12/v13a.
- EVAL_NAMES (primary): 0.9067 base -> best 1.0365 @500 (single NEW BEST; then ~1.29-1.44). word_acc 0.0; date 28/1845 never hit. seeds_fit 1.0562. mm006: 0.0624/0.7372 -> 0.0316/0.8714 @500.
- Verdict vs #37: freezing the encoder did NOT recover held-out names; essentially tied with v13a (1.0365 vs 1.0005). Both gentle levers fail on the 7 held-out crops while mm006 improves and seeds fit. Weight shifts to data composition (unique labels) or train/eval mismatch, not optimization mechanics. Cause NOT isolated; 7-crop limit stands.

## wave-2 labeling progress (2026-11-02 ~12:55 IST)
- s02 done: 8 persons / 16 cells; s03 done: 5 persons / 10 cells; s05 parked (faint left page, CLAHE works, 9 labelable, 38666 held-out).
- g05 done: 19 cells (9 name + 10 date) for 11 of 12 index persons. KEY LAYOUT FINDING: g-register (births) entries span the gutter like s-register: child name + gender on LEFT page (x160-400), birth date + baptism/rabbi on RIGHT page (date ink x2050-2470). Pair by same physical Y; half grids drift (L period 191 origin 171; R period 179 origin 35) and grid rules can cut ink (Rebecca date y170-370 crosses R0/R1 rule; Abraham entry below last L rule). Twin row L4 = Josef Leopold + Joseph (one cell, labeled Josef Leopold, twin noted). Giedel name cell SKIPPED (index 'Grünewald geb. Strauß', birth surname ambiguous) - date cell kept. 38103/38104 dup labeled once.
- Wave-2 running total: 45 new cells (s02 16 + s03 10 + g05 19). Projected training total if merged: 209 + 45 = 254.

## wave-2: g06 done (2026-11-02 ~13:09 IST)
- g06: 20 cells (10 name + 10 date) for 10 of 11 index persons; 38039 Clara ('um 1806') = index scan_file mis-mapping, skipped. Same gutter-spanning g-register layout as g05 (left: name/gender/father/mother; right: remarks/L.G./date/notes), pair by physical Y. Fanny labeled with index geb. name 'Fanny Mosbacher' (ink reads Mosbacher; index primary = married name Goldschmidt). Leopold Rothschild: index gives 'Leopold, Löb', ink reads Löb/Leopold.
- Wave-2 running total: 65 new cells (s02 16 + s03 10 + g05 19 + g06 20). Projected training total if merged: 209 + 65 = 274.

## wave-2: s06 done (2026-11-02 ~13:22 IST)
- s06: 16 cells (8 name + 8 date) for all 8 index persons (deaths 1856-1860). Left page = identity rows (name col x140-560), right page = Gestorben col (x1290-1530) + Begraben + age + remarks. Paired by order + age cross-check vs index gebdatum (infant Gerson '6 Monate', Sara 14, Seligmann 72, Mattche 82 - all exact). Empty spacer rows on both halves. No held-out persons on page.
- Wave-2 running total: 81 new cells. Projected training total if merged: 209 + 81 = 290.

## wave-2: s07 done (2026-11-02 ~13:25 IST)
- s07: 12 cells (6 name + 6 date) for 6 of 9 index persons; HELD-OUT excluded: 38678, 38693, 38696 (bands identified and skipped). Two infant 'Jeanette Strauß' (1869 entries #1/#2) assigned by chronological order + date reads (38694=08.02, 38695=11.02). Non-index spacer entries skipped. All conf=medium (page hand is loose; sequence+age+name-shape consistent).
- Wave-2 running total: 93 new cells. Projected training total if merged: 209 + 93 = 302.

## wave-2: s08 done (2026-11-02 ~13:30 IST)
- s08: 10 cells (5 name + 5 date) for 5 persons; faint page read via CLAHE. Mendel Mosbacher (3763/3764, d.1895) NOT on page - index scan_file mis-mapping, skipped. Siegmund's date cell is struck through but legible ('31. Mai 1870', weekday consistent) - kept with note. Gestorben col on s08 is wide (x1300-2170, year floats top-right). No held-out persons on this page.
- Wave-2 running total: 103 new cells. Projected training total if merged: 209 + 103 = 312.

## wave-2: s05 done - CHUNK 1 COMPLETE (2026-11-02 ~13:32 IST)
- s05: 20 cells (10 name + 10 date) for 10 persons; faint left page read via CLAHE. ALL 10 date cells weekday cross-checked against historical calendar (all pass). 38649/38650 same person labeled once. HELD-OUT 38666 excluded (band identified).
- CHUNK 1 COMPLETE: s02,s03,g05,g06,s06,s07,s08,s05 = 123 new cells (name+date). Projected training total if merged: 209 + 123 = 332.
- Notable calls: s07 twin infant Jeanettes assigned by chronological order; s08 Siegmund date cell struck-through but legible (kept, noted); s08 Mendel 1895 absent = index mis-mapping; g05 Giedel name skipped (unreadable birth surname).

## wave-2: chunk-2 start - narrative-page format discovery (2026-11-02 ~13:36 IST)
- All 12 remaining pages re-fetched to /tmp/pages (URL pattern https://files.juf.stadtarchiv-digital.de/quellen/eschau/standesregister/1081/NN_22.jpg): g01,g09,g11,g12,g13,h01-h06,s01.
- Table-spread leftovers resolved against batch1 manifest: g08/g10 fully labeled already; NEW cells available only g02 (38034 Madel Strauß - batch1 flagged DO-NOT-LABEL, day/month illegible; possibly on g04 band 1.0), g04 (38096 Jonas 18.01.1835), g07 (38744 Karoline 12.07.1849, 38797 Joseph 28.05.1849 = the old order-TBD pair). Parent-gen persons (vor 1790/no dates) stay skipped per name+DATE pair rule.
- g01 CONFIRMED narrative prose (single page 2149x1510): numbered entries per year, "N) Den <date> ... ist dem hiesigen Bürgersmann <father> und seiner Ehefrau <mother> ... ein Sohn/eine Tochter geboren worden, welche(r) den Namen <NAME> erhielt."
- g01 entry map (all visually read): 1821#1 = Beile Marx 38736 (date numeral 27-vs-29 AMBIGUOUS at current zoom - needs tighter zoom; parents Marx Joseph+Frieda match index), 1821#2 = TWINS Joseph 33729 + Merjem 38747 Mosbacher 23.11.1821 (one band, two name phrases, father Gerson/Gershon matches), 1822#1 = Beile Michel 38737 11.06.1822, 1822#2 = Julia Schönlein Strauß 38314 28.05.1822 ("Schönlein" ink visible), 1822#3 = Wolf Strauß 38813 29.07.1822 (mother Gudel 38812), 1822#4 = Vrommet Michel 38742 29.11.1822. Labelable = 7 children (name+date cells); parents-gen (38734,38735,38311,38811,38812) skip.
- Narrative labeling approach: name bbox = the name phrase in the "welche den Namen X erhielt" line; date bbox = the "Den <date>" phrase in line 1 of the entry; label text from index as usual. NEW visual domain (prose-inline phrases) - good for h-page generalization. Batch1 ambiguity precedent applies: ambiguous numeral => name cell only, date target excluded.
- Zooms saved: /tmp/wave2/g01_z1..z5.png (1.5x, regions z1 y380-700, z2 y640-1010, z3 y1000-1400, z4 y1350-1760, z5 y1700-2080).
- NEXT: view z4/z5, tighter zoom of 1821#1 numeral, assign g01 bboxes (7 name + up to 7 date), spot-check, commit; then g04 Jonas + g07 pair resolution; then g09/g11/g12/g13/h-pages format discovery.

## 2026-11-02 13:53 - v14 checkpoint LAUNCHED (owner-approved ONE short run) + held-out contamination found
- Owner go 13:44 (relayed by parent) for ONE short cheap checkpoint run on the wave-2 dataset, Kaggle free tier, held-out untouched.
- CONTAMINATION: the embedded 148-cell seed set (since v8) holds name+date cells for the 4 s05/s07 EVAL_NAMES persons (38666/38678/38693/38696) - person-level contamination of the occ-crop eval half in every fine-tune v8-v13. Eval crops themselves never trained; Gerson row clean; base baseline unaffected. Bias direction flatters fine-tunes, so worse-than-base verdicts likely stand but were measured dirty.
- v14 = v13-exact + decontamination (drop 8 cells -> 337 trained: 140 embedded + 61 batch1 + 136 batch2) + leak assert extended to all 6 held-out IdNrs + GENTLE=lowlr + MAX_STEPS=2000. New private Kaggle dataset kurrent-batch2 (136 cells + g01_22.jpg, which was missing from v1 mounts).
- Boot-crash rule: tests/test_kernel_v14.py - 7 behavioral tests exec the real v14 blocks vs tiny stand-ins; all pass pre-push. Pushed as kernel Version 16 (~30-40 min ETA).
