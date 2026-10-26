# v9 - 2-epoch underfitting test (2026-10-25)

Launched 19:47 IST via the Kaggle web editor (API tokens write-blocked since ~Oct 23; push 401 / pull 403).
Version 9 of reihlinstinct/kurrent-timing-run-1, GPU T4 x2, runtime 4h 12m 32s. From base.

Single change vs v8: `EPOCHS = 2` (same manifest, same embedded x40 cells, same frozen eval files).
Hypothesis tested: v8's low seeds_fit (0.40) meant undertraining, so more epochs would lift fit and
possibly recover EVAL_NAMES. Bounded result: WITH THIS RECIPE, two epochs did not improve held-out
names - EVAL_NAMES best stayed the step500 point (1.2447 vs base 0.9067), which equals v8's because
the data order is seed-deterministic (shared prefix, not independent confirmation); seeds_fit 0.3766
(v8: 0.4037); later steps 1.28-1.42. mm006 dates stable (0.0236/0.8983). score12 0.70 -> 0.48.
What this does NOT show: that underfitting is rejected in general, that repetition is exhausted as a
lever, or that new unique cells will fix names. New unique verified labels are a reasonable next
experiment (unproven).

timing_result_v9.json here is a HAND TRANSCRIPTION (see its _provenance). Run source: kernel_v9.py
(local v9push; = kernel_v8.py + header note + EPOCHS 1->2; push to repo deferred - kernel backups live
on branch colab-runner-versioned for v4/v8; v9 adds only the epochs constant).
