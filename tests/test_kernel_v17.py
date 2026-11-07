"""v17 regression guards: continuation init, gentle LR, short cap, clean-data gates."""
import re
from pathlib import Path

SRC = (Path(__file__).parent.parent / "scripts" / "kaggle_kernel_v17.py").read_text()


def test_json_import_present():
    assert re.search(r"^import json$", SRC, re.M)


def test_continuation_params():
    assert "MAX_STEPS = 250" in SRC
    assert "EVAL_EVERY, WARMUP, LR = 2, 8, 125, 50, 5e-6" in SRC
    assert "LR = 5e-7" in SRC
    assert re.search(r"^REPEAT = 6", SRC, re.M)


def test_checkpoint_mounted():
    assert "_find_ckpt" in SRC and "v15best" in SRC
    assert "__nonexistent_no_checkpoint__" in SRC  # fallback guard stays


def test_clean_manifest_and_decontamination_kept():
    assert 'KURRENT_EXPECT_CELLS", "301"' in SRC
    assert '"Gerson Mosbacher"' in SRC
    assert '{"33739", "35238", "38666", "38678", "38693", "38696"}' in SRC
