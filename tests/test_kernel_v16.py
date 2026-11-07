"""v16 regression guards: keep the levers and the clean-data gates honest."""
import re
from pathlib import Path

SRC = (Path(__file__).parent.parent / "scripts" / "kaggle_kernel_v16.py").read_text()


def test_json_import_present():
    # v15 Version 18 died on NameError: json not imported.
    assert re.search(r"^import json$", SRC, re.M)


def test_short_run_capped_at_productive_regime():
    assert "MAX_STEPS = 500" in SRC
    assert "EVAL_EVERY, WARMUP, LR = 2, 8, 250, 100, 5e-6" in SRC


def test_repeat_six():
    ns = {}
    for line in SRC.splitlines():
        m = re.match(r"^REPEAT = (\d+)", line)
        if m:
            ns["REPEAT"] = int(m.group(1))
    assert ns["REPEAT"] == 6


def test_clean_manifest_and_decontamination_kept():
    assert 'KURRENT_EXPECT_CELLS", "301"' in SRC
    assert '"Gerson Mosbacher"' in SRC
    assert '{"33739", "35238", "38666", "38678", "38693", "38696"}' in SRC
    assert "batch2_manifest.jsonl" in SRC


def test_from_base_not_checkpoint():
    assert "/kaggle/input/__nonexistent_no_checkpoint__" in SRC
