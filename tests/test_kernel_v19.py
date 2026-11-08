"""v19 regression guards: freeze-encoder variant, from-base, REPEAT 4, EVAL_EVERY 125, init-seeded names-CER selection, clean-data gates."""
import re
from pathlib import Path

SRC = (Path(__file__).parent.parent / "scripts" / "kaggle_kernel_v19.py").read_text()


def test_json_import_present():
    assert re.search(r"^import json$", SRC, re.M)


def test_v19_params():
    assert "MAX_STEPS = 500" in SRC
    assert "EVAL_EVERY, WARMUP, LR = 2, 8, 125, 100, 5e-6" in SRC
    assert re.search(r"^REPEAT = 4", SRC, re.M)


def test_freeze_encoder_variant():
    assert '"freeze-encoder")  # v19' in SRC
    assert "model.encoder.parameters()" in SRC
    assert "_p.requires_grad_(False)" in SRC


def test_init_seeded_names_selection():
    assert "basenames.get(\"cer\"" in SRC
    assert '"step": 0' in SRC
    assert "namesCER=" in SRC


def test_from_base_no_checkpoint_dataset():
    assert "kurrent-checkpoint-v15best" not in SRC


def test_clean_manifest_and_decontamination_kept():
    assert 'KURRENT_EXPECT_CELLS", "301"' in SRC
    assert '"Gerson Mosbacher"' in SRC
    assert '{"33739", "35238", "38666", "38678", "38693", "38696"}' in SRC
