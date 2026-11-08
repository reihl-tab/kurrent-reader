"""Offline tests for scripts/check_install_pins.py (#63). No network."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_install_pins", ROOT / "scripts" / "check_install_pins.py"
)
pins = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pins  # dataclasses look the module up
SPEC.loader.exec_module(pins)

CPU = "https://download.pytorch.org/whl/cpu"
HASHED = "pip install --require-hashes -r requirements/dev.txt"
HASHED_CPU = f"pip install --require-hashes --extra-index-url {CPU} -r requirements/core.txt"
PROJECT = "pip install --no-deps --no-build-isolation -e ./"


def rules(text: str, path: str = "README.md") -> list[str]:
    if path.endswith(".md") and "```" not in text:
        text = f"```bash\n{text}\n```\n"
    return [f.rule for f in pins.scan_text(path, text)]


# --- install commands -------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "pip install requests",
        "pip3 install requests",
        "pip3.13 install requests",
        "/opt/venv/bin/pip install requests",
        "python -m pip install requests",
        "python3.12 -m pip install requests",
        '"$PY" -m pip install requests',
        'pip install "torch>=2.9,<=2.14"',
    ],
)
def test_named_install_flagged(command):
    assert any("named" in r for r in rules(command))


@pytest.mark.parametrize(
    "command",
    [
        "python -m pip --disable-pip-version-check install requests",
        "pip -q install requests",
        "pip --retries 5 install requests",
        "pip --no-cache-dir --log /tmp/x install requests",
    ],
)
def test_global_options_do_not_hide_install(command):
    assert any("named" in r for r in rules(command))


def test_pinned_but_unhashed_named_install_flagged():
    assert rules("pip install numpy==2.4.6")


@pytest.mark.parametrize("command", ["pipx install mypy", "uv tool install mypy", "easy_install mypy"])
def test_tool_installers_flagged(command):
    assert any("not hash-locked" in r for r in rules(command))


def test_requirements_install_without_require_hashes_flagged():
    assert any("without --require-hashes" in r for r in rules("pip install -r requirements/dev.txt"))
    assert any("without --require-hashes" in r for r in rules("uv pip install -r requirements/core.txt"))
    assert any(
        "without --require-hashes" in r
        for r in rules(f"pip install --extra-index-url {CPU} -r requirements/core.txt")
    )


@pytest.mark.parametrize(
    "command",
    [
        HASHED,
        HASHED_CPU,
        f"pip install --require-hashes --extra-index-url={CPU} -r requirements/dev.txt",
        f"python -m pip install --require-hashes --extra-index-url {CPU} -r ./requirements/dev.txt",
        "pip install --require-hashes --no-deps -r requirements/dev.txt",
        "pip install --require-hashes --requirement=requirements/core.txt",
        "pip install --require-hashes -rrequirements/build.txt",
        "pip install --require-hashes -r requirements/build.txt",
        PROJECT,
        "pip install --no-deps --no-build-isolation ./",
        "pip freeze",
        "pip --version",
        "pip check",
        "uv pip compile --universal --torch-backend cpu pyproject.toml -o requirements/core.txt",
    ],
)
def test_allowed_shapes(command):
    assert rules(command) == []


def test_unknown_requirements_file_flagged():
    assert any("not a maintained lock" in r for r in rules("pip install --require-hashes -r other.txt"))
    assert any("not a maintained lock" in r for r in rules("pip install --require-hashes -r requirements/torch.in"))
    assert any(
        "not a maintained lock" in r
        for r in rules("pip install --require-hashes -c constraints.txt -r requirements/dev.txt")
    )


@pytest.mark.parametrize(
    "command",
    [
        # Today's CI and Dockerfile torch line, and README's.
        f'pip install --index-url {CPU} "torch>=2.9,<=2.14"',
        f"pip install --index-url {CPU} torch",
        # An index swap is flagged even on a hashed lock install.
        f"pip install --require-hashes --index-url {CPU} -r requirements/core.txt",
        f"pip install --require-hashes -i {CPU} -r requirements/core.txt",
        f"pip install --require-hashes -i{CPU} -r requirements/core.txt",
        f"pip --index-url {CPU} install --require-hashes -r requirements/core.txt",
        f"pip install --require-hashes --index-url={CPU} -r requirements/core.txt",
        "pip install --require-hashes --extra-index-url https://example.invalid/simple -r requirements/core.txt",
        "pip install --require-hashes --find-links /wheels -r requirements/core.txt",
        "pip install --require-hashes -f /wheels -r requirements/core.txt",
        f"uv pip install --require-hashes --index {CPU} -r requirements/core.txt",
        "uv pip install --require-hashes --torch-backend cpu -r requirements/core.txt",
    ],
)
def test_index_options_flagged(command):
    assert any("index option" in r for r in rules(command))


@pytest.mark.parametrize(
    "command",
    [f"pip install --extra-index-url {CPU} torch", f"pip install --extra-index-url {CPU} --no-deps --no-build-isolation ./"],
)
def test_cpu_index_needs_a_hashed_lock(command):
    assert any("without a hash-locked requirements file" in r for r in rules(command))


@pytest.mark.parametrize(
    "command",
    [
        'pip install -e ".[dev]"',
        "pip install -e '.[dev]' pytest",
        "pip install .",
        "pip install ./",
        "pip install --no-deps -e ./",
        "pip install --no-build-isolation ./",
        "pip install --no-deps --no-build-isolation .",
        "pip install --no-deps --no-build-isolation /wheels/kurrent_reader-0.6.0-py3-none-any.whl",
        "pip install --no-deps --no-build-isolation ./src",
    ],
)
def test_local_project_needs_the_exact_offline_shape(command):
    assert rules(command)


@pytest.mark.parametrize(
    "command",
    ["pip install git+https://example.invalid/x.git", "pip install --no-deps https://example.invalid/x.whl"],
)
def test_url_install_flagged(command):
    assert any("named" in r for r in rules(command))


@pytest.mark.parametrize("command", ["$PIP install requests", '"${PIP}" install -r requirements/dev.txt'])
def test_installer_behind_variable_flagged(command):
    assert any("variable" in r for r in rules(command))


@pytest.mark.parametrize(
    "command",
    [
        "sudo pip install requests",
        "env PIP_NO_CACHE_DIR=1 pip install requests",
        "echo ok && pip install requests",
        'bash -c "pip install requests"',
        "sh -c 'cd /src; python3 -m pip install requests'",
        f"{HASHED} && pip install requests",
    ],
)
def test_wrapped_and_chained_commands_flagged(command):
    assert rules(command)


def test_markdown_fence_continuation_and_inline_span():
    text = "```bash\npip install \\\n  requests\n```\n\nRun `pip install requests` first.\n"
    found = pins.scan_text("docs/x.md", text)
    assert [f.line for f in found] == [2, 6]


def test_markdown_prose_and_comments_not_commands():
    text = "Never pip install requests by name.\n\n```bash\n# pip install requests\n```\n"
    assert pins.scan_text("docs/x.md", text) == []


def test_dockerfile_run_lines():
    text = (
        "FROM python:3.11-slim\nRUN pip install --upgrade pip\nRUN pip install \\\n    requests\n"
        'RUN ["pip", "install", "mypy"]\nCOPY . /src\n'
    )
    assert [f.line for f in pins.scan_text("Dockerfile", text)] == [2, 3, 5]


def test_workflow_run_steps_and_blocks():
    text = (
        'jobs:\n  t:\n    steps:\n      - run: pip install -e ".[dev]"\n'
        "      - name: torch\n        run: pip install --index-url https://download.pytorch.org/whl/cpu torch\n"
        "      - run: |\n          echo hi\n          pip install mypy\n      - uses: x/y@v1\n"
    )
    assert [f.line for f in pins.scan_text(".github/workflows/ci.yml", text)] == [4, 6, 6, 9]


def test_untracked_kinds_not_scanned():
    assert pins.scan_text("src/notes.txt", "pip install requests") == []


def test_real_dockerfile_uses_every_lock_in_the_hashed_shapes():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert pins.scan_text("Dockerfile", text) == []
    runs = [unit for _, unit in pins.dockerfile_units(text) if "pip" in unit]
    assert runs == [
        "pip install --require-hashes -r requirements/build.txt",
        HASHED_CPU,
        "pip install --no-deps --no-build-isolation ./",
        "pip install --require-hashes --no-deps -r requirements/dev.txt",
    ]


# --- specifiers -------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "spec", "ok"),
    [
        ("0.7.0", "~=0.7", True),
        ("0.8.0", "~=0.7", True),
        ("1.0.0", "~=0.7", False),
        ("0.6.9", "~=0.7", False),
        ("8.2.1", ">=8.1,<8.3", True),
        ("8.3.0", ">=8.1,<8.3", False),
        ("2.14.0+cpu", ">=2.9,<=2.14", True),
        ("2.15.0+cpu", ">=2.9,<=2.14", False),
        ("4.57.6", ">=4.45,<5", True),
        ("5.0.0", ">=4.45,<5", False),
    ],
)
def test_satisfies(version, spec, ok):
    assert pins.satisfies(version, spec) is ok


# --- locks ------------------------------------------------------------------


def test_real_repo_is_clean():
    violations, notes = pins.check(ROOT)
    assert violations == []
    assert len(notes) == len(pins.load_pending(ROOT)[0])


def test_real_locks_carry_cpu_torch_and_no_cuda():
    for rel in ("requirements/core.txt", "requirements/dev.txt"):
        found, errors = pins.parse_lock(rel, (ROOT / rel).read_text(encoding="utf-8"))
        assert errors == []
        by_name = {(p.name, p.marker): p.version for p in found}
        assert by_name[("torch", "sys_platform != 'darwin'")].endswith("+cpu")
        assert by_name[("torchvision", "sys_platform != 'darwin'")].endswith("+cpu")
        assert not [p.name for p in found if p.name.startswith(pins.CUDA_PREFIXES)]


@pytest.fixture
def repo(tmp_path):
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copytree(ROOT / "requirements", tmp_path / "requirements")
    assert pins.check_locks(tmp_path) == []
    return tmp_path


def edit(path: Path, old: str, new: str, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, count), encoding="utf-8")


def rewrite_evidence(root: Path) -> None:
    (root / pins.EVIDENCE).write_text(json.dumps(pins.build_evidence(root)), encoding="utf-8")


def first_hash_line(path: Path) -> str:
    return next(line for line in path.read_text(encoding="utf-8").splitlines() if "--hash=sha256:" in line)


def drop_block(path: Path, head: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith(head))
    j = i + 1
    while j < len(lines) and lines[j].startswith(" "):
        j += 1
    path.write_text("\n".join(lines[:i] + lines[j:]) + "\n", encoding="utf-8")


def test_unhashed_pin_flagged(repo):
    lock = repo / "requirements/core.txt"
    lines = lock.read_text(encoding="utf-8").splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith("pillow=="))
    j = i + 1
    while lines[j].lstrip().startswith("--hash"):
        j += 1
    lines[i:j] = ["pillow==13.0.0"]
    lock.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rewrite_evidence(repo)
    assert any("without a sha256 hash" in e for e in pins.check_locks(repo))


def test_range_pin_flagged(repo):
    edit(repo / "requirements/core.txt", "kraken==", "kraken>=")
    rewrite_evidence(repo)
    assert any("not an exact == pin" in e for e in pins.check_locks(repo))


def test_option_line_flagged(repo):
    lock = repo / "requirements/core.txt"
    lock.write_text(lock.read_text(encoding="utf-8") + f"--extra-index-url {CPU}\n", encoding="utf-8")
    rewrite_evidence(repo)
    assert any("option line" in e for e in pins.check_locks(repo))


def test_malformed_hash_flagged(repo):
    lock = repo / "requirements/build.txt"
    line = first_hash_line(lock)
    edit(lock, line, line.replace("sha256:", "md5:"))
    rewrite_evidence(repo)
    assert any("malformed hash" in e for e in pins.check_locks(repo))


def test_lock_edited_without_evidence_flagged(repo):
    lock = repo / "requirements/dev.txt"
    line = first_hash_line(lock)
    edit(lock, line, line[:-3] + ("0" if line[-3] != "0" else "1") + line[-2:])
    assert any("lock changed without new evidence" in e for e in pins.check_locks(repo))


def test_spec_change_without_relock_flagged(repo):
    edit(repo / "pyproject.toml", '"click>=8.1,<8.3"', '"click>=8.1,<8.2"')
    errors = pins.check_locks(repo)
    assert any("click==8.2.1 is outside '>=8.1,<8.2'" in e for e in errors)
    assert any("inputs (pyproject specs" in e for e in errors)


def test_torch_range_change_without_relock_flagged(repo):
    (repo / "requirements/torch.in").write_text("torch>=2.9,<=2.13\n", encoding="utf-8")
    errors = pins.check_locks(repo)
    assert any("torch==2.14.0+cpu is outside '>=2.9,<=2.13'" in e for e in errors)
    assert any("requirements/core.txt: inputs" in e for e in errors)


def test_torch_in_holds_torch_only(repo):
    (repo / "requirements/torch.in").write_text("torch>=2.9,<=2.14\nnumpy>=1.24\n", encoding="utf-8")
    rewrite_evidence(repo)
    assert "requirements/torch.in: must hold the torch requirement only" in pins.check_locks(repo)


def test_missing_evidence_file_flagged(repo):
    (repo / pins.EVIDENCE).unlink()
    assert any("missing or unreadable" in e for e in pins.check_locks(repo))


def test_header_without_command_flagged(repo):
    edit(repo / "requirements/core.txt", "--torch-backend cpu", "--torch-backend auto")
    rewrite_evidence(repo)
    assert any("core.txt: header does not show" in e for e in pins.check_locks(repo))


def test_top_level_requirement_missing_from_lock_flagged(repo):
    drop_block(repo / "requirements/core.txt", "safetensors==")
    rewrite_evidence(repo)
    assert any("top-level requirement safetensors is not pinned" in e for e in pins.check_locks(repo))


def test_build_in_drift_flagged(repo):
    (repo / "requirements/build.in").write_text("setuptools>=70\n", encoding="utf-8")
    rewrite_evidence(repo)
    assert any("differs from [build-system] requires" in e for e in pins.check_locks(repo))


def test_pin_disagreeing_with_core_flagged(repo):
    edit(repo / "requirements/dev.txt", "pillow==13.0.0", "pillow==12.9.9")
    rewrite_evidence(repo)
    assert any("pillow disagrees with requirements/dev.txt" in e for e in pins.check_locks(repo))


def test_build_lock_disagreeing_with_core_flagged(repo):
    edit(repo / "requirements/build.txt", "setuptools==84.2.1", "setuptools==84.2.0")
    rewrite_evidence(repo)
    assert any("setuptools disagrees with requirements/core.txt" in e for e in pins.check_locks(repo))


def test_non_cpu_torch_flagged(repo):
    edit(repo / "requirements/core.txt", "torch==2.14.0+cpu ;", "torch==2.14.0 ;")
    rewrite_evidence(repo)
    errors = pins.check_locks(repo)
    assert any("torch==2.14.0 is not the CPU build (+cpu)" in e for e in errors)
    assert "requirements/core.txt: no CPU build (+cpu) of torch pinned" in errors


def test_cuda_package_flagged(repo):
    lock = repo / "requirements/core.txt"
    lock.write_text(
        lock.read_text(encoding="utf-8") + "nvidia-cublas-cu12==12.9.1.4 \\\n    --hash=sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    rewrite_evidence(repo)
    assert any("CUDA package nvidia-cublas-cu12" in e for e in pins.check_locks(repo))


def test_missing_lock_flagged(repo):
    (repo / "requirements/core.txt").unlink()
    assert any("maintained lock is missing" in e for e in pins.check_locks(repo))


def test_evidence_written_by_tool_matches_committed():
    committed = json.loads((ROOT / pins.EVIDENCE).read_text(encoding="utf-8"))
    assert committed == pins.build_evidence(ROOT)


def test_pyproject_specs_are_the_unlocked_ones():
    # The locks pin; the specs stay as they were (#63).
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["dependencies"] == [
        "numpy>=1.24",
        "pillow>=10.0",
        "opencv-python-headless>=4.8",
        "scikit-image>=0.22",
        "transformers>=4.45,<5",
        "kraken>=5.2",
        "safetensors~=0.7",
        "click>=8.1,<8.3",
    ]
    assert project["optional-dependencies"]["dev"] == ["pytest>=8", "mypy>=1.10"]
    assert (ROOT / "requirements/torch.in").read_text(encoding="utf-8") == "torch>=2.9,<=2.14\n"


# --- pending workflow lines -------------------------------------------------


def test_pending_allowlist_names_exactly_the_current_workflow_lines():
    entries, errors = pins.load_pending(ROOT)
    assert errors == []
    current = set()
    for wf in sorted((ROOT / ".github/workflows").glob("*.yml")):
        rel = wf.relative_to(ROOT).as_posix()
        current |= {(f.path, f.line, f.text) for f in pins.scan_text(rel, wf.read_text(encoding="utf-8"))}
    # Today: tests.yml 32/33, tests-next.yml 31/32 and release.yml 19. The
    # workflow change empties `current` and deletes the file in the same
    # commit, so this stays exact both ways.
    assert {(e["path"], e["line"], e["command"]) for e in entries} == current
    if (ROOT / pins.PENDING).is_file():
        assert entries, f"nothing is pending; delete {pins.PENDING}"


def workflow_repo(tmp_path, entries):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / ".github/workflows/ci.yml").write_text(
        "steps:\n  - run: pip install mypy\n", encoding="utf-8"
    )
    (tmp_path / "requirements").mkdir()
    (tmp_path / pins.PENDING).write_text(json.dumps({"entries": entries}), encoding="utf-8")
    violations, notes = pins.check(tmp_path)
    return [v for v in violations if not v.startswith("requirements/") or pins.PENDING in v], notes


def test_listed_workflow_line_is_pending_not_violation(tmp_path):
    entry = {"path": ".github/workflows/ci.yml", "line": 2, "command": "pip install mypy"}
    violations, notes = workflow_repo(tmp_path, [entry])
    assert violations == []
    assert notes == [".github/workflows/ci.yml:2: pip install mypy"]


def test_unlisted_workflow_line_fails(tmp_path):
    violations, _ = workflow_repo(tmp_path, [])
    assert any(v.startswith(".github/workflows/ci.yml:2:") for v in violations)


def test_stale_pending_entry_fails(tmp_path):
    stale = {"path": ".github/workflows/ci.yml", "line": 9, "command": "pip install mypy"}
    violations, _ = workflow_repo(tmp_path, [stale])
    assert any("matches no violating line" in v for v in violations)


def test_pending_accepts_only_workflow_paths(tmp_path):
    (tmp_path / "README.md").write_text("```\npip install mypy\n```\n", encoding="utf-8")
    entry = {"path": "README.md", "line": 2, "command": "pip install mypy"}
    violations, _ = workflow_repo(tmp_path, [entry])
    assert any("only .github/workflows/ lines may be pending" in v for v in violations)
    assert any(v.startswith("README.md:2:") for v in violations)


def test_main_exit_codes(capsys, tmp_path):
    assert pins.main(["--root", str(ROOT)]) == 0
    assert "install pins OK (" in capsys.readouterr().out
    workflow_repo(tmp_path, [])
    assert pins.main(["--root", str(tmp_path)]) == 1
