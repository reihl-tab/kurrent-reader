"""Two-way docs-vs-code drift check (issue #34).

- Every subcommand and flag in cli.py must appear in the docs, or carry an
  allowlist entry with a reason.
- Every CLI token the docs present as kurrent-reader syntax must exist in
  the code.
- The paths in the README layout section must exist on disk.
- The three version carriers (VERSION, pyproject.toml, __init__.__version__)
  must agree.

Stdlib only, offline; sources are parsed as text so no dependency is needed.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "src" / "kurrent_reader" / "cli.py"
DOC_FILES = sorted(ROOT.glob("*.md"))
ALLOWLIST = ROOT / ".github" / "docs-vs-code-allowlist.txt"

SUB_RE = re.compile(r'add_parser\("([a-z][a-z0-9-]*)"')
FLAG_RE = re.compile(r'add_argument\(\s*"(--[a-z][a-z0-9-]*)"')
DOC_FLAG_RE = re.compile(r"--[a-z][a-z0-9-]*")
DOC_CMD_RE = re.compile(r"kurrent-reader\s+([a-z][a-z0-9-]*)")


def _allowlist() -> dict[str, str]:
    out: dict[str, str] = {}
    if ALLOWLIST.is_file():
        for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token, _, reason = line.partition("#")
            token = token.strip()
            assert reason.strip(), f"allowlist entry without a reason: {line}"
            out[token] = reason.strip()
    return out


def cli_tokens() -> tuple[set[str], set[str]]:
    text = CLI.read_text(encoding="utf-8")
    return set(SUB_RE.findall(text)), set(FLAG_RE.findall(text)) - {"--help"}


def doc_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in DOC_FILES)


def test_every_cli_token_is_documented() -> None:
    text = re.sub(r"\s+", " ", doc_text()).casefold()
    allowed = _allowlist()
    subs, flags = cli_tokens()
    missing = [s for s in sorted(subs) if f"kurrent-reader {s}" not in text and s not in allowed]
    missing += [f for f in sorted(flags) if f.casefold() not in text and f not in allowed]
    assert not missing, f"CLI tokens missing from docs (document or allowlist): {missing}"


SPAN_RE = re.compile(r"```[^\n]*\n(.*?)```|`([^`\n]+)`", re.S)


def _cli_lines() -> list[str]:
    """Lines inside doc code spans that invoke kurrent-reader."""
    out = []
    for m in SPAN_RE.finditer(doc_text()):
        span = m.group(1) if m.group(1) is not None else m.group(2)
        for line in span.splitlines():
            if "kurrent-reader" in line:
                out.append(line)
    return out


def test_every_documented_cli_token_exists() -> None:
    allowed = _allowlist()
    subs, flags = cli_tokens()
    doc_cmds: set[str] = set()
    doc_flags: set[str] = set()
    for line in _cli_lines():
        for m in DOC_CMD_RE.finditer(line):
            doc_cmds.add(m.group(1))
            doc_flags.update(DOC_FLAG_RE.findall(line[m.end():]))
    doc_cmds -= {"--help", "command"}
    doc_flags.discard("--help")
    bogus = [c for c in sorted(doc_cmds) if c not in subs and c not in allowed]
    bogus += [f for f in sorted(doc_flags) if f not in flags and f not in allowed]
    assert not bogus, f"docs present CLI tokens the code does not have: {bogus}"


def test_readme_layout_paths_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"## Layout\s+```\n(.*?)```", readme, re.S)
    assert m, "README layout section not found"
    missing = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        token = line.split()[0]
        if token == "src/kurrent_reader/":
            path = ROOT / "src" / "kurrent_reader"
        elif token.endswith(".py"):
            path = ROOT / "src" / "kurrent_reader" / token
        else:
            path = ROOT / token
        if not path.exists():
            missing.append(token)
    assert not missing, f"README layout paths missing on disk: {missing}"


def test_version_carriers_agree() -> None:
    version_file = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'(?m)^version = "([^"]+)"', pyproject)
    assert m, "pyproject version not found"
    init = (ROOT / "src" / "kurrent_reader" / "__init__.py").read_text(encoding="utf-8")
    mi = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    assert mi, "__init__.__version__ not found"
    assert version_file == m.group(1) == mi.group(1), (
        f"version drift: VERSION={version_file} pyproject={m.group(1)} __init__={mi.group(1)}"
    )
