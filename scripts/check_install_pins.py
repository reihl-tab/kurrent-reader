#!/usr/bin/env python3
"""Offline guard for hash-locked installs (#63).

Standard library only, no network. Two checks:

1. Install commands. Tracked Markdown (fenced code and inline code spans),
   Dockerfile RUN lines and workflow `run:` steps under .github/workflows
   are scanned for pip, `python -m pip` (global options included), pip3.X,
   `uv pip`, `uv tool`, pipx and easy_install, wherever they appear in a
   command (after sudo, env, xargs, inside `bash -c "..."`). An installer
   reached through a variable fails, so a renamed command cannot step around
   the scan. Allowed shapes:
     - a hashed lock: `-r requirements/<lock>.txt` with --require-hashes;
     - the PyTorch CPU index, only as
       `--extra-index-url https://download.pytorch.org/whl/cpu` on such an
       install (the core and dev locks pin torch and torchvision from it);
     - the local project: `./` / `-e ./` with --no-deps --no-build-isolation.
   Named, unpinned, pinned-but-unhashed, URL and unknown-file installs fail,
   and so does any other index option (--index-url, -i, --find-links, ...).
2. Lock evidence. Each lock under requirements/ holds only exact `==` pins
   with sha256 hashes, shows its generating command in its header, pins
   every top-level requirement within its pyproject.toml (or .in) spec,
   agrees with every other lock on shared pins, carries the CPU build of
   torch and torchvision (`+cpu`) and no CUDA packages, and matches
   requirements/lock-evidence.json (digests of the lock and of its inputs,
   so a re-lock or a spec change without new evidence fails).

Workflow lines the maintainer has not converted yet are listed in
requirements/pending-workflow-installs.json by path, line and command; an
entry that no longer matches a violating line fails, and only
.github/workflows/ paths are accepted. See docs/DEPENDENCY-LOCKS.md.

Usage:
  python scripts/check_install_pins.py [--root DIR]
  python scripts/check_install_pins.py --write-evidence   # after re-locking
Exit 0 when everything conforms, 1 with one line per violation otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

UV = "uv pip compile --universal --generate-hashes --python-version 3.11"
TORCH = "--torch-backend cpu"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


@dataclass(frozen=True)
class LockSpec:
    """One maintained lock: its generating command and what it must cover."""

    command: str
    pyproject: tuple[str, ...]
    inputs: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()


LOCKS: dict[str, LockSpec] = {
    "requirements/build.txt": LockSpec(
        command=f"{UV} requirements/build.in -o requirements/build.txt",
        pyproject=("build-system.requires",),
        inputs=("requirements/build.in",),
    ),
    "requirements/core.txt": LockSpec(
        command=f"{UV} {TORCH} pyproject.toml requirements/torch.in -o requirements/core.txt",
        pyproject=("project.requires-python", "project.dependencies"),
        inputs=("requirements/torch.in",),
    ),
    "requirements/dev.txt": LockSpec(
        command=(
            f"{UV} {TORCH} pyproject.toml requirements/torch.in --extra dev"
            " -c requirements/core.txt -o requirements/dev.txt"
        ),
        pyproject=(
            "project.requires-python",
            "project.dependencies",
            "project.optional-dependencies.dev",
        ),
        inputs=("requirements/torch.in",),
        constraints=("requirements/core.txt",),
    ),
}
EVIDENCE = "requirements/lock-evidence.json"
PENDING = "requirements/pending-workflow-installs.json"
PENDING_PREFIX = ".github/workflows/"
# Locked from the PyTorch CPU index; off macOS the pin must be the CPU build.
CPU_BUILDS = ("torch", "torchvision")
CUDA_PREFIXES = ("nvidia-", "triton", "pytorch-triton")

# ---------------------------------------------------------------------------
# Lock files


@dataclass(frozen=True)
class Pin:
    name: str
    version: str
    marker: str
    hashes: tuple[str, ...]
    line: int


_PIN = re.compile(r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)==([A-Za-z0-9.+!_-]+)(?:\s*;\s*(.+?))?$")
_HASH = re.compile(r"^--hash=sha256:[0-9a-f]{64}$")
_REQ = re.compile(r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*(?:\[[^\]]*\])?\s*([^;]*)")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def logical_lines(text: str) -> list[tuple[int, str]]:
    """Join backslash continuations; drop comments and blank lines."""
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    start = 0
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split(" #", 1)[0] if not raw.lstrip().startswith("#") else ""
        if not buf:
            start = number
        if line.rstrip().endswith("\\"):
            buf.append(line.rstrip()[:-1])
            continue
        buf.append(line)
        joined = " ".join(part.strip() for part in buf).strip()
        buf = []
        if joined:
            out.append((start, joined))
    return out


def parse_lock(rel: str, text: str) -> tuple[list[Pin], list[str]]:
    pins: list[Pin] = []
    errors: list[str] = []
    for number, line in logical_lines(text):
        head, *opts = line.split()
        if head.startswith("-"):
            errors.append(f"{rel}:{number}: option line {head!r} is not allowed in a lock")
            continue
        match = _PIN.match(" ".join([head] + [o for o in opts if not o.startswith("--hash")]))
        if not match:
            errors.append(f"{rel}:{number}: not an exact == pin")
            continue
        hashes = tuple(o for o in opts if o.startswith("--hash"))
        if any(not _HASH.match(h) for h in hashes):
            errors.append(f"{rel}:{number}: malformed hash (sha256 only)")
        if not hashes:
            errors.append(f"{rel}:{number}: {match.group(1)} pinned without a sha256 hash")
        pins.append(Pin(normalize(match.group(1)), match.group(2), match.group(3) or "", hashes, number))
    if not pins:
        errors.append(f"{rel}: lock has no pins")
    return pins, errors


def _version_key(version: str) -> tuple[int, ...]:
    # The local label (+cpu) is ignored, as PEP 440 does for a public specifier.
    parts = re.findall(r"\d+", version.split("+")[0])
    if not parts:
        raise ValueError(version)
    return tuple(int(p) for p in parts[:4]) + (0,) * (4 - min(len(parts), 4))


def satisfies(version: str, specifier: str) -> bool:
    """Minimal comparison for the simple specifiers pyproject.toml uses."""
    for clause in filter(None, (c.strip() for c in specifier.split(","))):
        match = re.fullmatch(r"(~=|>=|<=|==|!=|>|<)\s*([0-9][0-9A-Za-z.+!-]*)", clause)
        if not match:
            raise ValueError(clause)
        op, raw = match.group(1), match.group(2)
        want, have = _version_key(raw), _version_key(version)
        if op == "~=":
            released = re.findall(r"\d+", raw)
            if len(released) < 2:
                raise ValueError(clause)
            prefix = tuple(int(p) for p in released[:-1])
            ok = have >= want and have[: len(prefix)] == prefix
        else:
            ok = {
                ">=": have >= want,
                "<=": have <= want,
                "==": have == want,
                "!=": have != want,
                ">": have > want,
                "<": have < want,
            }[op]
        if not ok:
            return False
    return True


def section(pyproject: dict, dotted: str) -> object:
    node: object = pyproject
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(dotted)
        node = node[key]
    return node


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inputs_digest(root: Path, spec: LockSpec, pyproject: dict) -> str:
    """Digest of what a lock is generated from: its command, input files,
    constraint locks and the pyproject.toml sections it covers."""
    parts: dict[str, object] = {"command": spec.command}
    for rel in spec.inputs + spec.constraints:
        path = root / rel
        parts[rel] = sha256_text(path.read_text(encoding="utf-8")) if path.is_file() else None
    for dotted in spec.pyproject:
        try:
            parts["pyproject.toml#" + dotted] = section(pyproject, dotted)
        except KeyError:
            parts["pyproject.toml#" + dotted] = None
    return sha256_text(json.dumps(parts, sort_keys=True, ensure_ascii=True))


def build_evidence(root: Path) -> dict:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    locks = {
        rel: {
            "command": spec.command,
            "inputs_sha256": inputs_digest(root, spec, pyproject),
            "lock_sha256": sha256_text((root / rel).read_text(encoding="utf-8")),
        }
        for rel, spec in sorted(LOCKS.items())
    }
    return {
        "version": 1,
        "comment": "Written by `python scripts/check_install_pins.py --write-evidence` after"
        " re-locking (docs/DEPENDENCY-LOCKS.md). The guard fails when a lock or one of its"
        " inputs changes without this file.",
        "locks": locks,
    }


def _top_level(root: Path, spec: LockSpec, pyproject: dict) -> tuple[list[str], list[str]]:
    reqs: list[str] = []
    errors: list[str] = []
    for rel in spec.inputs:
        path = root / rel
        if not path.is_file():
            errors.append(f"{rel}: lock input is missing")
            continue
        reqs += [line for _, line in logical_lines(path.read_text(encoding="utf-8"))]
    for dotted in spec.pyproject:
        try:
            value = section(pyproject, dotted)
        except KeyError:
            errors.append(f"pyproject.toml: {dotted} is missing")
            continue
        if isinstance(value, list):
            reqs += [str(v) for v in value]
    return reqs, errors


def _spec_errors(rel: str, pins: list[Pin], req: str) -> list[str]:
    match = _REQ.match(req.strip())
    if not match:
        return [f"{rel}: cannot read top-level requirement {req!r}"]
    name, specifier = normalize(match.group(1)), match.group(2).strip()
    found = [p for p in pins if p.name == name]
    if not found:
        return [f"{rel}: top-level requirement {name} is not pinned"]
    errors = []
    for pin in found:
        try:
            ok = satisfies(pin.version, specifier)
        except ValueError:
            ok = False
        if not ok:
            errors.append(f"{rel}:{pin.line}: {name}=={pin.version} is outside {specifier!r}")
    return errors


def _cpu_errors(rel: str, pins: list[Pin]) -> list[str]:
    errors = []
    for pin in pins:
        if pin.name.startswith(CUDA_PREFIXES):
            errors.append(f"{rel}:{pin.line}: CUDA package {pin.name} in a CPU-only lock")
        if pin.name in CPU_BUILDS and pin.marker != "sys_platform == 'darwin'" and not pin.version.endswith("+cpu"):
            errors.append(f"{rel}:{pin.line}: {pin.name}=={pin.version} is not the CPU build (+cpu)")
    for name in CPU_BUILDS:
        if not any(p.name == name and p.version.endswith("+cpu") for p in pins):
            errors.append(f"{rel}: no CPU build (+cpu) of {name} pinned")
    return errors


def check_locks(root: Path) -> list[str]:
    errors: list[str] = []
    try:
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return ["pyproject.toml: missing or unreadable"]
    parsed: dict[str, list[Pin]] = {}
    for rel, spec in sorted(LOCKS.items()):
        path = root / rel
        if not path.is_file():
            errors.append(f"{rel}: maintained lock is missing")
            continue
        text = path.read_text(encoding="utf-8")
        pins, lock_errors = parse_lock(rel, text)
        parsed[rel] = pins
        errors += lock_errors
        if f"#    {spec.command}" not in "\n".join(text.splitlines()[:3]):
            errors.append(f"{rel}: header does not show the generating command")
        reqs, req_errors = _top_level(root, spec, pyproject)
        errors += req_errors
        for req in reqs:
            errors += _spec_errors(rel, pins, req)
        if TORCH in spec.command:
            errors += _cpu_errors(rel, pins)
    build_in = root / "requirements/build.in"
    if build_in.is_file():
        declared = sorted(pyproject.get("build-system", {}).get("requires", []))
        listed = sorted(line for _, line in logical_lines(build_in.read_text(encoding="utf-8")))
        if declared != listed:
            errors.append("requirements/build.in: differs from [build-system] requires")
    torch_in = root / "requirements/torch.in"
    if torch_in.is_file():
        names = {normalize(m.group(1)) for _, line in logical_lines(torch_in.read_text(encoding="utf-8"))
                 if (m := _REQ.match(line))}
        if names != {"torch"}:
            errors.append("requirements/torch.in: must hold the torch requirement only")
    # Shared packages must agree across locks (same marker, same version and
    # hashes), so installing one lock after another never swaps a package.
    names = sorted(parsed)
    for i, rel in enumerate(names):
        for other_rel in names[i + 1 :]:
            base = {(p.name, p.marker): p for p in parsed[other_rel]}
            for pin in parsed[rel]:
                other = base.get((pin.name, pin.marker))
                if other and (other.version != pin.version or other.hashes != pin.hashes):
                    errors.append(f"{rel}:{pin.line}: {pin.name} disagrees with {other_rel}")
    errors += _evidence_errors(root, pyproject)
    return errors


def _evidence_errors(root: Path, pyproject: dict) -> list[str]:
    path = root / EVIDENCE
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))["locks"]
        assert isinstance(recorded, dict)
    except (OSError, ValueError, KeyError, AssertionError):
        return [f"{EVIDENCE}: missing or unreadable; run --write-evidence after re-locking"]
    errors = []
    if sorted(recorded) != sorted(LOCKS):
        errors.append(f"{EVIDENCE}: lists {sorted(recorded)}, expected {sorted(LOCKS)}")
    for rel, spec in sorted(LOCKS.items()):
        entry = recorded.get(rel)
        if not isinstance(entry, dict) or not (root / rel).is_file():
            continue
        if entry.get("command") != spec.command:
            errors.append(f"{EVIDENCE}: {rel}: recorded command differs")
        if entry.get("lock_sha256") != sha256_text((root / rel).read_text(encoding="utf-8")):
            errors.append(f"{EVIDENCE}: {rel}: lock changed without new evidence")
        if entry.get("inputs_sha256") != inputs_digest(root, spec, pyproject):
            errors.append(f"{EVIDENCE}: {rel}: inputs (pyproject specs, .in, constraints) changed; re-lock")
    return errors


# ---------------------------------------------------------------------------
# Install commands


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    text: str
    rule: str


SEPARATORS = {";", "&&", "||", "|", "&", "(", ")"}
INDEX_OPTIONS = {
    "-i",
    "--index-url",
    "--extra-index-url",
    "-f",
    "--find-links",
    "--index",
    "--default-index",
    "--torch-backend",
}
# Options taking a separate value, before or after `install`.
VALUED = INDEX_OPTIONS | {
    "-r",
    "--requirement",
    "-c",
    "--constraint",
    "-e",
    "--editable",
    "--retries",
    "--timeout",
    "--proxy",
    "--cert",
    "--client-cert",
    "--cache-dir",
    "--log",
    "--python",
    "--exists-action",
    "--trusted-host",
    "--keyring-provider",
    "-t",
    "--target",
    "--prefix",
    "--root",
    "--src",
    "--platform",
    "--python-version",
    "--implementation",
    "--abi",
    "--progress-bar",
    "--report",
    "--config-settings",
    "-C",
    "--global-option",
    "--upgrade-strategy",
    "--only-binary",
    "--no-binary",
    "--use-feature",
    "--use-deprecated",
    "--index-strategy",
}
PIP_NAME = re.compile(r"pip(?:3(?:\.\d+)?)?")


def simple_commands(command: str, depth: int = 0) -> list[list[str]]:
    """Split a shell line into simple commands; recurse into quoted scripts."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
    except ValueError:
        return [command.split()]
    out: list[list[str]] = [[]]
    for token in tokens:
        if token in SEPARATORS or (token and set(token) <= set(";&|()")):
            out.append([])
            continue
        out[-1].append(token)
        if depth < 3 and any(c.isspace() for c in token) and re.search(r"pip|install", token):
            out += simple_commands(token, depth + 1)
    return [c for c in out if c]


def installers(cmd: list[str]) -> list[tuple[str, int | None]]:
    """(kind, index where `install` is expected; None = always an install)."""
    found: list[tuple[str, int | None]] = []
    for i, token in enumerate(cmd):
        name = Path(token).name
        nxt = cmd[i + 1] if i + 1 < len(cmd) else ""
        if name == "easy_install":
            found.append(("easy_install", None))
        elif name == "pipx":
            found.append(("pipx", i + 1))
        elif name == "uv" and nxt in {"tool", "pip"}:
            found.append((f"uv {nxt}", i + 2))
        elif PIP_NAME.fullmatch(name) or token == "-mpip":
            # Also `python -m pip`, `"$PY" -m pip`, /venv/bin/pip3.13 ...
            found.append(("pip", i + 1))
        elif token.startswith("$") and nxt == "install":
            found.append(("variable", i + 1))
    return found


def _after_install(cmd: list[str], start: int | None) -> tuple[list[str], list[str]] | None:
    """(global options, arguments after `install`); None if no install."""
    if start is None:
        return [], []
    i = start
    while i < len(cmd) and cmd[i].startswith("-"):
        i += 2 if cmd[i] in VALUED else 1
    if i < len(cmd) and cmd[i] in {"install", "sync"}:
        return cmd[start:i], cmd[i + 1 :]
    return None


def _local_target(target: str) -> bool:
    return target.strip("'\"") == "./"


def _index_rules(options: list[tuple[str, str]], reqs: list[str]) -> list[str]:
    rules = []
    for opt, value in options:
        if opt in {"--extra-index-url"} and value == CPU_INDEX:
            if not reqs:
                rules.append(f"--extra-index-url {CPU_INDEX} without a hash-locked requirements file")
            continue
        rules.append(f"index option {opt} {value!r}; only --extra-index-url {CPU_INDEX} with a hashed lock")
    return rules


def install_rules(kind: str, args: list[str]) -> list[str]:
    if kind in {"pipx", "easy_install", "uv tool"}:
        return [f"{kind} install is not hash-locked; use a lock under requirements/"]
    if kind == "variable":
        return ["installer reached through a variable; spell the installer out"]
    flags: set[str] = set()
    reqs: list[str] = []
    targets: list[str] = []
    index: list[tuple[str, str]] = []
    i = 0
    while i < len(args):
        arg = args[i]
        value = args[i + 1] if i + 1 < len(args) else ""
        key = arg.split("=", 1)[0]
        if arg in {"-r", "--requirement", "-c", "--constraint"}:
            reqs.append(value)
            i += 2
            continue
        if key in INDEX_OPTIONS:
            if "=" in arg and arg.startswith("--"):
                index.append((key, arg.split("=", 1)[1]))
                i += 1
            elif arg in INDEX_OPTIONS:
                index.append((arg, value))
                i += 2
            else:  # -ihttps://... / -f./dir
                index.append((arg[:2], arg[2:]))
                i += 1
            continue
        if re.match(r"^(-[icf])(?!-).+", arg):
            index.append((arg[:2], arg[2:]))
        elif re.match(r"^(-r|-c)(?!-).+", arg):
            reqs.append(arg[2:])
        elif arg.startswith(("--requirement=", "--constraint=")):
            reqs.append(arg.split("=", 1)[1])
        elif arg in {"-e", "--editable"}:
            targets.append(value)
            i += 2
            continue
        elif arg.startswith("--editable="):
            targets.append(arg.split("=", 1)[1])
        elif arg.startswith("-"):
            flags.add(key)
            i += 2 if arg in VALUED else 1
            continue
        else:
            targets.append(arg)
        i += 1
    rules = []
    hashed_refs = []
    for ref in reqs:
        lock = re.sub(r"^\./", "", ref)
        if lock not in LOCKS:
            rules.append(f"requirements file {ref!r} is not a maintained lock")
        elif "--require-hashes" not in flags:
            rules.append(f"{ref} installed without --require-hashes")
        else:
            hashed_refs.append(ref)
    rules += _index_rules(index, hashed_refs)
    for target in targets:
        if not _local_target(target):
            rules.append(f"named, unpinned or URL install {target!r}; install from a lock")
        elif not {"--no-deps", "--no-build-isolation"} <= flags:
            rules.append(f"local project {target!r} installed without --no-deps --no-build-isolation")
    if not reqs and not targets:
        rules.append("install with no lock and no target")
    return rules


def scan_command(path: str, line: int, text: str) -> list[Finding]:
    findings = []
    for cmd in simple_commands(text):
        for kind, start in installers(cmd):
            parts = _after_install(cmd, start)
            if parts is None:
                continue
            options, args = parts
            # Global options before `install` count too (pip -i URL install ...).
            for rule in install_rules(kind, options + args):
                findings.append(Finding(path, line, text.strip(), rule))
    return findings


def markdown_units(text: str) -> list[tuple[int, str]]:
    """Fenced code lines (continuations joined) and inline code spans."""
    units: list[tuple[int, str]] = []
    fence: str | None = None
    buf: list[str] = []
    start = 0
    for number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        marker = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence is None and marker:
            fence = marker.group(1)[0] * 3
            continue
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
                continue
            if not buf:
                start = number
            if stripped.endswith("\\"):
                buf.append(stripped[:-1])
                continue
            buf.append(stripped)
            units.append((start, " ".join(buf)))
            buf = []
            continue
        units += [(number, span) for span in re.findall(r"`([^`\n]+)`", raw)]
    return units


def dockerfile_units(text: str) -> list[tuple[int, str]]:
    units = []
    for number, line in logical_lines(text):
        match = re.match(r"^\s*RUN\s+(.*)$", line, re.IGNORECASE)
        if not match:
            continue
        body = match.group(1).strip()
        if body.startswith("["):
            try:
                body = " ".join(shlex.quote(str(t)) for t in json.loads(body))
            except ValueError:
                pass
        units.append((number, body))
    return units


def workflow_units(text: str) -> list[tuple[int, str]]:
    """Each line of every `run:` step (block scalars included)."""
    units = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = re.match(r"^(\s*)(?:-\s+)?run:\s*(.*)$", lines[i])
        if not match:
            i += 1
            continue
        indent, value = len(match.group(1)), match.group(2).strip()
        if value and value[0] not in "|>":
            units.append((i + 1, value.strip("'\"") if value[0] in "'\"" else value))
            i += 1
            continue
        i += 1
        while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent):
            if lines[i].strip():
                units.append((i + 1, lines[i].strip()))
            i += 1
    return units


def source_kind(rel: str) -> str | None:
    name = rel.rsplit("/", 1)[-1]
    if rel.startswith(PENDING_PREFIX) and name.endswith((".yml", ".yaml")):
        return "workflow"
    if name.startswith("Dockerfile") or name.endswith(".dockerfile"):
        return "dockerfile"
    if name.endswith(".md"):
        return "markdown"
    return None


def scan_text(rel: str, text: str) -> list[Finding]:
    kind = source_kind(rel)
    units = {
        "workflow": workflow_units,
        "dockerfile": dockerfile_units,
        "markdown": markdown_units,
    }.get(kind or "", lambda _t: [])(text)
    findings = []
    for line, unit in units:
        findings += scan_command(rel, line, unit)
    return findings


def tracked_files(root: Path) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True
        ).stdout
        return sorted(n for n in out.decode("utf-8").split("\0") if n)
    except (subprocess.CalledProcessError, FileNotFoundError):
        skip = {".git", ".venv", "venv", "build", "dist", "__pycache__"}
        return sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and not skip & set(p.relative_to(root).parts)
        )


def load_pending(root: Path) -> tuple[list[dict], list[str]]:
    path = root / PENDING
    if not path.is_file():
        return [], []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
        assert isinstance(entries, list)
    except (ValueError, KeyError, AssertionError, TypeError):
        return [], [f"{PENDING}: unreadable"]
    good, errors = [], []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "line", "command"}:
            errors.append(f"{PENDING}: each entry needs exactly path, line, command")
        elif not str(entry["path"]).startswith(PENDING_PREFIX):
            errors.append(f"{PENDING}: only {PENDING_PREFIX} lines may be pending")
        else:
            good.append(entry)
    return good, errors


def check(root: Path) -> tuple[list[str], list[str]]:
    """Return (violations, pending notes) for the repository at root."""
    findings: list[Finding] = []
    for rel in tracked_files(root):
        if source_kind(rel) and (root / rel).is_file():
            findings += scan_text(rel, (root / rel).read_text(encoding="utf-8"))
    entries, errors = load_pending(root)
    pending_keys = {(e["path"], e["line"], e["command"]) for e in entries}
    finding_keys = {(f.path, f.line, f.text) for f in findings}
    for key in sorted(pending_keys - finding_keys, key=str):
        errors.append(f"{PENDING}: {key[0]}:{key[1]}: {key[2]!r} matches no violating line; delete it")
    violations = sorted(
        {f"{f.path}:{f.line}: {f.rule}" for f in findings if (f.path, f.line, f.text) not in pending_keys}
    )
    notes = sorted({f"{p}:{n}: {t}" for p, n, t in finding_keys & pending_keys})
    return violations + sorted(errors) + sorted(check_locks(root)), notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--root", default=".", help="repository root (default: cwd)")
    parser.add_argument("--write-evidence", action="store_true", help=f"rewrite {EVIDENCE} after re-locking")
    args = parser.parse_args(argv)
    root = Path(args.root).absolute()
    if args.write_evidence:
        text = json.dumps(build_evidence(root), indent=2, sort_keys=True) + "\n"
        (root / EVIDENCE).write_text(text, encoding="utf-8")
        print(f"wrote {EVIDENCE}")
        return 0
    violations, notes = check(root)
    for note in notes:
        print(f"pending workflow install (maintainer): {note}")
    for violation in violations:
        print(f"install violation: {violation}")
    if violations:
        print(f"{len(violations)} install violation(s); see docs/DEPENDENCY-LOCKS.md")
        return 1
    print(f"install pins OK ({len(notes)} pending workflow line(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
