#!/usr/bin/env python3
"""Automated semver release: bump, changelog, tag, GitHub release.

Bump is derived from conventional-commit subjects since the last tag:
  fix:/perf:/revert: -> patch, feat: -> minor, BREAKING CHANGE or `!:` -> major.
docs:/chore:/test:/ci:/style:/refactor:-only ranges produce no release.

Pure functions (parse_commits, bump, next_version, render_changelog_entry,
render_release_notes) are unit-tested; the CLI wraps git/gh calls.
"""
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

VERSION_FILE = Path("VERSION")
CHANGELOG_FILE = Path("CHANGELOG.md")

PATCH_TYPES = ("fix", "perf", "revert")
MINOR_TYPES = ("feat",)
SKIP_TYPES = ("docs", "chore", "test", "ci", "style", "refactor", "build")

HE_HEADERS = {"major": "שינויים שוברי תאימות", "minor": "תכונות חדשות", "patch": "תיקוני באגים"}
EN_HEADERS = {"major": "Breaking changes", "minor": "Features", "patch": "Bug fixes"}

SUBJECT_RE = re.compile(r"^(?P<type>[a-z]+)(?P<scope>\([^)]*\))?(?P<bang>!)?:\s*(?P<desc>.+)$")


def parse_commits(subjects: list[str]) -> list[dict]:
    """Parse conventional-commit subjects into typed entries."""
    out = []
    for s in subjects:
        m = SUBJECT_RE.match(s.strip())
        if not m:
            out.append({"type": "other", "breaking": False, "desc": s.strip()})
            continue
        out.append({
            "type": m.group("type"),
            "breaking": bool(m.group("bang")),
            "desc": m.group("desc").strip(),
        })
    return out


def bump(commits: list[dict]) -> str | None:
    """Return 'major'|'minor'|'patch', or None when nothing release-worthy."""
    if any(c["breaking"] for c in commits):
        return "major"
    types = {c["type"] for c in commits}
    if types & set(MINOR_TYPES):
        return "minor"
    if types & set(PATCH_TYPES):
        return "patch"
    return None


def next_version(current: str, level: str) -> str:
    major, minor, patch = (int(x) for x in current.strip().split("."))
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    if level == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unknown bump level: {level}")


def render_changelog_entry(version: str, level: str, commits: list[dict],
                           today: date) -> str:
    lines = [f"## [{version}] - {today.isoformat()}", ""]
    worthy = [c for c in commits
              if c["type"] in PATCH_TYPES + MINOR_TYPES or c["breaking"]]
    header = EN_HEADERS[level]
    lines.append(f"### {header}")
    for c in worthy:
        mark = " (**breaking**)" if c["breaking"] else ""
        lines.append(f"- {c['desc']}{mark}")
    return "\n".join(lines) + "\n"


def render_release_notes(version: str, level: str, commits: list[dict],
                         today: date) -> str:
    """Bilingual (English + Hebrew) release notes from commit subjects."""
    worthy = [c for c in commits
              if c["type"] in PATCH_TYPES + MINOR_TYPES or c["breaking"]]
    items = "\n".join(
        f"- {c['desc']}" + (" (**breaking**)" if c["breaking"] else "")
        for c in worthy
    ) or "- Maintenance release."
    return (
        f"## v{version} ({today.isoformat()})\n\n"
        f"### {EN_HEADERS[level]}\n{items}\n\n"
        f"---\n\n"
        f"### {HE_HEADERS[level]}\n{items}\n\n"
        f"_גרסה {version} - נוצרה אוטומטית מתוך ה-commit-ים שמוזגו._\n"
    )


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


def main(dry_run: bool = False) -> int:
    current = VERSION_FILE.read_text().strip()
    try:
        last_tag = sh("git", "describe", "--tags", "--abbrev=0")
    except subprocess.CalledProcessError:
        last_tag = ""
    rng = f"{last_tag}..HEAD" if last_tag else "HEAD"
    subjects = sh("git", "log", rng, "--pretty=%s").splitlines()
    head_subject = subjects[0] if subjects else ""
    if head_subject.startswith("chore(release):"):
        print("release commit at HEAD; nothing to do")
        return 0
    commits = parse_commits(subjects)
    level = bump(commits)
    if level is None:
        print("no release-worthy commits; skipping")
        return 0
    version = next_version(current, level)
    today = date.today()
    entry = render_changelog_entry(version, level, commits, today)
    notes = render_release_notes(version, level, commits, today)
    if dry_run:
        print(f"would release v{version} ({level}):\n{entry}\n{notes}")
        return 0
    VERSION_FILE.write_text(version + "\n")
    old = CHANGELOG_FILE.read_text()
    marker = "\n## ["
    idx = old.find(marker)
    CHANGELOG_FILE.write_text(old[:idx + 1] + entry + "\n" + old[idx + 1:])
    sh("git", "add", "VERSION", "CHANGELOG.md")
    sh("git", "commit", "-m", f"chore(release): v{version}")
    sh("git", "tag", f"v{version}")
    sh("git", "push", "origin", "main", "--follow-tags")
    notes_path = Path(f"/tmp/release-notes-{version}.md")
    notes_path.write_text(notes)
    sh("gh", "release", "create", f"v{version}", "--title", f"v{version}",
       "--notes-file", str(notes_path))
    print(f"released v{version}")
    return 0


if __name__ == "__main__":
    sys.exit(main(dry_run="--dry-run" in sys.argv))
