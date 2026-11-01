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
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise subprocess.CalledProcessError(
            exc.returncode, exc.cmd, exc.stdout, exc.stderr
        ) from None


RELEASE_SUBJECT_RE = re.compile(r"^chore\(release\): v(\d+\.\d+\.\d+)$")


def tag_exists(tag: str) -> bool:
    return subprocess.run(
        ("git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"),
        capture_output=True,
    ).returncode == 0


def complete_interrupted_release(version: str, commit: str) -> None:
    """Tag + publish a release commit that was pushed without its tag.

    A release job that fails after `git push` but before `gh release create`
    leaves `chore(release): vX` on main with no tag; finish it instead of
    skipping forever or cutting the next version on top.
    """
    subjects = sh("git", "log", f"{commit}^", "--pretty=%s").splitlines()
    level = bump(parse_commits(subjects)) or "patch"
    day = date.fromisoformat(sh("git", "show", "-s", "--format=%cs", commit))
    notes = render_release_notes(version, level, parse_commits(subjects), day)
    sh("git", "tag", "-a", f"v{version}", "-m", f"v{version}", commit)
    sh("git", "push", "origin", f"refs/tags/v{version}")
    notes_path = Path(f"/tmp/release-notes-{version}.md")
    notes_path.write_text(notes)
    sh("gh", "release", "create", f"v{version}", "--title", f"v{version}",
       "--notes-file", str(notes_path))
    print(f"completed interrupted release v{version} at {commit[:7]}")


def find_untagged_release_commit() -> tuple[str, str] | None:
    """Newest chore(release) commit on HEAD's first-parent chain without its tag."""
    for line in sh("git", "log", "--first-parent", "--pretty=%H %s").splitlines():
        sha, subject = line.split(" ", 1)
        m = RELEASE_SUBJECT_RE.match(subject)
        if m and not tag_exists(f"v{m.group(1)}"):
            return m.group(1), sha
        if m:
            return None  # newest release commit is tagged; history is healthy
    return None


def push_with_rebase_retry(attempts: int = 3) -> None:
    """Push main, rebasing onto a concurrently-moved origin/main (issue #21)."""
    for attempt in range(1, attempts + 1):
        try:
            sh("git", "push", "origin", "main")
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            sh("git", "fetch", "origin", "main")
            sh("git", "rebase", "origin/main")


def main(dry_run: bool = False) -> int:
    sh("git", "fetch", "origin", "main", "--tags")
    interrupted = find_untagged_release_commit()
    if interrupted and not dry_run:
        complete_interrupted_release(*interrupted)
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
    current = VERSION_FILE.read_text().strip()
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
    push_with_rebase_retry()
    sh("git", "tag", "-a", f"v{version}", "-m", f"v{version}")
    sh("git", "push", "origin", f"refs/tags/v{version}")
    notes_path = Path(f"/tmp/release-notes-{version}.md")
    notes_path.write_text(notes)
    sh("gh", "release", "create", f"v{version}", "--title", f"v{version}",
       "--notes-file", str(notes_path))
    print(f"released v{version}")
    return 0


if __name__ == "__main__":
    sys.exit(main(dry_run="--dry-run" in sys.argv))
