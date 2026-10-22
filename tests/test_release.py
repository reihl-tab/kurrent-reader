"""Tests for the automated release script (scripts/release.py)."""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import release  # noqa: E402

D = date(2026, 10, 22)


class TestParseCommits:
    def test_conventional(self):
        cs = release.parse_commits(["feat: add ruled-table reader", "fix(cli): crash"])
        assert cs[0] == {"type": "feat", "breaking": False, "desc": "add ruled-table reader"}
        assert cs[1]["type"] == "fix" and cs[1]["desc"] == "crash"

    def test_breaking_bang(self):
        cs = release.parse_commits(["feat!: drop py3.9"])
        assert cs[0]["breaking"] is True

    def test_non_conventional_passthrough(self):
        cs = release.parse_commits(["wip misc stuff"])
        assert cs[0]["type"] == "other"


class TestBump:
    def test_fix_is_patch(self):
        assert release.bump(release.parse_commits(["fix: typo", "docs: readme"])) == "patch"

    def test_feat_is_minor(self):
        assert release.bump(release.parse_commits(["feat: x", "fix: y"])) == "minor"

    def test_breaking_is_major(self):
        assert release.bump(release.parse_commits(["fix!: api change", "feat: z"])) == "major"

    def test_chore_only_no_release(self):
        assert release.bump(release.parse_commits(["chore: lint", "ci: cache", "docs: x"])) is None

    def test_empty_no_release(self):
        assert release.bump([]) is None


class TestNextVersion:
    def test_patch(self):
        assert release.next_version("0.1.0", "patch") == "0.1.1"

    def test_minor(self):
        assert release.next_version("0.1.0", "minor") == "0.2.0"

    def test_major(self):
        assert release.next_version("0.1.0", "major") == "1.0.0"


class TestRender:
    COMMITS = release.parse_commits(["feat: weekly training pipeline", "fix: stdout guard"])

    def test_changelog_entry(self):
        e = release.render_changelog_entry("0.2.0", "minor", self.COMMITS, D)
        assert e.startswith("## [0.2.0] - 2026-10-22")
        assert "### Features" in e
        assert "- weekly training pipeline" in e and "- stdout guard" in e

    def test_notes_bilingual(self):
        n = release.render_release_notes("0.2.0", "minor", self.COMMITS, D)
        assert "### Features" in n and "### תכונות חדשות" in n
        assert "v0.2.0" in n

    def test_notes_breaking_marked(self):
        cs = release.parse_commits(["feat!: new config format"])
        n = release.render_release_notes("1.0.0", "major", cs, D)
        assert "**breaking**" in n and "שינויים שוברי תאימות" in n

    def test_notes_no_personal_data_placeholder(self):
        # policy guard: notes are built from commit subjects only
        n = release.render_release_notes("0.1.1", "patch", self.COMMITS, D)
        assert "@" not in n.split("###")[1]
