# Versioning standard

This repo follows semantic versioning, automated end to end. Releases are
lightweight labeled snapshots of `main`; continuous deployment (where a site or
service exists) is unchanged.

## Rules

- **semver**: bug fix = PATCH, feature = MINOR, breaking change = MAJOR.
- **Conventional commits** drive the bump: `fix:`/`perf:` -> patch,
  `feat:` -> minor, `BREAKING CHANGE` footer or `!:` -> major.
  `docs:`/`chore:`/`test:`/`ci:`-only ranges produce no release.
- Every merge to `main` that contains a release-worthy commit produces a
  release automatically (several per day is fine): the CI bumps `VERSION`,
  prepends a CHANGELOG section, tags `vX.Y.Z`, and creates a GitHub release
  with notes in **English and Hebrew**.
- **Tests**: every bug fix ships with a regression test that fails without the
  fix. PRs merge only with CI green.
- **Version display**: repos with a site or app show the version in the footer.
  Library/CLI repos (like this one) expose it as `kurrent-reader --version`
  reading the single source of truth: the `VERSION` file at repo root.
- Release notes never include personal details about the owner or family.
