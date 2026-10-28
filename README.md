# Kurrent Reader

Planned free local reading aid for 19th-century German Kurrent register scans.
An extra blind reader only: never an authority for a transcription, and never a
Geni editor or synchronization tool.

## Default-branch status

The default branch currently contains this README only. It has no installable
package, reader command, test suite, training runner or release automation.
There is therefore no install or test command to run from main yet.

Work is preserved in separate, unmerged PRs:

| PR | Proposed work | Status caveat |
| --- | --- | --- |
| [#2](https://github.com/reihlinstinct/kurrent-reader/pull/2) | Cleanup, segmentation, TrOCR reader and pipeline | Draft implementation; not shipped on main |
| [#4](https://github.com/reihlinstinct/kurrent-reader/pull/4) | Table crops and training/evaluation pipeline | Draft; existing hold remains in effect |
| [#6](https://github.com/reihlinstinct/kurrent-reader/pull/6) | Versioned Colab runners | Separate runner proposal, not a main-branch command |
| [#8](https://github.com/reihlinstinct/kurrent-reader/pull/8) | Version and release automation | Proposed automation, not enabled by this README |
| [#9](https://github.com/reihlinstinct/kurrent-reader/pull/9) | Experiment preservation | Backup only; keep unmerged |

PR descriptions and reported experiment scores are not independently verified
results for the default branch. Model requirements, licenses, memory limits,
training/evaluation separation and reproducible checks must be reviewed with
the relevant implementation before it becomes a supported workflow. This page
does not lift a hold or authorize merging any of those PRs.

## Intended safeguards

Preserve original scans. Keep cleaned derivatives and proposed readings
separate, retain uncertain alternatives and require human source review before
promoting a reading to a genealogical fact. Do not infer missing people or facts
from OCR output. Keep private scans and identifying material in their approved
private audience, not public examples or logs.

Changes should start with an issue and a small signed PR, with tests where code
exists and honest limits where it does not. Independent review is required
before merge; a successful test or merge is not approval to train, publish data
or change any external family tree.
