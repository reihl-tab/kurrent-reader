# Eschau 1081 labeling universe (2026-10-23)

- eschau1081_index.json: all 235 person records of the JUF person index whose quellen reference quelleId=1081 (Eschau Standesregister). Public archive data (all persons 19th century / long deceased).
- Method: full enumeration of POST https://juf.stadtarchiv-digital.de/read/85 with body {"limit":{"page":N,"count":500},"orderBy":{"column":"","direction":"DESC"},"searchParameters":[]} (70 pages, 34,907/34,907 rows, zero failed pages; ~1 req/s). API contract captured from the archive's own search UI (render/82). read/85 with no/limit body returns only 5 rows - the limit block is required. quellen carries anchors with quelleId + paketNr + primkey.
- Field availability: 218/235 with a date, 59/235 with beruf. All 6 spot-checked known IdNrs (33739, 35238, 38624, 38627, 38646, 38796) present.
- Paket distribution: register sections are the low paketNrs (0-28); high paketNrs (>=35, up to 4791265) look like cross-source/document references - filter before use. Membership must be rebuilt from raw quellen (entry-viewer-map.json is known-buggy); rows = children only via strict quellen regex (parents/witnesses inflate counts).
- paketNr -> scan file (s/g/hNN_22.jpg) mapping: NOT yet established - next step, via the archive viewer (JUF_viewer.php?quelleId=1081&paketNr=N), read-only.
- Prior labeled coverage: 78 IdNrs / 148 cells (see experiments/2026-10-23/). This universe is the pool for the new unique-label wave (hypothesis, not proven lever).

## paketNr -> scan file (2026-10-23)
Each record's 1081 anchors were resolved once via JUF_viewer.php (read-only): paket_map.json.
Sections: g=Geburten, h=Heiraten, s=Sterben. eschau1081_index.json now carries the full anchors
and a resolved scan_files list per person. Records whose only 1081 pakets are unprobed/high
(cross-source refs) have no scan_files yet.

## Page inventory + types (2026-10-26)
All 26 referenced scan files downloaded once (read-only, ~1 req/s) and classified visually:
page_types.json. Key finding: format varies within books by year - g01 (1821-22) and all h pages
are NARRATIVE prose; g02+ and s02-s08 are ruled TABLE SPREADS (two book-pages per scan, 2807px);
g09/g11/g12 look like narrative spreads (unconfirmed). Geometry extraction must be per-page-type:
table spreads need per-half banded rule detection (full-page projection fails on skew/light rules,
issue #4); narrative pages need line segmentation, not ruled_table. Raw detection output:
geometry_raw_fullpage.json. Downloaded pages themselves are NOT committed (archive ToS; re-fetch
from files.juf.stadtarchiv-digital.de when needed).
