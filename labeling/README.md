# Eschau 1081 labeling universe (2026-10-23)

- eschau1081_index.json: all 235 person records of the JUF person index whose quellen reference quelleId=1081 (Eschau Standesregister). Public archive data (all persons 19th century / long deceased).
- Method: full enumeration of POST https://juf.stadtarchiv-digital.de/read/85 with body {"limit":{"page":N,"count":500},"orderBy":{"column":"","direction":"DESC"},"searchParameters":[]} (70 pages, 34,907/34,907 rows, zero failed pages; ~1 req/s). API contract captured from the archive's own search UI (render/82). read/85 with no/limit body returns only 5 rows - the limit block is required. quellen carries anchors with quelleId + paketNr + primkey.
- Field availability: 218/235 with a date, 59/235 with beruf. All 6 spot-checked known IdNrs (33739, 35238, 38624, 38627, 38646, 38796) present.
- Paket distribution: register sections are the low paketNrs (0-28); high paketNrs (>=35, up to 4791265) look like cross-source/document references - filter before use. Membership must be rebuilt from raw quellen (entry-viewer-map.json is known-buggy); rows = children only via strict quellen regex (parents/witnesses inflate counts).
- paketNr -> scan file (s/g/hNN_22.jpg) mapping: NOT yet established - next step, via the archive viewer (JUF_viewer.php?quelleId=1081&paketNr=N), read-only.
- Prior labeled coverage: 78 IdNrs / 148 cells (see experiments/2026-10-23/). This universe is the pool for the new unique-label wave (hypothesis, not proven lever).
