"""Real-corpus acquisition, validation, extraction and legal-structure parsing.

manifest.py  the reviewed list of documents and their claimed provenance (+ machine-written lock file)
fetch.py     allowlisted, rate-limited, robots-respecting download; manual import for CAPTCHA-gated sources
extract.py   PDF -> pages of text with layout facts (margin notes, page furniture removed)
structure.py Acts -> chapter/section/sub-section; reported judgments -> headnote + numbered paragraphs
validate.py  the checks and the official_verified / supplementary_source / pending_review / invalid decision
ingest.py    manifest -> Postgres (per-document transaction, idempotent, failures recorded)
"""
