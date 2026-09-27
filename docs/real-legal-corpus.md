# The real Indian legal corpus

This documents `legalrag.corpus` — the pipeline that acquires, extracts, validates and ingests
real Indian statutes and Supreme Court judgments, as distinct from the synthetic fixture corpus
used everywhere else in the repo (`legalrag/tests/fixtures/sample_corpus`, used by `phase0`,
`offline_demo`, `local_ollama`). The two corpora live in **separate databases**
(`legalrag` vs `legalrag_real`) and separate directories, and neither is deleted or overwritten by
the other: `real_corpus.yaml` is the only config that touches the real corpus.

## 1. Where the data comes from

| Document type | Source | Authority |
|---|---|---|
| Bharatiya Nyaya Sanhita, Bharatiya Nagarik Suraksha Sanhita, Bharatiya Sakshya Adhiniyam (2023) | Gazette-text PDFs hosted by the Ministry of Home Affairs (`www.mha.gov.in`) | `official` |
| Other central Acts (Constitution, Indian Contract Act, ...) | Ministry of Law and Justice, Legislative Department (`legislative.gov.in` / `lddashboard.legislative.gov.in`) | `official` |
| Supreme Court judgments | "Indian Supreme Court Judgments" dataset (Dattam Labs, AWS Open Data registry, CC-BY-4.0, derived from eCourts) | `supplementary` |

Every entry records both `source_authority` (what the manifest *claims*) and
`verification_status` (what `validate.py` actually *found*, see §2). Only a document whose
downloaded bytes carry the identity text the manifest expects, and whose host is a `.gov.in`/
`.nic.in` domain, can become `official_verified`. A third-party mirror copy (the AWS judgments
dataset is not the eCourts portal itself) is `supplementary_source` at best; it is not
byte-compared against the government original, and every such document is labelled with a
`SUPPLEMENTARY_COPY` warning that flows through to the final answer's disclaimer.

## 2. Why each source is trusted (and what isn't)

`legalrag/corpus/validate.py` decides trust from the downloaded bytes, not the manifest's say-so:

- `official_verified` needs (a) the manifest to claim `official`, (b) the URL's host to end in
  `.gov.in`/`.nic.in`, (c) the extracted text to contain the exact identity strings declared in
  `identity.all_of` (e.g. `"THE BHARATIYA NYAYA SANHITA, 2023"`, `"NO. 45 OF 2023"`), and (d) no
  error-severity validation finding.
- `supplementary_source`: a valid, correctly-identified document from a non-government host.
- `pending_review`: extraction succeeded but something needs a person (typically a page with no
  text layer and OCR off).
- `invalid`: never indexed. Reasons are logged (wrong document, corrupt file, duplicate, citation
  that doesn't match the PDF header, implausible date, ...) and printed by `corpus validate`.

**Sources not used, and why:** the official eCourts judgment portal
(`judgments.ecourts.gov.in/pdfsearch/`) is CAPTCHA-gated and is never automated. India Code
(`indiacode.nic.in` → `indiacode.gov.in`) has been unreachable from this environment on every
attempt (timeouts / 403 / 502). Indian Kanoon is a paid API and is not used at all. None of these
are silently substituted with something less authoritative — see §16 for the manifest-based way
this is recorded instead.

## 3. How documents are downloaded

`legalrag/corpus/fetch.py`'s `Fetcher`:

- refuses non-HTTPS URLs and any host not in `CorpusConfig.source_allowlist`;
- checks `robots.txt` before every new host (RFC 9309: a 4xx response means "no restriction", a
  5xx or unreachable host means "assume disallowed" — never assume permission);
- rate-limits to one request per `download_min_interval_s` (default 1s) **per host**;
- identifies honestly: `User-Agent: legalrag-research/0.1 (academic prototype; not a crawler)`.
  This is a deliberate choice, not an oversight — see §17 for what it costs;
- streams the download with a byte cap (`max_download_mb`) and validates it is really a PDF
  (`%PDF-` header, `%%EOF` trailer, not a CAPTCHA/error page) before writing it;
- writes to a `.part` file and renames atomically, so a crash never leaves a half-written file
  mistaken for a complete one;
- is idempotent: re-running `corpus fetch` on a file already on disk re-validates it and does not
  re-download, unless `--force` is given;
- records SHA-256, size, HTTP status, ETag/Last-Modified and retrieval time in
  `corpus/manifest.lock.json` (machine-written, kept separate from the human-reviewed manifest).

For hosts whose bot-protection blocks even an honestly-identified client (§17), the same
provenance guarantees are obtained through `corpus import <doc_uid> <path>`: a person downloads
the file in a real browser, and the command hashes, validates and registers it exactly as
`fetch` would, recording `method: manual_import` in the lock file so the distinction is never lost.

## 4. How PDFs are extracted

`legalrag/corpus/extract.py` (`pdfplumber`, `x_tolerance=1.0` to recover justified text whose word
gaps are narrower than the default tolerance):

- extracts per-page, so every character in the joined text can be mapped back to a PDF page
  (`ExtractedDoc.page_of(offset)`);
- separates a right/left marginal-note column (Gazette pages, where a section's title sits) from
  the body, by an x-coordinate cutoff (`margin_x`/`margin_left_x` per manifest entry);
- strips Supreme Court Reports margin letters (A, B, C, ... running down the right edge of a page)
  only when they actually form a strictly increasing run there — an ordinary capital letter at a
  line end is never touched;
- strips running headers/footers (including page numbers) only when the same normalised line
  repeats at the top/bottom of at least 30% of pages — never a one-off match;
- with `strip_footnotes: true` (India Code consolidations), drops the amendment footnotes below
  each page's footnote rule (a short horizontal line at the left text edge in the lower two-thirds
  of the page); they read like section starts ("2. Subs. by Act 3 of 1951, ...") and are not law.
  Such Acts use `numbering: india_code` (`parse_india_code`): it skips the "Arrangement of
  Sections", reads inline titles, lettered insertions (19A) and glued footnote markers ("1151." =
  marker 1 + s.151), and indexes repealed-section stubs ("76. [Title.] Rep. by ...");
- a PDF whose text was converted to vector outlines ("printed to PDF") has no text layer at all,
  looks normal on screen, and extracts as nothing: replace it with a copy whose text can be selected;
- flags a page with fewer than `min_chars_per_page` characters as `needs_ocr`; OCR (Tesseract via
  `pytesseract`) only runs when `corpus.ocr_enabled` is set and the binary is installed, otherwise
  the document is not ingested until a person resolves it.

Both the raw PDF (`raw/`) and the extracted text plus a page map (`extracted/<doc_uid>.txt` and
`.pages.json`) are kept on disk, so extraction is auditable against the original.

## 5. How cleaning works

There is no separate "cleaning" pass distinct from extraction: the same module removes only page
furniture (headers, footers, page numbers, margin letters) under the repeat/position conditions
above, and nothing else. Section numbers, sub-section numbers, article numbers, provisos,
headings, citations and case names are never touched — they are exactly the anchors
`structure.py` (§7) depends on. Extracted text is stored verbatim (`extracted/*.txt`) alongside
the original PDF, so the "raw vs cleaned" distinction the spec asks for is "PDF vs extracted text",
both kept, neither ever silently rewritten.

## 6. How legal metadata is generated

`legalrag/corpus/manifest.py` defines the metadata schema (Pydantic, `extra="forbid"`, so a typo
in a field name fails loudly rather than being silently ignored): source name/URL/licence, doc
type, identity anchors, language, document status, and — for statutes — `act_name`, `act_year`,
`act_number`, `enactment_date`, `period` (see §7); for judgments — `court`, `decided_on`,
`neutral_citation`, `reporter_citations`, `case_number`. **Nothing here is inferred or guessed**:
every field is either read directly from the source (the Gazette PDF's own printed text, the AWS
dataset's own metadata parquet) or left absent. `corpus/manifest.yaml` is hand-authored and
reviewed; `validate.py` then checks the claims against the downloaded bytes (§2) rather than
trusting them.

## 7. How chunking works

`legalrag/corpus/structure.py`:

- **Statutes**: Act → Chapter (`CHAPTER I - PRELIMINARY`) → Section → sub-section. A number is
  accepted as a new section only if it is the *next* number in sequence, so a numbered list
  inside a section's text, or a quoted section number, is never mistaken for a new provision — and
  a genuinely lost sequence is reported as a structural problem rather than silently truncating
  the Act. A section's marginal note (its title, e.g. "Murder penalty.") is read from the Gazette
  page's margin column beside it. A section short enough for one chunk stays one chunk; a longer
  one is split at its `(1)`, `(2)`, ... sub-sections, and if a sub-section is *still* too long it
  is split further by words with overlap, keeping the same section/sub-section metadata on every
  piece — a retrieved chunk never loses which provision it came from.
- **Judgments**: editorial front matter (headnote, list of acts, issue for consideration, ...) is
  labelled `headnote` and kept separate from the judgment text itself; a few blocks
  (appearances, "case arising from") are dropped outright as pure procedure. Numbered paragraphs
  are split one-per-chunk (or by words if a paragraph is very long) and classified by a keyword
  heuristic into `arguments` / `facts` / `reasoning` / `holding` / `judgment_text` — labelled
  `role_source: heuristic_v1` so a consumer knows this is a heuristic, not certainty.
- Every chunk's `char_start:char_end` is an exact slice of the stored extracted text
  (`extract_pdf` → `write_extracted`), checked by `test_corpus_units.py` and
  `test_corpus_db.py::test_every_chunk_is_a_verbatim_slice_...`.

## 8. How embeddings are generated

`real_corpus.yaml` configures `embedding.kind: sentence_transformers`,
`BAAI/bge-small-en-v1.5` (384-d, CPU, batch size 16 — sized for this machine's 4GB GPU/6GB RAM by
staying on CPU). What is actually embedded (`legalrag/corpus/ingest.py:embed_text`) is not the
bare chunk text but the chunk **plus the context its verbatim text lacks** — which Act/section, or
which case — e.g. `"Bharatiya Nyaya Sanhita, 2023, section 103 - Punishment for murder. <text>"`.
The chunk stored and returned as evidence is always the untouched verbatim text; only the
embedding input is enriched.

## 9. How data enters PostgreSQL + pgvector

`legalrag/corpus/ingest.py:build()` processes one manifest entry at a time, in its own
transaction: validate → extract → parse → validate again → decide trust status (§2) → index (or
skip, with the reason logged to `ingestion_events`). A document that fails validation **never
reaches `documents`**. Re-running is idempotent: `_same_chunks()` compares the new chunk spans
against what's stored, and an unchanged document is neither re-chunked nor re-embedded. A
document that has *become* invalid or pending on a re-run (e.g. a tampered file) is removed from
the index and the removal is logged, never silently left stale. Full provenance
(`document_provenance`: source URL, licence, retrieval time, hash, extraction tool/method, OCR
flag, page count, applicable period) is recorded for every indexed document — see schema in
`legalrag/store/migrations/0002_real_corpus.sql`, additive only, nothing dropped or renamed from
the synthetic-corpus schema.

## 10. How retrieval works

Unchanged from the rest of the project (`legalrag/pipeline/retrieval.py`): lexical `ts_rank_cd` +
dense cosine search, fused by Reciprocal Rank Fusion, then a metadata-boost rerank
(`real_corpus.yaml` additionally configures a real cross-encoder,
`cross-encoder/ms-marco-MiniLM-L-6-v2`). What real-corpus data adds is a **trust filter**:
`retrieval.allowed_verification_statuses: [official_verified, supplementary_source]` means
`pending_review`/`invalid` documents can never be retrieved, even if a build run indexed one with
`--include-pending` for testing (`test_retrieval_only_returns_trusted_documents` in
`test_corpus_db.py` proves the filter, not absence, is what hides them).

## 11. How temporal statute versions work

A statute's `period.valid_from`/`valid_until` (manifest) becomes `statute_sections.valid_from`/
`valid_until` in the database. Two manifest entries for the *same* Act name and year with
overlapping validity windows are refused by `validate_manifest` (`CONFLICTING_VERSIONS`) before
either can be indexed — a new version is added by giving it its own `valid_from` (and closing the
old one's `valid_until`), never by overwriting the old text in place. `verify.as_of` in the config
is the date the `temporal` check evaluates every statute reference against.

**Known limitation** (see §17): the *query itself* is not parsed for a date. Asking "what was the
law in 2020?" does not make retrieval or verification switch to a 2020 view of the corpus — it
evaluates as of `verify.as_of`. Building query-date extraction is a query-analysis feature, not a
corpus one, and is out of scope for this increment.

## 12. How citations are generated

`legalrag/pipeline/generate.py` (`source_kind`, `pinpoint`) types every source
(`statutory_text`, `reporter_headnote`, `party_arguments`, `court_reasoning`, `court_holding`,
`judgment_text_unclassified`) and gives it a pinpoint (`"s. 103, p. 34"`, `"para 6, p. 8"`) built
from the same page/section/paragraph metadata `structure.py` attached to the chunk. Every
`source_details` entry also carries `doc_uid`, `verification_status` and `source_authority`, so a
citation is traceable: chunk → document (`doc_uid`) → `document_provenance.source_url` → the
manifest's `source_url`. Nothing is cited that isn't a chunk actually retrieved and verified.

## 13. How verification uses the corpus

Unchanged architecture (`legalrag/verify/`), five checks per (claim, evidence) pair
(`authority`, `citation_metadata`, `statute`, `claim_support`, `temporal`) — see the main
`README.md`. What real-corpus data changes is configuration, not code:
`lexical_can_verify: false` (word overlap can FLAG but never VERIFY — it cannot detect a
contradiction, and there is no real generator installed to make an LLM judge meaningful yet) and
`judge_prompt: v2` (direct-quote vs inferred-interpretation, for when a real generator is added).
Confidence on every answer run against `real_corpus` in this state is capped low, correctly:
without a real generator the "claims" being verified are the fake generator's placeholder text.

## 14. How to add new statutes

1. Find the authoritative URL yourself (an allowlisted `.gov.in`/`.nic.in` host, or note why one
   can't be reached — see §16).
2. Add an entry to `corpus/manifest.yaml`: `doc_uid`, `source`, `local_file`, `identity.all_of`
   (exact text from the PDF's own first pages), `act_name`/`act_year`/`period.valid_from`.
3. If the host is reachable automatically: add it to `corpus.source_allowlist` in
   `legalrag/config/real_corpus.yaml`, then `corpus fetch --config real_corpus --only <doc_uid>`.
   If not: download it yourself and `corpus import --config real_corpus <doc_uid> <path>`.
4. `corpus validate --config real_corpus --only <doc_uid>` — fix manifest fields (never the PDF
   text) until it validates or is honestly `pending_review`.
5. `corpus build --config real_corpus --only <doc_uid>`.

An older/differently-typeset Act may need small, additive extensions to
`structure.py:parse_statute`'s heading/margin-note regexes — never a full rewrite, and never at
the cost of accepting a section the parser can't actually locate (an unparseable Act stays
`invalid`, logged, not force-indexed).

## 15. How to add new judgments

`corpus discover-sc --config real_corpus --years Y1 Y2 --terms "..." --out path.yaml` screens the
AWS metadata for judgments whose first 3 pages match any of `--terms`, caching PDFs under
`<corpus_dir>/discovery` (never the corpus itself) and writing **candidates only** — nothing is
added to the manifest automatically. A person reviews the candidates file, moves the ones that
belong into `corpus/manifest.yaml`, then `fetch` → `validate` → `build` as above.

## 16. How to re-run/extend ingestion

```powershell
docker compose up -d db                                             # Postgres + pgvector
python -m legalrag.corpus fetch    --config real_corpus             # or `import` for manual files
python -m legalrag.corpus validate --config real_corpus
python -m legalrag.corpus build    --config real_corpus
python -m legalrag.corpus report   --config real_corpus             # data-quality counts
python -m legalrag.corpus status   --config real_corpus
python -m legalrag.pipeline.orchestrate --config real_corpus --query "..."
python -m legalrag.eval.compare    --config real_corpus
python -m pytest -q
```

Every step is idempotent and safe to re-run: `fetch`/`build` skip unchanged documents, `validate`
never mutates anything. Sources that stay unreachable (blocked host, CAPTCHA) are recorded, not
silently swapped for a lower-quality substitute: they remain absent from `manifest.yaml` with the
URL and the reason noted in this document and in `reports/`, until either the host becomes
reachable or a person supplies the file via `corpus import`.

## 17. Known limitations

- **Official `.gov.in` sources actively block the honest User-Agent.** `www.mha.gov.in` (which
  this pipeline fetched from successfully before) and `www.legislative.gov.in`/
  `lddashboard.legislative.gov.in` all returned HTTP 403 specifically to
  `legalrag-research/0.1 (academic prototype; not a crawler)` in live testing, while an unset or
  ordinary browser User-Agent succeeded on the same URL. This is not a robots.txt restriction
  (robots.txt itself 403s the same way) — it is UA-fingerprint bot-blocking, and it appears to
  have started after this corpus's first successful `mha.gov.in` fetches. The project keeps its
  honest, self-identifying User-Agent rather than spoofing a browser, so every additional statute
  from a blocked host currently requires a person to download it and run `corpus import`.
- **India Code** (`indiacode.nic.in`/`indiacode.gov.in`) has not been reachable from this
  environment on any attempt. **eCourts** (`judgments.ecourts.gov.in`) is CAPTCHA-gated and is
  never automated, by design.
- **Supreme Court judgments are supplementary, not official.** They come from a CC-BY-4.0
  third-party mirror of eCourts data, compressed and re-derived — not byte-identical to the
  official record. Every such document is labelled `supplementary_source` and its citations carry
  that caveat through to the final answer.
- **Temporal checking does not parse dates out of the query.** "What was the law in 2020?"
  evaluates against `verify.as_of` (a config-wide date), not a date extracted from the question.
- **Schedules are not indexed** (BNS/BNSS/BSA's own schedules are explicitly skipped by
  `parse_statute`, logged as a warning, not silently dropped without a trace).
- **No real generator is configured.** `real_corpus.yaml` uses `llm.provider: fake`; retrieval,
  reranking and verification of hand-written claims are real, but end-to-end example answers are
  plumbing, not model output, until a local (Ollama) or API generator is wired in the same way
  `local_ollama.yaml` does for the synthetic corpus.
- **The corpus is still small** relative to the spec's eventual target (12 statutes / 50-100+
  judgments): see `reports/corpus_report.json` (via `corpus report`) for the exact current counts.
