# legalrag: confidence-aware verification for Indian legal RAG

Research prototype from the project synopsis *Confidence-Aware, Source-Grounded Verification of
AI-Generated Legal Answers*: **retrieve the evidence, verify the evidence, then generate the
answer.** Everything in the sample corpus is a synthetic fixture, not real Indian law. Not legal
advice.

```
query -> analysis -> [coverage guard] -> hybrid retrieval -> rerank -> draft -> decompose
      -> verify (authority | citation_metadata | statute | claim_support | temporal)
      -> VERIFIED / FLAGGED / REJECTED -> grounded final answer (+ caveats, sources, confidence)
```

## Status

| piece | state |
|---|---|
| Ingest, Postgres + pgvector store, statute version history, citation registry | done |
| Query analysis, hybrid retrieval (ts_rank_cd + dense, RRF), metadata-boost reranking | done |
| **Verification layer**: five real checks, config-driven VERIFIED/FLAGGED/REJECTED rules, caveats | done |
| Coverage guard (abstain when the question names an authority the corpus cannot ground) | done |
| Three arms: baseline LLM, standard RAG, verification-enhanced RAG | done |
| Benchmark (question, gold authority, verbatim gold passage, answer rubric) + AVA probes | done, **synthetic** |
| Metrics: accuracy, Recall@K, CVR, CSR, EGA, UCR, AVA, HCER, error taxonomy | done |
| Real embedder / cross-encoder / local LLM adapters | written and unit-tested with stub models; **not yet run against real models** |
| Real Indian corpus, real benchmark, real measured results | **not started** (needs data, see below) |

Numbers from the offline demo prove the harness works. They say nothing about model quality.

## Setup (Windows)

```powershell
uv venv C:\Users\<you>\venvs\legalrag --python 3.12     # keep the venv out of OneDrive
$env:VIRTUAL_ENV = "C:\Users\<you>\venvs\legalrag"; uv pip install -e ".[dev,api]"
docker compose up -d db                                  # Postgres 16 + pgvector on 127.0.0.1:5432
```

Use `127.0.0.1`, not `localhost`: on Windows `localhost` tries IPv6 first and every
connection stalls ~130 s. The default DSN already does this; override with `LEGALRAG_DSN`.

## Run

```powershell
python -m legalrag.ingest.build --config offline_demo    # idempotent; prints corpus statistics
python -m legalrag.pipeline.orchestrate --config offline_demo --query "..."
python -m legalrag.eval.compare --config offline_demo    # 3 arms x 18 questions -> results/<ts>-offline_demo.{json,md}
python -m pytest                                         # DB tests skip with a reason if Postgres is down
python -m legalrag.ui.server --config offline_demo --open   # web UI on http://127.0.0.1:8765
```

### Web UI

`legalrag/ui/` is a local, single-user frontend with no extra dependencies (standard-library server, plain
HTML/JS, no build step). It binds to `127.0.0.1` only and needs the same Postgres as everything else.

- **Ask**: one question through the baseline, standard RAG and verified RAG side by side. Verified answers show
  per-statement status, caveats and sources; "Inspect verification detail" opens the query analysis, coverage
  guard, every claim with its five check results and explanations, and the retrieved passages. Citations from the
  other two arms are labelled as unchecked. Deep link: `#ask?q=...&arms=verified_rag&run=1`.
- **History**: every question asked in the UI (or any logged run), reopened with its full evidence trail.
- **Benchmark**: saved `eval.compare` results, with the plumbing-run warning kept prominent.
- **Corpus**: documents, statute version history and the citation registry.
- The **Config** selector switches configs live and keeps the old one if the new one cannot load (missing `[ml]`
  extra, or an embedding model that does not match the database). Questions asked here are logged to `runs`
  under experiment `ui`. The hidden draft is never sent to the browser.

### Configs

| config | generator | embeddings / reranker | verifier |
|---|---|---|---|
| `phase0` | fake | hash / lexical | fake (plumbing only) |
| `offline_demo` | fake | hash / lexical | **real**, lexical claim-support |
| `local_ollama` | Ollama `qwen2.5:3b-instruct` | MiniLM / MiniLM cross-encoder | **real**, LLM claim-support |

`local_ollama` needs `ollama pull qwen2.5:3b-instruct` and `uv pip install -e ".[ml]"` (pulls torch).
It is sized for ~6 GB RAM and a 4 GB GPU (embedder and reranker stay on CPU). The embedding model is
pinned into the database, so changing it needs a fresh DB. Set `verify.as_of` in every experiment
config: an unset date means "today" and makes runs irreproducible.

## How verification works

Each (claim, evidence) pair gets five checks (`legalrag/verify/`), each PASS / WARN / FAIL /
NOT_APPLICABLE with an explanation and an evidence pointer:

| check | catches |
|---|---|
| `authority` | fabricated cases (name or citation not in the registry); real cases whose text we do not hold (WARN) |
| `citation_metadata` | real citation on the wrong case name; wrong year or court |
| `statute` | a section that does not exist in the Act; an Act that does not exist |
| `claim_support` | evidence that does not establish the claim; 'arguments' sections used as if they were holdings |
| `temporal` | superseded statute versions; judgments predating an amendment to a section they apply; conflicting versions |

A FAIL in a `reject_on_fail` check means REJECTED; any other FAIL or WARN means FLAGGED; VERIFIED also
needs `claim_support` to PASS. With `include_flagged: true`, claims resting only on FLAGGED evidence
survive **with their caveats attached in code** (the model never rewrites them). Answer confidence is the
weakest claim behind the answer.

## Known limitations

- **Verification checks support, not relevance.** A claim can be faithfully supported by evidence that
  does not answer the question, and it will still be scored as well supported. The coverage guard closes the
  case where the question names an authority we lack; the general case (unrelated but real support) is open.
  Confidence does not yet include retrieval relevance because the reranker's scores are not calibrated.
- `claim_support` with the lexical judge is word overlap (plus a number-mismatch guard); it cannot detect a
  contradiction. Use `support_judge: llm` for real runs. The report states how well the audit judge agrees
  with the gold claim labels.
- Reference extraction is regex-based and conservative: a reference it cannot parse is not checked.
- Which Act a judgment's section belongs to is inferred from the passage text; when ambiguous it is skipped.
- Temporal checking flags for review; it does not decide what current law is.
- AVA probes were written alongside the checks: they are a regression suite, not a held-out estimate.
- Answer accuracy is phrase-rule based (transparent but coarse). HCER compares verification-derived
  confidence (verified arm) with model-stated confidence (other arms).
- The API-backed generator cannot be run at temperature 0; see `legalrag/llm/anthropic_api.py`.

## Layout

- `legalrag/verify/`: the checks, `rules.py` (status engine), `query_check.py` (coverage guard), `build.py`.
- `legalrag/pipeline/`: stages, `arms.py` (the three systems), versioned prompts in `prompts/`.
- `legalrag/eval/`: `benchmark.py`, `metrics.py`, `probes.py` (AVA), `compare.py`, data in `eval/data/`.
- `legalrag/authority.py`: in-memory index of registry + statute versions; built from Postgres or, for
  offline tests, from the corpus files (`ingest/corpus_view.py`).
- `legalrag/ui/`: `server.py` (JSON API + serialisers) and `static/` (the frontend).
- `legalrag/llm/`, `legalrag/embed/`: single adapter interfaces (`fake`, `ollama`, `anthropic`; `hash`, `sentence_transformers`).
- `legalrag/settings.py`: config models; `config_hash` covers everything except `db`.

## Moving to real data

Replace the fixture corpus with Supreme Court judgments and Acts in the same front-matter format
(`ingest/parse.py`), add a real registry listing CSV, write a benchmark in the `eval/data/*.jsonl` format
(`benchmark.py` validates that every gold passage occurs verbatim in the corpus), and point
`ingest.corpus_dir` and `eval.benchmark_path` at them. Chunking still counts whitespace words rather than
the embedding model's tokens.
#   M i c r o  
 