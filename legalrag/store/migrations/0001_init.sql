-- Core schema. `{{EMBED_DIM}}` is substituted by the migration runner from config so the
-- vector column width always matches the configured embedding model.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Records the embedding model/dim the chunk vectors were built with, so a mismatched
-- config fails loudly instead of returning garbage similarities.
CREATE TABLE store_meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);

CREATE TABLE documents (
    id                  bigserial PRIMARY KEY,
    doc_type            text NOT NULL CHECK (doc_type IN ('judgment', 'statute')),
    title               text NOT NULL,
    court               text,
    decided_on          date,
    neutral_citation    text,
    reporter_citations  jsonb NOT NULL DEFAULT '[]'::jsonb,
    source_url          text,
    raw_text_path       text NOT NULL UNIQUE,   -- natural key: makes ingestion idempotent
    sha256              text NOT NULL
);
CREATE INDEX documents_doc_type_idx   ON documents (doc_type);
CREATE INDEX documents_court_idx      ON documents (court);
CREATE INDEX documents_decided_on_idx ON documents (decided_on);
CREATE INDEX documents_neutral_idx    ON documents (neutral_citation);
CREATE INDEX documents_sha256_idx     ON documents (sha256);
CREATE INDEX documents_title_trgm_idx ON documents USING gin (title gin_trgm_ops);

CREATE TABLE chunks (
    id             bigserial PRIMARY KEY,
    document_id    bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ordinal        integer NOT NULL,
    text           text NOT NULL,
    char_start     integer NOT NULL,
    char_end       integer NOT NULL,
    section_label  text,                         -- headnote|facts|arguments|reasoning|holding|statute_section|...
    embedding      vector({{EMBED_DIM}}) NOT NULL,
    -- Postgres full-text ranking (ts_rank_cd). Not true BM25; the evaluation report
    -- must call it "lexical (ts_rank_cd)" unless BM25 is added later.
    tsv            tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (document_id, ordinal),
    CHECK (char_end > char_start)
);
CREATE INDEX chunks_document_idx      ON chunks (document_id);
CREATE INDEX chunks_section_label_idx ON chunks (section_label);
CREATE INDEX chunks_tsv_idx           ON chunks USING gin (tsv);
CREATE INDEX chunks_embedding_idx     ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE statute_sections (
    id                bigserial PRIMARY KEY,
    document_id       bigint REFERENCES documents (id) ON DELETE CASCADE,
    chunk_id          bigint REFERENCES chunks (id) ON DELETE SET NULL,  -- the retrievable chunk for this version
    act_name          text NOT NULL,
    act_year          integer NOT NULL,
    section_number    text NOT NULL,
    subsection        text NOT NULL DEFAULT '',
    text              text NOT NULL,
    valid_from        date NOT NULL,
    valid_until       date,                       -- NULL = still in force
    superseded_by_id  bigint REFERENCES statute_sections (id) ON DELETE SET NULL,
    amendment_note    text,
    UNIQUE (act_name, act_year, section_number, subsection, valid_from),
    CHECK (valid_until IS NULL OR valid_until > valid_from)
);
CREATE INDEX statute_sections_act_idx        ON statute_sections (act_name, act_year);
CREATE INDEX statute_sections_section_idx    ON statute_sections (section_number);
CREATE INDEX statute_sections_validity_idx   ON statute_sections (valid_from, valid_until);
CREATE INDEX statute_sections_superseded_idx ON statute_sections (superseded_by_id);
CREATE INDEX statute_sections_document_idx   ON statute_sections (document_id);
CREATE INDEX statute_sections_chunk_idx      ON statute_sections (chunk_id);

-- Deliberately larger than `documents`: name+citation records for cases we do NOT hold
-- full text for, so the authority check can tell "does not exist" from "outside corpus".
CREATE TABLE citation_registry (
    id           bigserial PRIMARY KEY,
    case_name    text NOT NULL,
    citation     text NOT NULL,
    court        text,
    decided_on   date,
    in_corpus    boolean NOT NULL DEFAULT false,
    document_id  bigint REFERENCES documents (id) ON DELETE SET NULL,
    UNIQUE (citation, case_name)
);
CREATE INDEX citation_registry_citation_idx ON citation_registry (citation);
CREATE INDEX citation_registry_court_idx    ON citation_registry (court);
CREATE INDEX citation_registry_decided_idx  ON citation_registry (decided_on);
CREATE INDEX citation_registry_in_corpus_idx ON citation_registry (in_corpus);
CREATE INDEX citation_registry_document_idx ON citation_registry (document_id);
CREATE INDEX citation_registry_name_trgm_idx ON citation_registry USING gin (case_name gin_trgm_ops);

-- Evaluation / query logging. `runs` is one row per pipeline invocation (one query);
-- `experiment` groups the rows of one evaluation batch.
CREATE TABLE runs (
    id             bigserial PRIMARY KEY,
    experiment     text NOT NULL,
    config_name    text NOT NULL,
    config_hash    text NOT NULL,
    corpus_sha     text NOT NULL,
    model_info     jsonb NOT NULL,
    query          text NOT NULL,
    outcome        text NOT NULL CHECK (outcome IN ('answered', 'abstained', 'error')),
    final_answer   jsonb,
    trace          jsonb NOT NULL DEFAULT '{}'::jsonb,   -- per-stage structured output
    latency_ms     jsonb NOT NULL DEFAULT '{}'::jsonb,   -- per-stage latency
    created_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX runs_experiment_idx  ON runs (experiment);
CREATE INDEX runs_config_hash_idx ON runs (config_hash);
CREATE INDEX runs_outcome_idx     ON runs (outcome);
CREATE INDEX runs_created_idx     ON runs (created_at);

-- One row per Evidence object per stage per run. `evidence` is the serialised
-- Evidence dataclass; metrics are computed from it, not from ad hoc side channels.
CREATE TABLE run_items (
    id          bigserial PRIMARY KEY,
    run_id      bigint NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    stage       text NOT NULL CHECK (stage IN ('retrieved', 'reranked', 'verified')),
    claim_idx   integer,                    -- set for stage = 'verified'
    rank        integer NOT NULL,
    chunk_id    bigint,
    status      text,
    evidence    jsonb NOT NULL
);
CREATE INDEX run_items_run_idx    ON run_items (run_id);
CREATE INDEX run_items_stage_idx  ON run_items (stage);
CREATE INDEX run_items_chunk_idx  ON run_items (chunk_id);
CREATE INDEX run_items_status_idx ON run_items (status);

CREATE TABLE check_results (
    id            bigserial PRIMARY KEY,
    run_item_id   bigint NOT NULL REFERENCES run_items (id) ON DELETE CASCADE,
    check_name    text NOT NULL,
    outcome       text NOT NULL CHECK (outcome IN ('PASS', 'WARN', 'FAIL', 'NOT_APPLICABLE')),
    score         double precision NOT NULL CHECK (score >= 0 AND score <= 1),
    evidence_pointer text,
    explanation   text NOT NULL DEFAULT '',
    UNIQUE (run_item_id, check_name)
);
CREATE INDEX check_results_item_idx    ON check_results (run_item_id);
CREATE INDEX check_results_name_idx    ON check_results (check_name);
CREATE INDEX check_results_outcome_idx ON check_results (outcome);
