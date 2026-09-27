-- Real-corpus support. ADDITIVE ONLY: nothing is dropped, renamed or rewritten, so the synthetic
-- fixture corpus and every existing query keep working. Legacy rows get explicit defaults that say
-- what they are (source_authority = 'synthetic').

ALTER TABLE documents
    ADD COLUMN doc_uid              text,           -- stable manifest id, e.g. ACT-BNS-2023
    ADD COLUMN source_authority     text NOT NULL DEFAULT 'synthetic'
        CHECK (source_authority IN ('official', 'supplementary', 'synthetic')),
    ADD COLUMN verification_status  text NOT NULL DEFAULT 'pending_review'
        CHECK (verification_status IN ('official_verified', 'supplementary_source', 'pending_review', 'invalid')),
    ADD COLUMN original_url         text,           -- URL of the original file, where it differs from source_url
    ADD COLUMN retrieved_at         timestamptz,
    ADD COLUMN source_sha256        text,           -- hash of the ORIGINAL file as downloaded (sha256 = extracted text)
    ADD COLUMN language             text;
CREATE UNIQUE INDEX documents_doc_uid_idx       ON documents (doc_uid) WHERE doc_uid IS NOT NULL;
CREATE UNIQUE INDEX documents_source_sha256_idx ON documents (source_sha256) WHERE source_sha256 IS NOT NULL;
CREATE INDEX documents_verification_idx         ON documents (verification_status);

-- Structural location of each chunk in the ORIGINAL document, so a retrieved passage can be
-- checked against the source: pages, section/paragraph, and the heading it sits under.
ALTER TABLE chunks
    ADD COLUMN page_start        integer,
    ADD COLUMN page_end          integer,
    ADD COLUMN act_name          text,
    ADD COLUMN chapter           text,
    ADD COLUMN section_number    text,
    ADD COLUMN subsection        text,
    ADD COLUMN heading           text,              -- e.g. the marginal note of a statute section
    ADD COLUMN paragraph_number  text,
    ADD COLUMN metadata          jsonb NOT NULL DEFAULT '{}'::jsonb;  -- role_source, opinion index, ...
CREATE INDEX chunks_section_number_idx ON chunks (act_name, section_number);

-- The lexical index must see a statute section's title (its marginal note) and Act name, which live
-- outside the verbatim chunk text. `tsv` is a derived column, so it is recreated (recomputed from the
-- same rows) with those fields included. For legacy rows heading/act_name are NULL, so the indexed
-- tokens - and therefore ts_rank_cd scores - are identical to before.
ALTER TABLE chunks DROP COLUMN tsv;
ALTER TABLE chunks ADD COLUMN tsv tsvector GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(heading, '') || ' ' || coalesce(act_name, '') || ' ' || text)) STORED;
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);

-- Where each document came from and what was done to it.
CREATE TABLE document_provenance (
    document_id        bigint PRIMARY KEY REFERENCES documents (id) ON DELETE CASCADE,
    manifest_id        text NOT NULL,
    source_name        text NOT NULL,
    source_url         text NOT NULL,
    source_license     text,
    retrieved_at       timestamptz NOT NULL,
    source_version     text,                        -- ETag / Last-Modified / dataset release, whatever the source gave
    source_file_path   text NOT NULL,
    source_sha256      text NOT NULL,
    source_size_bytes  bigint NOT NULL,
    extraction_method  text NOT NULL,               -- 'pdfplumber' | 'pdfplumber+ocr' | ...
    extraction_tool    text NOT NULL,               -- tool + version
    ocr_used           boolean NOT NULL DEFAULT false,
    n_pages            integer NOT NULL,
    processing_status  text NOT NULL CHECK (processing_status IN ('ingested', 'ingested_with_warnings')),
    validation_errors  jsonb NOT NULL DEFAULT '[]'::jsonb,
    validation_warnings jsonb NOT NULL DEFAULT '[]'::jsonb,
    applicable_from    date,
    applicable_until   date,
    period_basis       text,                        -- how the applicable period was established
    processed_at       timestamptz NOT NULL DEFAULT now()
);

-- Every attempt, including the failures: a document that is rejected never reaches `documents`,
-- so this table is where "why is it missing?" gets answered.
CREATE TABLE ingestion_events (
    id           bigserial PRIMARY KEY,
    manifest_id  text NOT NULL,
    stage        text NOT NULL,                     -- fetch | validate | extract | chunk | index
    status       text NOT NULL CHECK (status IN ('ok', 'warning', 'failed', 'rejected', 'skipped')),
    message      text NOT NULL DEFAULT '',
    detail       jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ingestion_events_manifest_idx ON ingestion_events (manifest_id);
CREATE INDEX ingestion_events_status_idx   ON ingestion_events (status);
