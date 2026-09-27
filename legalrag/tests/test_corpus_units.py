"""Unit tests for the real-corpus layer: fetch policy, extraction, legal structure, validation.
All PDFs are generated here (tests/pdfmaker.py); nothing depends on a downloaded document."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest

from legalrag.corpus.extract import extract_pdf
from legalrag.corpus.fetch import FetchError, Fetcher, check_pdf_bytes, import_file, sha256_bytes, sha256_file
from legalrag.corpus.manifest import LockRecord, Manifest, ManifestEntry, utcnow
from legalrag.corpus.structure import classify_role, parse_judgment, parse_statute
from legalrag.corpus.validate import (
    ValidationReport, decide_status, is_official_host, validate_extracted, validate_manifest, validate_source)
from legalrag.settings import CorpusConfig
from legalrag.tests.pdfmaker import Item, lines_page, make_pdf

CFG = CorpusConfig()


def write_pdf(tmp_path: Path, pages: list[list[Item]], name: str = "doc.pdf") -> Path:
    p = tmp_path / name
    p.write_bytes(make_pdf(pages))
    return p


# ------------------------------------------------------------------------------------ hashing / bytes

def test_sha256_matches_between_bytes_and_file(tmp_path):
    data = make_pdf([lines_page(["hello"])])
    p = tmp_path / "a.pdf"
    p.write_bytes(data)
    assert sha256_bytes(data) == sha256_file(p) and len(sha256_bytes(data)) == 64


@pytest.mark.parametrize("data,why", [
    (b"", "too small"),
    (b"<html>" + b"x" * 2000 + b"</html>", "not a PDF"),                     # e.g. a CAPTCHA/error page
    (make_pdf([lines_page(["hello"] * 3)])[:-40] + b"x" * 1100, "truncated"),  # no %%EOF at the end
])
def test_check_pdf_bytes_rejects_bad_files(data, why):
    with pytest.raises(FetchError):
        check_pdf_bytes(data)


def test_check_pdf_bytes_accepts_a_real_pdf():
    check_pdf_bytes(make_pdf([lines_page(["x " * 400])]), min_bytes=100)


# ------------------------------------------------------------------------------------ fetch policy

def _entry(**kw) -> ManifestEntry:
    base = dict(
        doc_uid="ACT-TEST-2023", doc_type="statute", title="Test Act", source_authority="official",
        source={"name": "MHA", "url": "https://www.mha.gov.in/x.pdf"}, local_file="acts/x.pdf",
        identity={"all_of": ["TEST"]}, act_name="Test Sanhita", act_year=2023,
        period={"valid_from": "2024-07-01", "basis": "gazette_text"})
    base.update(kw)
    return ManifestEntry.model_validate(base)


def _fetcher(handler, **cfgkw) -> Fetcher:
    cfg = CorpusConfig(download_min_interval_s=0.0, **cfgkw)
    return Fetcher(cfg, httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True))


PDF = make_pdf([lines_page(["a real looking document " * 40])])


def test_fetch_refuses_hosts_outside_the_allowlist(tmp_path):
    f = _fetcher(lambda r: httpx.Response(200, content=PDF))
    e = _entry(source={"name": "x", "url": "https://evil.example.com/x.pdf"})
    with pytest.raises(FetchError, match="allowlist"):
        f.fetch(e, tmp_path)
    assert not list(tmp_path.rglob("*.pdf"))


def test_fetch_refuses_plain_http(tmp_path):
    f = _fetcher(lambda r: httpx.Response(200, content=PDF))
    with pytest.raises(FetchError, match="https"):
        f.fetch(_entry(source={"name": "x", "url": "http://www.mha.gov.in/x.pdf"}), tmp_path)


def test_fetch_respects_robots_disallow(tmp_path):
    def handler(req):
        if req.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /")
        return httpx.Response(200, content=PDF)
    with pytest.raises(FetchError, match="robots"):
        _fetcher(handler).fetch(_entry(), tmp_path)


def test_fetch_treats_unreachable_robots_as_disallow(tmp_path):
    def handler(req):
        return httpx.Response(503) if req.url.path == "/robots.txt" else httpx.Response(200, content=PDF)
    with pytest.raises(FetchError, match="robots"):
        _fetcher(handler).fetch(_entry(), tmp_path)


def test_fetch_does_not_retry_or_work_around_a_403(tmp_path):
    seen = []

    def handler(req):
        seen.append(req.headers.get("user-agent"))
        return httpx.Response(404) if req.url.path == "/robots.txt" else httpx.Response(403)
    with pytest.raises(FetchError, match="403"):
        _fetcher(handler).fetch(_entry(), tmp_path)
    assert all("legalrag" in ua for ua in seen)  # always the honest agent string


def test_fetch_writes_original_bytes_unmodified_and_records_the_hash(tmp_path):
    def handler(req):
        return httpx.Response(404) if req.url.path == "/robots.txt" else httpx.Response(
            200, content=PDF, headers={"etag": '"abc"', "content-type": "application/pdf"})
    rec = _fetcher(handler).fetch(_entry(), tmp_path)
    assert (tmp_path / "acts/x.pdf").read_bytes() == PDF
    assert rec.sha256 == sha256_bytes(PDF) and rec.size_bytes == len(PDF) and rec.etag == '"abc"'


def test_fetch_rejects_an_html_page_served_as_200(tmp_path):
    def handler(req):
        return httpx.Response(404) if req.url.path == "/robots.txt" else httpx.Response(200, content=b"<html>captcha</html>" * 200)
    with pytest.raises(FetchError, match="not a PDF"):
        _fetcher(handler).fetch(_entry(), tmp_path)
    assert not (tmp_path / "acts/x.pdf").exists()


def test_fetch_enforces_the_size_cap(tmp_path):
    big = b"%PDF-1.4\n" + b"x" * (2 * 1024 * 1024) + b"\n%%EOF\n"

    def handler(req):
        return httpx.Response(404) if req.url.path == "/robots.txt" else httpx.Response(200, content=big)
    with pytest.raises(FetchError, match="max_download_mb"):
        _fetcher(handler, max_download_mb=1).fetch(_entry(), tmp_path)


def test_manual_import_copies_bytes_and_marks_the_method(tmp_path):
    src = tmp_path / "downloaded.pdf"
    src.write_bytes(PDF)
    rec = import_file(_entry(), src, tmp_path / "raw")
    assert rec.method == "manual_import" and (tmp_path / "raw/acts/x.pdf").read_bytes() == PDF


# ------------------------------------------------------------------------------------ extraction

def test_extract_keeps_pages_and_offsets_consistent(tmp_path):
    pages = [lines_page([f"page {i} line {j}" for j in range(3)], header="RUNNING HEADER", page_number=i)
             for i in range(1, 5)]
    d = extract_pdf(write_pdf(tmp_path, pages), CFG)
    assert d.n_pages == 4
    for r in d.line_refs():
        assert d.text[r.start:r.end] == r.text
        assert d.page_of(r.start) == r.page
    assert all("RUNNING HEADER" in " ".join(p.removed) for p in d.pages)   # repeated furniture removed
    assert "RUNNING HEADER" not in d.text


def test_a_three_line_running_header_including_the_page_number_is_removed(tmp_path):
    topics = {52: "alpha", 53: "bravo", 54: "charlie", 55: "delta"}

    def page(n):
        return [(122, 40, "THE GAZETTE OF INDIA EXTRAORDINARY"), (293, 52, str(n)), (122, 64, "_" * 30),
                (142, 90, f"{n}. Body text about {topics[n]} that must stay."), (118, 104, f"{topics[n]} continues here."),
                (118, 118, f"second {topics[n]} line."), (118, 132, f"third {topics[n]} line."), (118, 146, f"fourth {topics[n]} line."),
                (118, 160, f"fifth {topics[n]} line."), (118, 174, f"sixth {topics[n]} line."), (118, 188, f"seventh {topics[n]} line.")]
    d = extract_pdf(write_pdf(tmp_path, [page(n) for n in (52, 53, 54, 55)]), CFG)
    for p in d.pages:
        assert len(p.lines) == 8 and p.lines[0].text.startswith(f"{p.number + 51}. Body"), p.lines
        assert len(p.removed) == 3


def test_a_line_that_appears_once_is_not_treated_as_furniture(tmp_path):
    pages = [lines_page(["unique first line", "body"], header=None) for _ in range(4)]
    pages[0][0] = (118, 90, "ONLY ON PAGE ONE")
    d = extract_pdf(write_pdf(tmp_path, pages), CFG)
    assert "ONLY ON PAGE ONE" in d.text


def test_margin_columns_are_split_on_both_sides(tmp_path):
    right = [(142, 90, "5. Body text of the right page."), (490, 91, "Right note.")]
    left = [(142, 90, "6. Body text of the left page."), (58, 91, "Left note.")]
    d = extract_pdf(write_pdf(tmp_path, [right, left, right]), CFG, margin_x=483, margin_left_x=116)
    assert "Right note" not in d.text and "Left note" not in d.text
    assert [m.text for m in d.pages[0].margin] == ["Right note."]
    assert [m.text for m in d.pages[1].margin] == ["Left note."]
    assert d.pages[1].lines[0].text.startswith("6. Body")


def test_margin_letters_are_dropped_only_as_an_ascending_run(tmp_path):
    def page(letters):
        return [(118, 90 + 20 * i, f"line {i} of the report") for i in range(len(letters))] + \
               [(520, 90 + 20 * i, ch) for i, ch in enumerate(letters)]
    d = extract_pdf(write_pdf(tmp_path, [page("ABC"), page("A"), page("BA")]), CFG)
    assert "margin letters ABC" in d.pages[0].removed and " A" not in d.pages[0].text
    # a lone "A" and a descending pair are ordinary text/furniture the rule must not touch
    assert d.pages[1].text.count("A") >= 1 and "B" in d.pages[2].text and "A" in d.pages[2].text


def test_a_page_without_text_is_flagged_not_silently_accepted(tmp_path):
    d = extract_pdf(write_pdf(tmp_path, [lines_page(["real text here " * 5]), [], lines_page(["more text " * 8])]), CFG)
    assert [p.number for p in d.pages if p.needs_ocr] == [2]
    assert any("no text layer" in w for w in d.warnings)


# ------------------------------------------------------------------------------------ statutes

def gazette_pages() -> list[list[Item]]:
    hdr = lambda n: [(122, 40, f"THE GAZETTE OF INDIA EXTRAORDINARY {n}"), (122, 52, "_" * 40)]  # noqa: E731
    p1 = hdr(1) + [
        (274, 90, "CHAPTERI"), (273, 106, "PRELIMINARY"),
        (142, 130, "1. (1) This Act may be called the Test Sanhita, 2023."), (490, 131, "Short title."),
        (118, 144, "It extends to the whole of India."),
        (142, 166, "2.In this Sanhita, unless the context otherwise requires,"), (490, 167, "Definitions."),
        (166, 180, '(1) "act" denotes a series of acts;'),
        (166, 194, '(2) "court" means a Judge empowered to act judicially;'),
    ]
    def sub_lines(label: str, y0: float, first_prefix: str = "") -> list[Item]:
        # 24 words wrapped over 3 lines (a real page wraps well inside the body column)
        words = [f"{label}w{i}" for i in range(24)]
        return [(142 if k == 0 else 118, y0 + 14 * k, (first_prefix if k == 0 else "") + " ".join(words[8 * k:8 * k + 8]))
                for k in range(3)]

    p2 = hdr(2) + [
        (58, 90, "Murder"), (58, 100, "penalty."), (58, 110, "2 of 1974."),
        (142, 91, "3. Whoever commits murder shall be punished with death."),
        (58, 150, "Long"), (58, 160, "rule."),
    ] + sub_lines("a", 151, "4. (1) ") + sub_lines("b", 200, "(2) ") + sub_lines("c", 250, "(3) ") + [
        (271, 300, "CHAPTERII"), (250, 316, "OF PUNISHMENTS"),
        (250, 340, "Of offences against property"),
        (142, 362, "5. Whoever steals shall be punished."),
    ]
    p3 = hdr(3) + [
        (490, 91, "Repeal and savings."),
        (142, 90, "6. The old law is repealed."),
        (271, 130, "THE FIRST SCHEDULE"),
        (142, 150, "1. Not a section, a schedule row."),
    ]
    return [p1, p2, p3]


def test_parse_statute_finds_chapters_sections_titles_and_subsections(tmp_path):
    d = extract_pdf(write_pdf(tmp_path, gazette_pages()), CFG, margin_x=483, margin_left_x=116)
    st = parse_statute(d, "Test Sanhita", max_words=60)
    assert st.problems == [] and st.n_sections == 6
    by = {(u.section_number, u.subsection): u for u in st.units}
    assert by[("1", "")].heading == "Short title." and by[("1", "")].chapter == "CHAPTER I - PRELIMINARY"
    assert by[("2", "")].text.startswith("2.In this Sanhita")            # "2.In" with no space still a section
    assert by[("3", "")].heading == "Murder penalty."           # left-hand margin; "2 of 1974." cross-ref removed
    assert by[("5", "")].chapter == "CHAPTER II - OF PUNISHMENTS"
    assert "Of offences against property" not in by[("4", "3")].text      # centred sub-heading left out of section 4
    assert {k for k in by if k[0] == "4"} == {("4", "1"), ("4", "2"), ("4", "3")}  # long section split at sub-sections
    assert by[("6", "")].heading == "Repeal and savings."
    assert "Not a section" not in by[("6", "")].text                     # the Schedule is not swallowed
    assert any("Schedules are not indexed" in w for w in st.warnings)


def test_every_statute_chunk_is_an_exact_slice_with_its_pages(tmp_path):
    d = extract_pdf(write_pdf(tmp_path, gazette_pages()), CFG, margin_x=483, margin_left_x=116)
    st = parse_statute(d, "Test Sanhita", max_words=60)
    for u in st.units:
        assert u.chunks and d.text[u.char_start:u.char_end] == u.text
        for c in u.chunks:
            assert d.text[c.char_start:c.char_end] == c.text
            assert c.section_number == u.section_number and c.act_name == "Test Sanhita"
            assert 1 <= c.page_start <= c.page_end <= d.n_pages


def test_a_short_section_is_never_split_and_sections_are_never_merged(tmp_path):
    d = extract_pdf(write_pdf(tmp_path, gazette_pages()), CFG, margin_x=483, margin_left_x=116)
    st = parse_statute(d, "Test Sanhita", max_words=500)
    assert [(u.section_number, u.subsection) for u in st.units] == [(str(i), "") for i in range(1, 7)]
    assert all(len(u.chunks) == 1 for u in st.units)


def test_a_lost_section_sequence_is_reported_not_silently_truncated(tmp_path):
    pages = [[(274, 90, "CHAPTERI"), (273, 106, "PRELIMINARY"),
              (142, 130, "1. First section text here."), (142, 150, "2. Second section text here."),
              (142, 170, "3 Third section lost its full stop"), (142, 190, "4. Fourth section text here."),
              (142, 210, "5. Fifth section text here.")]] * 1
    d = extract_pdf(write_pdf(tmp_path, pages), CFG)
    st = parse_statute(d, "Test Sanhita", max_words=100)
    assert any("stalled after 2" in p for p in st.problems)


def test_no_chapter_heading_is_a_problem(tmp_path):
    d = extract_pdf(write_pdf(tmp_path, [lines_page(["just some text " * 10])]), CFG)
    assert parse_statute(d, "X", max_words=50).problems


# ------------------------------------------------------------------------- constitutions (PART/Article)

def constitution_pages() -> list[list[Item]]:
    toc = [
        (122, 40, "CONTENTS"), (122, 60, "PART I"), (122, 74, "THE UNION AND ITS TERRITORY"),
        (122, 88, "ARTICLES"),
        (122, 102, "1. Name and territory of the Union .......................... 2"),
        (122, 116, "2. Admission of new States .......................... 2"),
    ]
    body = [
        (122, 40, "PART I"), (122, 54, "THE UNION AND ITS TERRITORY"),
        (142, 80, "1. Name and territory of the Union.-(1) India shall be a Union of States."),
        (142, 94, "(2) The territories shall be as specified in the First Schedule."),
        (142, 116, "2. Admission of new States.-Parliament may by law admit new States."),
        (142, 138, "2A. Transitional provision.-This provision applies during the transition."),
        (142, 160, "3. Formation of new States.-Parliament may alter boundaries of States."),
        (271, 190, "THE FIRST SCHEDULE"), (142, 210, "1. Not indexed schedule content."),
    ]
    return [toc, body]


def test_parse_constitution_skips_toc_and_finds_parts_articles_and_letter_insertions(tmp_path):
    from legalrag.corpus.structure import parse_constitution

    d = extract_pdf(write_pdf(tmp_path, constitution_pages()), CFG)
    st = parse_constitution(d, "Constitution of India", max_words=200)
    assert st.problems == [] and st.n_sections == 4
    by = {u.section_number: u for u in st.units}
    assert set(by) == {"1", "2", "2A", "3"}
    assert by["1"].heading == "Name and territory of the Union"
    assert by["1"].chapter == "PART I - THE UNION AND ITS TERRITORY"
    assert "India shall be a Union of States" in by["1"].text
    assert "Not indexed schedule" not in by["3"].text                # the Schedule is not swallowed
    assert any("Schedules are not indexed" in w for w in st.warnings)
    # the ToC's dot-leadered "1."/"2." entries were not mistaken for the real articles:
    assert "Union .." not in by["1"].text and ".......... 2" not in by["2"].text


def test_constitution_chunks_are_exact_slices_of_the_extracted_text(tmp_path):
    from legalrag.corpus.structure import parse_constitution

    d = extract_pdf(write_pdf(tmp_path, constitution_pages()), CFG)
    st = parse_constitution(d, "Constitution of India", max_words=200)
    for u in st.units:
        assert u.chunks and d.text[u.char_start:u.char_end] == u.text
        for c in u.chunks:
            assert d.text[c.char_start:c.char_end] == c.text
            assert c.act_name == "Constitution of India" and c.section_number == u.section_number


def test_constitution_article_numbering_stall_is_reported(tmp_path):
    from legalrag.corpus.structure import parse_constitution

    pages = [[
        (122, 40, "PART I"), (122, 54, "THE UNION AND ITS TERRITORY"),
        (142, 80, "1. Name and territory of the Union.-(1) India shall be a Union of States."),
        (142, 100, "2 Admission of new States lost its full stop and is unrecognisable."),
        (142, 120, "3. Formation of new States.-Parliament may alter boundaries of States."),
    ]]
    d = extract_pdf(write_pdf(tmp_path, pages), CFG)
    st = parse_constitution(d, "Constitution of India", max_words=200)
    assert any("stalled after 1" in p for p in st.problems)


def test_no_part_or_chapter_heading_outside_toc_is_a_problem(tmp_path):
    from legalrag.corpus.structure import parse_constitution

    d = extract_pdf(write_pdf(tmp_path, [lines_page(["just some text " * 10])]), CFG)
    assert parse_constitution(d, "Constitution of India", max_words=50).problems


# ------------------------------------------------------------------------------------ judgments

def judgment_pages() -> list[list[Item]]:
    furn = lambda n: [(118, 36, "Digital Supreme Court Reports"), (118, 48, f"page {n} [2024] 10 S.C.R.")]  # noqa: E731
    l = lambda ys, texts, x=75: [(x, y, t) for y, t in zip(ys, texts)]  # noqa: E731
    p1 = furn(1)[:0] + [(118, 45, "[2024] 10 S.C.R. 930 : 2024 INSC 819")] + l(
        [78, 93, 108, 126, 147, 168, 201, 219, 240, 258, 290, 305, 320, 335, 350, 365],
        ["Central Bureau of Investigation", "v.", "Ashok Sirpal", "(Criminal Appeal No. 4277 of 2024)", "24 October 2024",
         "[Abhay S. Oka and Augustine George Masih, JJ.]", "Issue for Consideration", "Matter pertains to the order suspending sentence.",
         "Headnotes", "Held: 1.1 The court may suspend the fine. [Paras 6-10]", "Case Arising From",
         "CRIMINAL APPELLATE JURISDICTION: Criminal Appeal No. 4277 of 2024", "Appearances for Parties",
         "Advs. for the Appellant.", "Judgment", "1. The factual controversy in this appeal is very limited."])
    p2 = furn(2) + l([90, 104, 118, 132, 146, 160], [
        "2. On 19th March 2018 while issuing notice this Court passed an order.",
        "3. Shri K M Nataraj, learned Additional Solicitor General of India, submitted that the order be set aside.",
        "4. The FIR was registered against the respondent in the year 2016.",
        "5. We are of the view that the sentence of fine can be suspended.",
        "6. Accordingly, the appeal is allowed.", "7. Section 389 of the Code reads thus."])
    p3 = furn(3) + l([90, 104, 118], ["8. A closing paragraph of the judgment.", "Result of the case: Appeal disposed of.",
                                        "Headnotes prepared by: Someone"])
    return [p1, p2, p3]


def test_parse_judgment_headnote_paragraphs_roles_and_tail(tmp_path):
    d = extract_pdf(write_pdf(tmp_path, judgment_pages()), CFG)
    j = parse_judgment(d, max_words=200)
    assert j.problems == []
    assert j.reporter_citation == "[2024] 10 S.C.R. 930" and j.neutral_citation == "2024 INSC 819"
    assert j.n_paragraphs == 8                                    # "1.1 The court..." inside the headnote is not a paragraph
    head = [c for c in j.chunks if c.section_label == "headnote"]
    assert head and all(c.metadata.get("editorial") for c in head)
    assert not any("Appearances" in c.text or "Advs. for" in c.text for c in j.chunks)   # metadata blocks dropped
    paras = {c.paragraph_number: c for c in j.chunks if c.paragraph_number}
    assert list(paras) == [str(i) for i in range(1, 9)]
    assert paras["3"].section_label == "arguments" and paras["4"].section_label == "facts"
    assert paras["5"].section_label == "reasoning" and paras["6"].section_label == "holding"
    assert paras["7"].section_label == "judgment_text"            # nothing matched: left unclassified, not guessed
    assert all(c.metadata["role_source"] == "heuristic_v1" for c in paras.values())
    assert "Result of the case" not in d.text[paras["8"].char_start:paras["8"].char_end]
    for c in j.chunks:
        assert d.text[c.char_start:c.char_end] == c.text and c.page_start >= 1


def test_no_judgment_marker_falls_back_to_paragraph_one_with_a_warning(tmp_path):
    pages = judgment_pages()
    pages[0] = [it for it in pages[0] if it[2] != "Judgment"]
    j = parse_judgment(extract_pdf(write_pdf(tmp_path, pages), CFG), max_words=200)
    assert j.problems == [] and any("no 'JUDGMENT' marker" in w for w in j.warnings)


def test_a_judgment_with_no_numbered_paragraphs_is_a_problem(tmp_path):
    pages = [[(118, 45, "[2024] 10 S.C.R. 930 : 2024 INSC 819"), (75, 100, "[A B, JJ.]"), (75, 120, "Judgment"),
              (75, 140, "Unnumbered prose only " * 5)]] * 3
    j = parse_judgment(extract_pdf(write_pdf(tmp_path, pages), CFG), max_words=200)
    assert j.problems


@pytest.mark.parametrize("text,role", [
    ("Learned counsel for the appellant submitted that the notice was defective.", "arguments"),
    ("The respondent contended that the order was illegal.", "arguments"),
    ("We hold that the decree is not maintainable.", "holding"),
    ("The appeal is accordingly allowed and the impugned order is set aside.", "holding"),
    ("The FIR was registered on 3rd March 2019.", "facts"),
    ("In our considered opinion the High Court erred.", "reasoning"),
    ("Section 389 of the Code reads as follows.", "judgment_text"),
    # party arguments beat holding words inside them: what a party said is not what the court held
    ("Learned counsel submitted that we hold the appeal to be maintainable.", "arguments"),
])
def test_role_heuristics(text, role):
    assert classify_role(text) == role


# ------------------------------------------------------------------------------------ validation

def _lock(path: Path, uid="ACT-TEST-2023") -> LockRecord:
    return LockRecord(doc_uid=uid, url="https://www.mha.gov.in/x.pdf", retrieved_at=utcnow(), sha256=sha256_file(path),
                      size_bytes=path.stat().st_size)


def _raw(tmp_path, entry, pdf: bytes = PDF) -> tuple[Path, LockRecord]:
    dest = tmp_path / "raw" / entry.local_file
    dest.parent.mkdir(parents=True)
    dest.write_bytes(pdf)
    return tmp_path / "raw", _lock(dest, entry.doc_uid)


def test_official_claim_needs_a_government_host():
    assert is_official_host("https://www.mha.gov.in/a.pdf") and not is_official_host("https://x.s3.amazonaws.com/a.pdf")


def test_official_claim_on_a_mirror_is_invalid(tmp_path):
    e = _entry(source={"name": "mirror", "url": "https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com/a.pdf"})
    raw, lock = _raw(tmp_path, e)
    rep = validate_source(e, lock, raw, CFG, {e.doc_uid: lock})
    assert any("OFFICIAL_CLAIM_UNSUPPORTED" in m for m in rep.errors) and decide_status(e, rep) == "invalid"


def test_unacquired_missing_and_tampered_files_are_invalid(tmp_path):
    e = _entry()
    assert "NOT_ACQUIRED" in validate_source(e, None, tmp_path, CFG, {}).errors[0]
    raw, lock = _raw(tmp_path, e)
    (raw / e.local_file).write_bytes(PDF + b"tampered\n%%EOF\n")
    assert any("HASH_MISMATCH" in m for m in validate_source(e, lock, raw, CFG, {e.doc_uid: lock}).errors)
    (raw / e.local_file).unlink()
    assert any("MISSING_FILE" in m for m in validate_source(e, lock, raw, CFG, {e.doc_uid: lock}).errors)


def test_corrupt_pdf_is_invalid(tmp_path):
    e = _entry()
    raw, lock = _raw(tmp_path, e, b"<html>not a pdf</html>" * 200)
    assert any("CORRUPT_FILE" in m for m in validate_source(e, lock, raw, CFG, {e.doc_uid: lock}).errors)


def test_duplicate_files_are_detected(tmp_path):
    a, b = _entry(), _entry(doc_uid="ACT-TEST2-2023", local_file="acts/y.pdf")
    raw, la = _raw(tmp_path, a)
    (raw / b.local_file).write_bytes(PDF)
    lb = _lock(raw / b.local_file, b.doc_uid)
    rep = validate_source(b, lb, raw, CFG, {a.doc_uid: la, b.doc_uid: lb})
    assert any("DUPLICATE_FILE" in m for m in rep.errors)


def test_statute_without_an_applicable_period_is_invalid(tmp_path):
    e = _entry(period={})
    raw, lock = _raw(tmp_path, e)
    assert any("NO_APPLICABLE_PERIOD" in m for m in validate_source(e, lock, raw, CFG, {e.doc_uid: lock}).errors)


def test_secondary_source_period_and_repealed_acts_warn(tmp_path):
    e = _entry(period={"valid_from": "2024-07-01", "basis": "unverified_secondary"}, document_status="repealed")
    raw, lock = _raw(tmp_path, e)
    w = " ".join(validate_source(e, lock, raw, CFG, {e.doc_uid: lock}).warnings)
    assert "UNVERIFIED_PERIOD" in w and "OUTDATED_LEGISLATION" in w


def test_judgment_citation_formats_are_checked(tmp_path):
    j = dict(doc_type="judgment", court="Supreme Court of India", decided_on="2024-10-24", act_name=None, act_year=None,
             period={}, document_status="reported_judgment", source_authority="supplementary",
             source={"name": "m", "url": "https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com/a.pdf"})
    e = _entry(neutral_citation="2024 SCC 12", **j)
    raw, lock = _raw(tmp_path, e)
    assert any("BAD_CITATION_FORMAT" in m for m in validate_source(e, lock, raw, CFG, {e.doc_uid: lock}).errors)
    e2 = _entry(neutral_citation="2024 INSC 819", reporter_citations=["10 SCR 930"], **j)
    assert any("BAD_CITATION_FORMAT" in m for m in validate_source(e2, lock, raw, CFG, {e2.doc_uid: lock}).errors)


def _judgment_entry(**kw) -> ManifestEntry:
    base = dict(doc_uid="SC-2024-INSC-819", doc_type="judgment", title="CBI v. Ashok Sirpal", court="Supreme Court of India",
                decided_on="2024-10-24", neutral_citation="2024 INSC 819", reporter_citations=["[2024] 10 S.C.R. 930"],
                source_authority="supplementary", document_status="reported_judgment", act_name=None, act_year=None,
                period={}, source={"name": "m", "url": "https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com/a.pdf"},
                local_file="sc/a.pdf", identity={"pages": 2, "all_of": ["2024 INSC 819"]})
    base.update(kw)
    return ManifestEntry.model_validate(base)


def _validate_judgment(tmp_path, entry) -> ValidationReport:
    d = extract_pdf(write_pdf(tmp_path, judgment_pages()), CFG)
    return validate_extracted(entry, d, parse_judgment(d, max_words=200), ValidationReport(entry.doc_uid))


def test_a_matching_judgment_validates_and_a_mirror_copy_is_supplementary(tmp_path):
    e = _judgment_entry()
    rep = _validate_judgment(tmp_path, e)
    assert rep.errors == [] and decide_status(e, rep) == "supplementary_source"


def test_a_manually_promoted_mirror_copy_becomes_official_verified(tmp_path):
    e = _judgment_entry(promoted_by="reviewer", promoted_on="2026-09-21", promotion_note="compared with sci.gov.in PDF")
    assert decide_status(e, _validate_judgment(tmp_path, e)) == "official_verified"


@pytest.mark.parametrize("kw,code", [
    (dict(neutral_citation="2024 INSC 900", identity={"pages": 2, "all_of": ["2024 INSC 900"]}), "IDENTITY_MISMATCH"),
    (dict(neutral_citation="2024 INSC 820", identity={"pages": 2, "all_of": ["2024 INSC 819"]}), "CITATION_MISMATCH"),
    (dict(reporter_citations=["[2024] 10 S.C.R. 999"]), "CITATION_MISMATCH"),
    (dict(decided_on="2024-10-25"), "DATE_MISMATCH"),
    (dict(identity={"pages": 2, "all_of": ["Some Other Party Name"]}), "IDENTITY_MISMATCH"),
])
def test_wrong_document_is_invalid(tmp_path, kw, code):
    e = _judgment_entry(**kw)
    rep = _validate_judgment(tmp_path, e)
    assert any(code in m for m in rep.errors) and decide_status(e, rep) == "invalid"


def test_scanned_pages_without_ocr_mean_pending_review_not_ingestion(tmp_path):
    e = _judgment_entry()
    pages = judgment_pages()
    pages.append([])                                    # a page with no text layer
    d = extract_pdf(write_pdf(tmp_path, pages), CFG)
    rep = validate_extracted(e, d, parse_judgment(d, max_words=200), ValidationReport(e.doc_uid))
    assert rep.errors == [] and any("NEEDS_OCR" in m for m in rep.reviews)
    assert decide_status(e, rep) == "pending_review"


def test_a_broken_section_sequence_makes_a_statute_invalid(tmp_path):
    e = _entry(identity={"all_of": ["CHAPTER"]})
    pages = [[(274, 90, "CHAPTERI"), (142, 130, "1. First section text " * 6), (142, 150, "3. Third " * 30)]] * 3
    d = extract_pdf(write_pdf(tmp_path, pages), CFG)
    rep = validate_extracted(e, d, parse_statute(d, "Test Sanhita", max_words=100), ValidationReport(e.doc_uid))
    assert decide_status(e, rep) == "invalid"


def test_empty_text_is_invalid(tmp_path):
    e = _entry(identity={"all_of": ["x"]})
    d = extract_pdf(write_pdf(tmp_path, [lines_page(["hi"])]), CFG)
    rep = validate_extracted(e, d, parse_statute(d, "X", max_words=50), ValidationReport(e.doc_uid))
    assert any("EMPTY_TEXT" in m for m in rep.errors)


# ------------------------------------------------------------------------------------ manifest

def test_manifest_rejects_duplicate_ids_and_files():
    a, b = _entry(), _entry()
    with pytest.raises(ValueError, match="duplicate doc_uid"):
        Manifest(entries=[a, b])
    with pytest.raises(ValueError, match="share a local_file"):
        Manifest(entries=[a, _entry(doc_uid="ACT-OTHER-2023")])


def test_manifest_entry_shape_rules():
    with pytest.raises(ValueError, match="act_name and act_year"):
        _entry(act_name=None)
    with pytest.raises(ValueError, match="promotion needs"):
        _judgment_entry(promoted_by="me")
    with pytest.raises(ValueError):
        _entry(unknown_field=1)


def test_conflicting_act_versions_and_duplicate_citations_are_flagged():
    a = _entry(period={"valid_from": "2024-07-01", "basis": "gazette_text"})
    b = _entry(doc_uid="ACT-TEST-V2", local_file="acts/y.pdf", period={"valid_from": "2025-01-01", "basis": "gazette_text"})
    out = validate_manifest(Manifest(entries=[a, b]), CFG)
    assert out[a.doc_uid] and out[b.doc_uid] and out[a.doc_uid][0].code == "CONFLICTING_VERSIONS"
    a2 = _entry(period={"valid_from": "2024-07-01", "valid_until": "2024-12-31", "basis": "gazette_text"})
    ok = validate_manifest(Manifest(entries=[a2, b]), CFG)
    assert not ok[a2.doc_uid] and not ok[b.doc_uid]                       # consecutive, non-overlapping: fine
    j1, j2 = _judgment_entry(), _judgment_entry(doc_uid="SC-2024-INSC-819-B", local_file="sc/b.pdf")
    dup = validate_manifest(Manifest(entries=[j1, j2]), CFG)
    assert dup[j2.doc_uid][0].code == "DUPLICATE_CITATION"


# ------------------------------------------------------------------------------------ India Code Acts

def india_code_pages() -> list[list[Item]]:
    toc = [
        (200, 40, "THE TEST CONTRACT ACT, 1872"), (200, 60, "ARRANGEMENT OF SECTIONS"),
        (250, 80, "PRELIMINARY"), (90, 94, "1. Short title."),
        (250, 110, "CHAPTER I"), (180, 124, "OF PROPOSALS"),
        (90, 138, "2. Communication of proposals."), (90, 152, "2A. Power to set aside."),
        (90, 166, "3. Revocation how made."),
    ]
    body = [
        (200, 40, "THE TEST CONTRACT ACT, 1872"),
        (90, 70, "1. Short title.-This Act may be called the Test Contract Act, 1872."),
        (250, 90, "CHAPTER I"), (180, 104, "OF PROPOSALS"),
        (90, 124, "2.Communication of proposals.-The communication of proposals is made by any act"),
        (72, 138, "of the party proposing in 1[India]."),
        (90, 152, "2A. Power to set aside.-Where consent was caused by undue influence the"),
        (72, 166, "contract is voidable."),
        (250, 186, "Void agreements"),
        # footnotes under the rule: they look like section starts and must not become sections
        (72, 700, "1. Subs. by Act 3 of 1951, s. 3 and Sch., for Part A States."),
        (72, 714, "3. Ins. by Act 4 of 1997, s. 2 (w.e.f. 8-1-1997)."),
    ]
    body2 = [
        (90, 60, "13. Revocation how made.-A proposal is revoked by notice of revocation to the other"),
        (72, 74, "party, or by the lapse of the time prescribed in the proposal for its acceptance."),
        (102, 94, "[CHAPTER IV.-SALES OF GOODS.] Rep. by the Test Sale of Goods Act, 1930 (3 of"),
        (72, 108, "1930), s. 65."),
        (90, 124, "4.[Goods defined.] Rep. by s. 65, ibid."),
        (239, 140, "[DELIVERY.]Rep. by s. 65, ibid."),
        (90, 156, "5. [Delivery how made.] Rep. by s. 65, ibid."),
        (87, 176, "SCHEDULE.-[Enactments repealed.] Rep. by the Repealing Act, 1914."),
    ]
    return [toc, body, body2]


def _india_code_doc(tmp_path, strip=True):
    path = tmp_path / "ic.pdf"
    path.write_bytes(make_pdf(india_code_pages(), rules={1: [(72, 216, 690)]}))
    return extract_pdf(path, CFG, strip_footnotes=strip)


def test_parse_india_code_skips_toc_and_handles_insertions_glued_markers_and_repeals(tmp_path):
    from legalrag.corpus.structure import parse_india_code

    d = _india_code_doc(tmp_path)
    st = parse_india_code(d, "Test Contract Act", max_words=200)
    assert st.problems == []
    by = {u.section_number: u for u in st.units}
    assert list(by) == ["1", "2", "2A", "3", "4", "5"]
    assert by["1"].heading == "Short title" and by["1"].chapter is None
    assert by["2"].heading == "Communication of proposals"                  # "2.Communication": no space
    assert by["2"].chapter == "CHAPTER I - OF PROPOSALS"
    assert by["2A"].heading == "Power to set aside"
    assert "Void agreements" not in by["2A"].text                            # centred sub-heading dropped
    # "13." is footnote marker 1 glued to section 3, and its title wraps before the dash
    assert by["3"].heading == "Revocation how made" and any("'13'" in w for w in st.warnings)
    # repealed chapter heading (and its wrapped line) belongs to no section; stubs keep bracket titles
    assert "SALES OF GOODS" not in by["3"].text and "1930), s. 65." not in by["3"].text
    assert by["4"].chapter == "CHAPTER IV - SALES OF GOODS (repealed)" and by["4"].heading == "Goods defined"
    assert "[DELIVERY.]" not in by["4"].text
    assert by["5"].heading == "Delivery how made" and "Enactments repealed" not in by["5"].text
    assert any("Schedules are not indexed" in w for w in st.warnings)
    for u in st.units:
        assert d.text[u.char_start:u.char_end] == u.text


def test_india_code_footnotes_below_the_rule_are_stripped_only_when_asked(tmp_path):
    kept = _india_code_doc(tmp_path, strip=False)
    assert "Subs. by Act 3 of 1951" in kept.text
    d = _india_code_doc(tmp_path)
    assert "Subs. by Act 3 of 1951" not in d.text and "w.e.f." not in d.text
    assert "contract is voidable." in d.text                                 # body above the rule kept
    assert any("footnote line(s)" in r for r in d.pages[1].removed)


def test_india_code_without_a_body_is_a_problem(tmp_path):
    from legalrag.corpus.structure import parse_india_code

    d = extract_pdf(write_pdf(tmp_path, [india_code_pages()[0]]), CFG)      # ToC only: no "1. Title.-"
    assert parse_india_code(d, "Test Contract Act", max_words=50).problems


def test_manifest_accepts_india_code_numbering():
    e = _entry(numbering="india_code", strip_footnotes=True)
    assert e.numbering == "india_code" and e.strip_footnotes
