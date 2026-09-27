"""Build tiny valid PDFs with text at exact coordinates, so extraction/parsing tests are deterministic
and never depend on a downloaded (and possibly copyrighted) document."""

from __future__ import annotations

Item = tuple[float, float, str]  # (x from left, y from top, text)


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


Rule = tuple[float, float, float]  # a horizontal line: (x0, x1, y from top)


def make_pdf(pages: list[list[Item]], *, size: tuple[int, int] = (595, 842), font_size: int = 10,
             rules: dict[int, list[Rule]] | None = None) -> bytes:
    """One content stream per page; Helvetica; every item placed with its own Td.
    `rules`: page index -> horizontal lines to draw (e.g. an India Code footnote separator)."""
    w, h = size
    objs: list[bytes] = []

    def add(body: str | bytes) -> int:
        objs.append(body.encode("latin-1") if isinstance(body, str) else body)
        return len(objs)

    add("<< /Type /Catalog /Pages 2 0 R >>")                     # 1
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages)))
    add(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")  # 2
    add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")  # 3
    for i, items in enumerate(pages):
        page_no, content_no = 4 + 2 * i, 5 + 2 * i
        add(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] /Contents {content_no} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>")
        ops = ["BT", f"/F1 {font_size} Tf"]
        px, py = 0.0, 0.0
        for x, y_top, text in items:
            y = h - y_top - font_size
            ops.append(f"{x - px:.2f} {y - py:.2f} Td ({_esc(text)}) Tj")
            px, py = x, y
        ops.append("ET")
        for x0, x1, y_top in (rules or {}).get(i, []):
            ops.append(f"0.5 w {x0:.2f} {h - y_top:.2f} m {x1:.2f} {h - y_top:.2f} l S")
        stream = "\n".join(ops)
        add(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def lines_page(lines: list[str], *, x: float = 118, top: float = 90, step: float = 14,
               header: str | None = None, page_number: int | None = None) -> list[Item]:
    """A page of body lines, optionally with a running header (the same text on every page)."""
    items: list[Item] = []
    if header is not None:
        items.append((x, 40, header if page_number is None else f"{header} {page_number}"))
    for i, ln in enumerate(lines):
        items.append((x, top + i * step, ln))
    return items
