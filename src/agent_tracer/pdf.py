"""Render the markdown report to PDF via Markdown → HTML → weasyprint.

Both ``markdown`` and ``weasyprint`` are pulled in by the ``[pdf]`` extra.
This module imports them lazily so the rest of the package stays usable
without them.

The CSS targets a printable A4 page: a serif-ish body, monospace tables
and code, and SVG images sized to the page width.
"""

from __future__ import annotations

from pathlib import Path

_CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica,
               Arial, sans-serif;
  font-size: 10pt;
  color: #1f2937;
  line-height: 1.45;
}
h1 { font-size: 20pt; border-bottom: 2px solid #1f2937; padding-bottom: 4pt; }
h2 { font-size: 13pt; margin-top: 18pt; border-bottom: 1px solid #94a3b8;
     padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 11pt; margin-top: 12pt; page-break-after: avoid; }
p, li { font-size: 9.5pt; }
code, pre { font-family: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
            font-size: 8.5pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 12pt 0;
        page-break-inside: auto; font-size: 8.5pt; }
th, td { border: 1px solid #cbd5e1; padding: 3pt 6pt; text-align: left;
         vertical-align: top; }
th { background: #f1f5f9; font-weight: 600; }
td:nth-child(n+2):not([data-text]) { text-align: right; }
tr { page-break-inside: avoid; }
img { max-width: 100%; height: auto; display: block; margin: 6pt auto;
      page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #cbd5e1; margin: 12pt 0; }
"""


def render(markdown_path: Path, pdf_path: Path) -> None:
    """Convert ``markdown_path`` (UTF-8 .md) to ``pdf_path``.

    Images referenced with relative paths in the markdown resolve
    relative to ``markdown_path``'s directory.
    """
    import markdown as md_lib
    from weasyprint import CSS, HTML

    text = markdown_path.read_text(encoding="utf-8")
    html_body = md_lib.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html",
    )
    full_html = (
        "<!doctype html><html><head><meta charset='utf-8'></head>"
        f"<body>{html_body}</body></html>"
    )
    HTML(string=full_html, base_url=str(markdown_path.parent)).write_pdf(
        target=str(pdf_path), stylesheets=[CSS(string=_CSS)]
    )
