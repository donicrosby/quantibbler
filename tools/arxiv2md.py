#!/usr/bin/env python3
"""Convert arXiv LaTeXML HTML to provenance-stamped markdown.

Stdlib-only. Preserves math by substituting each <math> element with its
LaTeX `alttext` inline ($...$), so spec authors can cite formulas verbatim.

Usage: arxiv2md.py <input.html> <output.md> <source-url>
"""

import hashlib
import re
import sys
from html.parser import HTMLParser


class LatexmlToMd(HTMLParser):
    SKIP = {"script", "style", "head", "svg", "annotation", "semantics"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.skip_depth = 0
        self.math_depth = 0
        self.math_alt = None
        self.list_stack = []

    # -- tag handling ----------------------------------------------------
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.SKIP:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "math":
            self.math_depth += 1
            if self.math_depth == 1:
                self.math_alt = a.get("alttext", "")
            return
        if self.math_depth:
            return  # ignore MathML internals; we keep alttext only
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "p":
            self.out.append("\n\n")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag == "tr":
            self.out.append("\n| ")
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag == "figcaption":
            self.out.append("\n\n*Figure: ")
        elif tag == "br":
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth:
            return
        if tag == "math":
            self.math_depth = max(0, self.math_depth - 1)
            if self.math_depth == 0:
                alt = (self.math_alt or "").strip()
                self.out.append(f" ${alt}$ " if alt else " ")
                self.math_alt = None
            return
        if self.math_depth:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append("\n")
        elif tag == "figcaption":
            self.out.append("*")

    def handle_data(self, data):
        if self.skip_depth or self.math_depth:
            return
        self.out.append(data)

    def text(self):
        raw = "".join(self.out)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip() + "\n"


# Domains whose links are REDACTED from clean-side copies: code/model hosts.
# Papers are the only permitted evidence; links into repos would invite
# implementers to read encumbered source. See cleanroom/papers/README.md.
REDACT_DOMAINS = (
    "github.com",
    "githubusercontent.com",
    "github.io",
    "gitlab.com",
    "huggingface.co",
    "hf.co",
    "bitbucket.org",
    "gitee.com",
    "sourceforge.net",
)

_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?[a-z0-9.-]+/[^\s\)\]\}<>\"']+", re.IGNORECASE)


def redact_code_links(text: str) -> tuple[str, int]:
    """Replace links to code/model hosts with a redaction marker."""
    redacted = 0

    def repl(m: re.Match) -> str:
        nonlocal redacted
        if any(d in m.group(0).lower() for d in REDACT_DOMAINS):
            redacted += 1
            return "[redacted code link]"
        return m.group(0)

    # Repeat: redactions can unmask adjacent matches hiding behind earlier ones.
    while True:
        before = text
        text = _URL_RE.sub(repl, text)
        if text == before:
            break
    return text, redacted


def main():
    src, dst, url = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(src, "rb") as f:
        blob = f.read()
    sha = hashlib.sha256(blob).hexdigest()
    p = LatexmlToMd()
    p.feed(blob.decode("utf-8", errors="replace"))
    body, redacted = redact_code_links(p.text())
    header = (
        f"---\n"
        f"source: {url}\n"
        f"fetched: 2026-10-17\n"
        f"source_sha256: {sha}\n"
        f"converter: tools/arxiv2md.py (stdlib; math kept as LaTeX alttext)\n"
        f"note: Reference copy for spec citation. Links to code hosts are "
        f"removed; copyright stays with the authors (arXiv distribution terms).\n"
        f"---\n\n"
    )
    with open(dst, "w") as f:
        f.write(header + body)
    print(f"{dst}: {len(body)} chars, redacted_links={redacted}, sha256={sha[:16]}...")


if __name__ == "__main__":
    main()
