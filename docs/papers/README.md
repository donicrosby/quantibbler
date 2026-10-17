# Reference papers

These are arXiv-derived reference copies used as the citation basis for the
specs under `openspec/`. Each markdown file records its source URL and the
sha256 of the fetched HTML in its frontmatter, and is regenerated with:

```
python3 tools/arxiv2md.py <fetched.html> <out.md> <arxiv-html-url>
```

Links to third-party code hosts are removed during conversion; cite the
papers, not repositories. Keys used across the specs:

- `[LB]` — littlebit-2506.13771v5 (LittleBit, NeurIPS 2025)
- `[LB2]` — littlebit2-2603.00042v2 (LittleBit-2, ICML 2026)
- `[NQ]` — nanoquant-2602.06694v3 (NanoQuant, ICML 2026)
