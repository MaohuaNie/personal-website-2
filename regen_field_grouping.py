"""One-off maintenance: re-render existing digest HTML pages with the new
field-grouped layout (Economics / Psychology / Neuroscience).

For each dated page it:
  1. parses the paper records already baked into the HTML,
  2. asks the LLM to assign each paper a field (same taxonomy the live
     pipeline now uses),
  3. re-renders via paper_search.format_email_body_html,
  4. writes a .bak backup and overwrites the page.

Requires ANTHROPIC_API_KEY (and, because paper_search.init_clients validates
all keys up front, OPENAI_API_KEY + ELSEVIER_API_KEY too). Run from the repo
root, e.g.:

    python3 regen_field_grouping.py                # last 10 dated pages
    python3 regen_field_grouping.py 2026-05-15 ...  # specific pages
"""
import os
import re
import sys
import html
import json
import datetime

import paper_search as ps

DIGEST_DIR = ps.DIGEST_DIR


def parse_page(src):
    """Extract paper records from a generated digest page.

    Splits on each `.paper-item` div regardless of attribute order (older pages
    emit `class="paper-item"` before `id`/`data-journal`; newer ones after).
    """
    # Split before each <div> whose class attribute contains "paper-item"
    # (handles single "paper-item", doubled "paper-item paper-item", and the
    # new "paper-item paper", in any attribute order).
    chunks = re.split(r'(?=<div\b[^>]*class="[^"]*paper-item)', src)
    chunks = [c for c in chunks if re.match(r'<div\b[^>]*class="[^"]*paper-item', c)]
    papers = []
    for ch in chunks:
        # data-journal lives in the opening tag; grab the first occurrence.
        dj = re.search(r'data-journal="([^"]*)"', ch)
        journal = html.unescape(dj.group(1)) if dj else ""
        title = re.search(r'<h3 style="margin:0 0 10px[^>]*>(.*?)</h3>', ch, re.S)
        summary = re.search(r"<p style='margin:0 0 12px;font-size:14px;color:#374151;line-height:1.6;'>(.*?)</p>", ch, re.S)
        authors = re.search(r'<b>Authors:</b>\s*(.*?)</div>', ch, re.S)
        published = re.search(r'<b>Published:</b>\s*(.*?)</div>', ch, re.S)
        score = re.search(r'<b>Relevance Score:</b>\s*(.*?)</div>', ch, re.S)
        doi = re.search(r"href='https://doi\.org/(.*?)'", ch)
        abstract = re.search(r'<b>Abstract:</b><br><span[^>]*>(.*?)</span>', ch, re.S)

        def u(m):
            return html.unescape(m.group(1).strip()) if m else ""

        rec = {
            "title": u(title),
            "summary": u(summary),
            "authors": u(authors),
            "published": u(published),
            "relevance_score": u(score),
            "doi": u(doi),
            "abstract": u(abstract),
            "journal": journal,
        }
        papers.append(rec)
    return papers


def classify_field(title, abstract):
    """Assign one field (Economics/Psychology/Neuroscience) to an existing paper."""
    prompt = f"""Classify this paper into ONE field based on its own content,
methods, and framing — NOT the journal it appeared in.

- "Economics": behavioral / experimental economics, risk & uncertainty in an
  economic tradition, markets, incentives, preference elicitation, finance,
  game-theoretic behavior.
- "Psychology": cognitive / experimental psychology, judgment and decision
  making, cognitive modeling, attention, memory, perception, response times.
- "Neuroscience": neural mechanisms, brain imaging / recording, computational
  neuroscience, neural correlates of value, choice, or attention.

Return ONLY JSON: {{"field": "Economics" | "Psychology" | "Neuroscience"}}

Title: {title}
Abstract: {abstract}"""
    resp = ps.anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=60,
        messages=[{"role": "user", "content": prompt}],
    )
    content = resp.content[0].text.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return ps.normalize_field(json.loads(content).get("field", ""))
    except Exception:
        return ps.DEFAULT_FIELD


def report_date_from_name(path):
    return datetime.date.fromisoformat(os.path.basename(path)[:-5])


def regen(path):
    src = open(path, encoding="utf-8").read()
    papers = parse_page(src)
    report_date = report_date_from_name(path)
    start_day, end_day = ps.get_interval_from_report_date(report_date)

    for i, r in enumerate(papers, 1):
        r["field"] = classify_field(r["title"], r["abstract"])
        ps.log(f"    [{i}/{len(papers)}] {r['field']:<12} {r['title'][:60]}")

    papers.sort(key=lambda r: float(r["relevance_score"] or 0), reverse=True)
    new_html = ps.format_email_body_html(papers, start_day, end_day)

    open(path + ".bak", "w", encoding="utf-8").write(src)   # backup once
    open(path, "w", encoding="utf-8").write(new_html)
    ps.log(f"  regenerated {os.path.basename(path)} ({len(papers)} papers)")


def main():
    ps.init_clients()
    if len(sys.argv) > 1:
        names = [a if a.endswith(".html") else f"{a}.html" for a in sys.argv[1:]]
        paths = [str(DIGEST_DIR / n) for n in names]
    else:
        pages = sorted(DIGEST_DIR.glob("20*.html"), key=lambda p: p.name, reverse=True)
        paths = [str(p) for p in pages[:10]]

    for path in paths:
        ps.log(f"Regenerating {os.path.basename(path)} ...")
        regen(path)
    ps.log("Done. Backups written as <page>.html.bak")


if __name__ == "__main__":
    main()
