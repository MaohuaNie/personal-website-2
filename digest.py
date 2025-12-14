import os
import json
import requests
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import re
from email_utils import send_email  # sends HTML email


JOURNALS = [
    # ——— Your original list ———
    {"name": "Cognition", "issn": "0010-0277"},
    {"name": "JEP: General", "issn": "0096-3445"},
    {"name": "Judgment and Decision Making", "issn": "1930-2975"},
    {"name": "OBHDP", "issn": "0749-5978"},  # Organizational Behavior and Human Decision Processes
    {"name": "Journal of Risk and Uncertainty", "issn": "0895-5646"},
    {"name": "Journal of Behavioral Decision Making", "issn": "0894-3257"},
    {"name": "Cognitive Science", "issn": "0364-0213"},
    {"name": "Memory & Cognition", "issn": "0090-502X"},

    # # ——— From Shelf 1 ———
    {"name": "Trends in Cognitive Sciences", "issn": "1364-6613"},
    {"name": "Cognitive Psychology", "issn": "0010-0285"},
    {"name": "Psychological Review", "issn": "0033-295X"},

    # ——— From Shelf 2 ———
    {"name": "Journal of Experimental Psychology: Learning, Memory, and Cognition", "issn": "0278-7393"},

    # ——— From Shelf 3 (Duplicates removed) ———
    {"name": "Journal of Economic Psychology", "issn": "0167-4870"},

    # ——— Big Review + Methods journals ———
    {"name": "Psychological Bulletin", "issn": "0033-2909"},
    {"name": "Psychological Science", "issn": "0956-7976"},
    {"name": "Psychological Methods", "issn": "1082-989X"},
    {"name": "Journal of Mathematical Psychology", "issn": "0022-2496"},
    {"name": "Current Directions in Psychological Science", "issn": "0963-7214"},
    {"name": "Perspectives on Psychological Science", "issn": "1745-6916"},
    {"name": "Behavior Research Methods", "issn": "1554-351X"},
    {"name": "Psychonomic Bulletin & Review", "issn": "1069-9384"},

    # ——— High-impact applied / interdisciplinary ———
    {"name": "Nature Human Behaviour", "issn": "2397-3374"},
    {"name": "American Economic Review", "issn": "0002-8282"},
    {"name": "Management Science", "issn": "0025-1909"},
    {"name": "The Quarterly Journal of Economics", "issn": "0033-5533"},
]

LOOKBACK_DAYS = 14
TOP_K_PER_JOURNAL = 20

TOPIC_TEXT = """
decision making under risk and uncertainty,
gain–loss domain effects, loss aversion, ambiguity,
risk taking, judgment, adaptive behavior, metacognition,

choice complexity, option complexity, complexity aversion,
decisions from description, decisions from experience,
description–experience differences, epistemic uncertainty,
multi-attribute choice, decision strategies, signal-to-noise ratio, noise,

prospect theory, cumulative prospect theory, probability weighting,
expected utility, higher-order risk preferences, skewness preferences,
heuristics, bounded rationality,

process-level analysis of decisions,
response times, attention and information processing, eye tracking,
speed–accuracy tradeoff,

drift diffusion model, evidence accumulation models, sequential sampling models,
quantitative and computational modeling, Bayesian cognitive modeling,
model-based inference of preferences, integration of choice and RT data,

real-world risky behavior,
financial decision making, gambling behavior, investment decisions,
random utility models
"""

EMAIL_SUBJECT = "Bi-weekly Research Digest: Decision Making"


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

topic_emb = client.embeddings.create(
    model="text-embedding-3-small",
    input=[TOPIC_TEXT]
).data[0].embedding
topic_emb = np.array(topic_emb)


def format_authors(author_list):
    if not author_list:
        return "Unknown"
    out = []
    for a in author_list:
        given = a.get("given", "")
        family = a.get("family", "")
        if given and family:
            out.append(f"{family}, {given[0]}.")
        elif family:
            out.append(family)
    return ", ".join(out)

def format_pub_date(item):
    for key in ["published-print", "published-online", "created"]:
        if key in item:
            d = item[key].get("date-parts", [[]])[0]
            if len(d) >= 2:
                return f"{d[0]}-{str(d[1]).zfill(2)}"
            if len(d) == 1:
                return str(d[0])
    return "n.d."

def parse_pub_datetime(item):
    for key in ["published-print", "published-online", "created"]:
        if key in item:
            d = item[key].get("date-parts", [[]])[0]
            try:
                if len(d) >= 3:
                    return datetime(d[0], d[1], d[2]).date()
                elif len(d) == 2:
                    return datetime(d[0], d[1], 1).date()
                elif len(d) == 1:
                    return datetime(d[0], 1, 1).date()
            except:
                return None
    return None

def clean_abstract_text(text):
    """
    Remove JATS/XML tags and normalize whitespace.
    """
    if not text:
        return ""

    # Remove XML/JATS tags like <jats:p>, <jats:title>, etc.
    text = re.sub(r"<[^>]+>", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def embed_text_batch(texts):
    # Ensure we never send invalid payloads
    if not texts:
        return []

    # Ensure every element is a string (no None)
    clean_texts = []
    for t in texts:
        if t is None:
            t = ""
        if not isinstance(t, str):
            t = str(t)
        clean_texts.append(t)

    # OPTIONAL safety: avoid sending purely empty inputs
    # (Embeddings can accept empty strings sometimes, but it's safer to drop them)
    # If you drop items, you must keep alignment with other arrays — so here we do NOT drop.
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=clean_texts
    )
    return [np.array(x.embedding) for x in resp.data]


def gpt_relevance_and_summary(title, abstract):
    """Return relevance boolean, reason, and a 1–2 sentence summary."""
    prompt = f"""
        You are assisting a PhD student in economic psychology who studies human decision making.
        His research focuses on how people evaluate and choose between options, particularly under risk and uncertainty, and how these processes operate in real-world contexts such as financial decision making and gambling.

        His work draws on behavioral economics, cognitive psychology, and computational modeling, with an emphasis on process-level mechanisms (e.g., response times) and formal models such as drift diffusion and evidence accumulation models.

        Evaluate whether this paper is relevant to that research agenda.

        Treat a paper as RELEVANT if it substantially concerns one or more of the following:
        - decision making under risk or uncertainty;
        - gain–loss domain effects, loss aversion, or ambiguity;
        - choice complexity, description–experience differences, or adaptive strategies;
        - process-level analyses of decision making (e.g., response times, cognitive mechanisms);
        - computational or quantitative models of choice (e.g., DDM, EAM);
        - applications to real-world risky behavior (e.g., finance, gambling).

        Return ONLY this JSON:
        {{
        "relevant": true/false,
        "reason": "1–2 sentences explaining why",
        "summary": "1–2 sentence plain-language summary of the paper"
        }}

        Title: {title}
        Abstract: {abstract}
        """
    resp = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}]
    )
    content = resp.choices[0].message.content

    try:
        j = json.loads(content)
        return (
            bool(j.get("relevant", False)),
            j.get("reason", ""),
            j.get("summary", "")
        )
    except:
        return False, "Parse error", ""


def sanitize_for_embedding(text):
    if not isinstance(text, str):
        return ""
    # Remove invalid unicode / control characters
    return (
        text.encode("utf-8", errors="ignore")
            .decode("utf-8", errors="ignore")
            .strip()
    )

def fetch_semantic_scholar_abstract(title):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "limit": 1,
        "fields": "abstract"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("data") and data["data"][0].get("abstract"):
            return data["data"][0]["abstract"]
    except:
        pass
    return ""

def get_abstract_with_fallback(it):
    abstract = clean_abstract_text(it.get("abstract", ""))
    if abstract:
        return abstract, "crossref"

    title = it.get("title", [""])[0]
    ss_abstract = fetch_semantic_scholar_abstract(title)
    if ss_abstract:
        return clean_abstract_text(ss_abstract), "semantic_scholar" 

    return "", "none"



def fetch_range_for_journal(issn, start_date, end_date):
    if hasattr(start_date, "isoformat"):
        start_date = start_date.isoformat()
    if hasattr(end_date, "isoformat"):
        end_date = end_date.isoformat()

    url = (
        f"https://api.crossref.org/journals/{issn}/works"
        f"?filter=from-pub-date:{start_date},until-pub-date:{end_date}"
        f"&rows=500"
    )
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return [
        it for it in r.json()["message"]["items"]
        if it.get("type") == "journal-article"
        and not it.get("title", [""])[0].lower().startswith("supplement")
    ]

def log(msg):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

# =========================================================
# MAIN LOGIC (ONLY abstract line changed)
# =========================================================

def find_recent_relevant_papers(days=14):
    today = datetime.today().date()
    start_day = today - timedelta(days=days)

    all_results = []

    log(f"Starting digest for last {days} days ({start_day} → {today})")

    for j_idx, j in enumerate(JOURNALS, start=1):
        log(f"[{j_idx}/{len(JOURNALS)}] Fetching {j['name']} ({j['issn']})")

        try:
            items = fetch_range_for_journal(j["issn"], start_day, today)
        except Exception as e:
            log(f"  ✗ Fetch failed: {e}")
            continue

        log(f"  → {len(items)} raw items fetched")

        valid_items = [
            it for it in items
            if parse_pub_datetime(it)
            and start_day <= parse_pub_datetime(it) <= today
        ]

        log(f"  → {len(valid_items)} items in date range")

        if not valid_items:
            log("  → Skipping (no valid items)")
            continue

        texts = []
        abstracts = []
        sources = []

        for it in valid_items:
            abs_text, source = get_abstract_with_fallback(it)
            title = it.get("title", [""])[0] if it.get("title") else ""
            texts.append(
                sanitize_for_embedding(f"{title}\n\n{abs_text or ''}")
            )
            abstracts.append(sanitize_for_embedding(abs_text or ""))
            sources.append(source)

        log(f"  → Creating embeddings for {len(texts)} papers")
        embeds = embed_text_batch(texts)

        if not embeds:
            log("  ✗ Embedding failed / empty")
            continue

        sims = [cosine_sim(e, topic_emb) for e in embeds]

        ranked_idx = np.argsort(sims)[::-1][:TOP_K_PER_JOURNAL]
        log(f"  → Running GPT relevance on top {len(ranked_idx)} papers")

        kept = 0
        for rank, idx in enumerate(ranked_idx, start=1):
            title = valid_items[idx].get("title", [""])[0] if valid_items[idx].get("title") else ""
            relevant, reason, summary = gpt_relevance_and_summary(
                title,
                abstracts[idx]
            )

            if relevant:
                kept += 1
                all_results.append({
                    "title": title,
                    "authors": format_authors(valid_items[idx].get("author", [])),
                    "published": format_pub_date(valid_items[idx]),
                    "journal": j["name"],
                    "relevance_score": round(sims[idx], 2),
                    "doi": valid_items[idx].get("DOI"),
                    "abstract": abstracts[idx],
                    "abstract_source": sources[idx],
                    "summary": summary
                })

        log(f"  ✓ Kept {kept} relevant papers from {j['name']}")

    log(f"Finished. Total relevant papers: {len(all_results)}")

    return sorted(all_results, key=lambda x: x["relevance_score"], reverse=True)

def format_email_body_html(results):
    today = datetime.today().date()
    start_day = today - timedelta(days=LOOKBACK_DAYS)

    # Summary counts by journal (keep original order of appearance)
    journal_counts = {}
    for r in results:
        j = r.get("journal", "Unknown journal")
        journal_counts[j] = journal_counts.get(j, 0) + 1

    def esc(s):
        """Minimal HTML escaping to avoid broken markup."""
        if s is None:
            return ""
        s = str(s)
        return (
            s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;")
        )

    html = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <link rel="stylesheet" href="/assets/site.css">
    </head>
    <body style="
        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
        background:#f6f7fb;
        padding:20px;
        color:#1f2937;
    ">
    <a href="/" style="
    display:inline-block;
    padding:6px 12px;
    border-radius:10px;
    background:#f3f4f6;
    border:1px solid #e5e7eb;
    color:#1f2937;
    font-size:14px;
    text-decoration:none;
    ">
    ← Home
    </a>
      <div style="
          max-width:900px;
          margin:0 auto;
          background:#ffffff;
          border-radius:14px;
          padding:26px;
          box-shadow:0 6px 18px rgba(0,0,0,0.08);
          border:1px solid #eef0f4;
      ">

        <div style="display:flex;align-items:flex-end;justify-content:space-between;gap:14px;flex-wrap:wrap;">
          <div>
            <h1 style="margin:0;font-size:26px;line-height:1.2;color:#111827;">
              Bi-weekly Research Digest
            </h1>
            <p style="margin:8px 0 0;font-size:14px;color:#4b5563;">
              <b>Date range:</b> {start_day} → {today}<br>
              <b>Total relevant papers:</b> {len(results)}
            </p>
          </div>
          <div style="
              padding:10px 12px;
              background:#f3f4f6;
              border:1px solid #e5e7eb;
              border-radius:12px;
              font-size:12px;
              color:#374151;
          ">
            Topic: Decision Making
          </div>
        </div>

        <h2 style="
            margin:26px 0 10px;
            font-size:18px;
            border-bottom:1px solid #e5e7eb;
            padding-bottom:8px;
            color:#111827;
        ">
          Summary by Journal
        </h2>

        <ul style="margin:10px 0 0;padding-left:18px;color:#374151;line-height:1.7;">
    """

    for j, c in journal_counts.items():
        html += f"<li><b>{esc(j)}</b>: {c}</li>"

    html += "</ul>"

    # Per-journal sections (with card layout)
    for journal in journal_counts.keys():
        html += f"""
        <h2 style="
            margin:34px 0 10px;
            font-size:24px;
            color:#b91c1c;
        ">
          {esc(journal)}
        </h2>
        """

        for r in results:
            if r.get("journal") != journal:
                continue

            title = esc(r.get("title", "Untitled"))
            summary = esc(r.get("summary", "")).strip()
            authors = esc(r.get("authors", "Unknown"))
            published = esc(r.get("published", "n.d."))
            score = esc(r.get("relevance_score", ""))
            doi = r.get("doi") or ""
            doi_url = f"https://doi.org/{doi}" if doi else ""
            abstract = (r.get("abstract") or "").strip()
            abstract_source = esc(r.get("abstract_source", ""))

            # Abstract display rule (exactly what you asked)
            abstract_html = esc(abstract) if abstract else "Not available."

            # Optional badge: show where abstract came from if you want it visible
            badge = ""
            if abstract_source:
                badge = f"""
                <span style="
                    display:inline-block;
                    margin-left:8px;
                    padding:2px 8px;
                    font-size:12px;
                    border-radius:999px;
                    background:#f3f4f6;
                    border:1px solid #e5e7eb;
                    color:#4b5563;
                ">
                  abstract: {abstract_source}
                </span>
                """

            html += f"""
            <div style="
                background:#ffffff;
                border:1px solid #e5e7eb;
                border-radius:14px;
                padding:18px;
                margin:14px 0 18px;
                box-shadow:0 2px 10px rgba(0,0,0,0.04);
            ">
              <h3 style="margin:0 0 10px;font-size:20px;color:#111827;">
                {title}
              </h3>

              {"<p style='margin:0 0 12px;font-size:14px;color:#374151;line-height:1.6;'>" + summary + "</p>" if summary else ""}

              <div style="font-size:13px;color:#4b5563;line-height:1.7;">
                <div><b>Authors:</b> {authors}</div>
                <div><b>Published:</b> {published}</div>
                <div><b>Relevance Score:</b> {score}</div>
                <div style="margin-top:10px;">
                  <b>DOI:</b>
                  {"<a href='" + esc(doi_url) + "' style='color:#2563eb;text-decoration:none;'>" + esc(doi_url) + "</a>" if doi_url else "Not available."}
                  
                </div>

                <div style="
                    margin-top:12px;
                    padding:12px;
                    background:#f9fafb;
                    border:1px solid #eef0f4;
                    border-radius:12px;
                ">
                  <b>Abstract:</b><br>
                  <span style="color:#374151;">{abstract_html}</span>
                </div>
              </div>
            </div>
            """

    html += """
      </div>
    </body>
    </html>
    """

    return html

# =========================================================
# MAIN
# =========================================================

def save_digest_html(html):
    os.makedirs("research-digest", exist_ok=True)

    today = datetime.today().date().isoformat()
    filename = f"{today}.html"
    path = os.path.join("research-digest", filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    return filename

def main():
    results = find_recent_relevant_papers(LOOKBACK_DAYS)

    html = format_email_body_html(results)



    # 2. Save for website
    save_digest_html(html)

if __name__ == "__main__":
    main()
