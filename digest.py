import os
import json
import requests
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
import re
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).parent
DIGEST_DIR = ROOT / "research-digest"
DIGEST_DIR.mkdir(exist_ok=True)

DIGEST_INDEX = DIGEST_DIR / "digests.json"
ELSEVIER_CACHE = {}

JOURNALS = [

    {"name": "Cognition", "issn": "0010-0277"},
    {"name": "JEP: General", "issn": "0096-3445"},
    {"name": "Judgment and Decision Making", "issn": "1930-2975"},
    {"name": "OBHDP", "issn": "0749-5978"},  # Organizational Behavior and Human Decision Processes
    {"name": "Journal of Risk and Uncertainty", "issn": "0895-5646"},
    {"name": "Journal of Behavioral Decision Making", "issn": "0894-3257"},
    {"name": "Cognitive Science", "issn": "0364-0213"},
    {"name": "Memory & Cognition", "issn": "0090-502X"},


    {"name": "Trends in Cognitive Sciences", "issn": "1364-6613"},
    {"name": "Cognitive Psychology", "issn": "0010-0285"},
    {"name": "Psychological Review", "issn": "0033-295X"},


    {"name": "Journal of Experimental Psychology: Learning, Memory, and Cognition", "issn": "0278-7393"},


    {"name": "Journal of Economic Psychology", "issn": "0167-4870"},

    # # ——— Big Review + Methods journals ———
    {"name": "Psychological Bulletin", "issn": "0033-2909"},
    {"name": "Psychological Science", "issn": "0956-7976"},
    {"name": "Psychological Methods", "issn": "1082-989X"},
    {"name": "Journal of Mathematical Psychology", "issn": "0022-2496"},
    {"name": "Current Directions in Psychological Science", "issn": "0963-7214"},
    {"name": "Perspectives on Psychological Science", "issn": "1745-6916"},
    {"name": "Behavior Research Methods", "issn": "1554-351X"},
    {"name": "Psychonomic Bulletin & Review", "issn": "1069-9384"},

    # # ——— High-impact applied / interdisciplinary ———
    {"name": "Nature Human Behaviour", "issn": "2397-3374"},
    {"name": "American Economic Review", "issn": "0002-8282"},
    {"name": "Management Science", "issn": "0025-1909"},
    {"name": "The Quarterly Journal of Economics", "issn": "0033-5533"},
]

ELSEVIER_ISSNS = {
    "0010-0277",  # Cognition
    "0749-5978",  # OBHDP
    "0010-0285",  # Cognitive Psychology
    "0167-4870",  # Journal of Economic Psychology
    "0022-2496",  # Journal of Mathematical Psychology
}

ELSEVIER_API_KEY = os.getenv("ELSEVIER_API_KEY")
if not ELSEVIER_API_KEY:
    raise RuntimeError("Missing ELSEVIER_API_KEY environment variable")


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

def fetch_elsevier_metadata(doi):
    """
    Fetch abstract + authors + publication date from Elsevier (Scopus).
    Returns dict or None.
    """
    if doi in ELSEVIER_CACHE:
        return ELSEVIER_CACHE[doi]
    

    encoded_doi = quote(doi, safe="")
    url = f"https://api.elsevier.com/content/abstract/doi/{encoded_doi}?view=FULL"
    headers = {
        "X-ELS-APIKey": ELSEVIER_API_KEY,
        "Accept": "application/json",
        "User-Agent": "research-digest/1.0 (mailto:maohua.nie@unibas.ch)"
    }

    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            log(f"Elsevier failed ({r.status_code}) for DOI {doi}")
            return None
        data = r.json()
    except Exception as e:
        log(f"Elsevier request exception for DOI {doi}: {e}")
        return None

    # ---- extract abstract (schema-safe) ----
    abstract = None
    head = (
        data.get("abstracts-retrieval-response", {})
            .get("item", {})
            .get("bibrecord", {})
            .get("head", {})
    )
    abstracts = head.get("abstracts")

    if isinstance(abstracts, str):
        abstract = abstracts

    elif isinstance(abstracts, dict):
        a = abstracts.get("abstract")

        if isinstance(a, str):
            abstract = a

        elif isinstance(a, dict):
            # Case 1: direct para
            para = a.get("ce:para")
            if isinstance(para, list):
                abstract = " ".join(para)
            elif isinstance(para, str):
                abstract = para

            # Case 2: sectioned abstract (VERY common in Cognition)
            sections = a.get("ce:sections", {}).get("ce:section")
            if sections:
                if isinstance(sections, dict):
                    sections = [sections]
                paras = []
                for sec in sections:
                    p = sec.get("ce:para")
                    if isinstance(p, list):
                        paras.extend(p)
                    elif isinstance(p, str):
                        paras.append(p)
                if paras:
                    abstract = " ".join(p.strip() for p in paras)

    # ---- extract authors ----
    authors = []
    ag = head.get("author-group", None)

    # author-group may be dict OR list
    if isinstance(ag, dict):
        ag_list = [ag]
    elif isinstance(ag, list):
        ag_list = ag
    else:
        ag_list = []

    for group in ag_list:
        auths = group.get("author", [])
        if isinstance(auths, dict):
            auths = [auths]
        for a in auths:
            authors.append({
                "given": a.get("ce:given-name", ""),
                "family": a.get("ce:surname", ""),
            })


    # ---- extract publication year ----
    pub_year = None
    pubdate = (
        data.get("abstracts-retrieval-response", {})
            .get("item", {})
            .get("bibrecord", {})
            .get("item-info", {})
            .get("history", {})
            .get("publication-date", {})
    )
    if isinstance(pubdate, dict):
        pub_year = pubdate.get("year")
    
    result = {
            "abstract": clean_abstract_text(abstract or ""),
            "authors": authors,
            "year": pub_year,
            "source": "elsevier",
    }

    ELSEVIER_CACHE[doi] = result

    return result


TOP_K_PER_JOURNAL = 30

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






OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")

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




def gpt_relevance_and_summary(title, abstract):
    """Return relevance boolean, reason, and a 1–2 sentence summary."""
    prompt = f"""
        You are assisting a PhD student in economic psychology who studies **individual-level decision making**.

        His research focuses on how **people mentally represent, evaluate, and compare choice options**, particularly under **risk, uncertainty, and complexity**, and how these processes generate observable behavior such as **choices, response times, and inconsistencies**.

        His work draws on behavioral economics, cognitive psychology, and computational modeling, with a strong emphasis on:
        - cognitive mechanisms,
        - internal noise or uncertainty in representations,
        - process-level evidence (e.g., response times, memory, attention),
        - and formal models of decision making (e.g., Bayesian models, drift diffusion / evidence accumulation models).

        Your task is to evaluate whether the paper is relevant to **this cognitive and process-oriented research agenda**.



        ### Treat a paper as RELEVANT if it substantially concerns one or more of the following:

        - **Individual decision making under risk or uncertainty**, where behavior is explained in terms of cognitive representations, preferences, or noise.
        - **Choice complexity**, cognitive load, or informational complexity, and how these affect individual choices or decision processes.
        - **Gain–loss asymmetries**, loss aversion, ambiguity, probability weighting, or related preference phenomena at the individual level.
        - **Process-level evidence**, such as response times, attention, memory, learning dynamics, or choice inconsistency.
        - **Computational or formal cognitive models** of choice (e.g., Bayesian inference models, drift diffusion models, evidence accumulation, stochastic choice models with psychological interpretation).
        - Experimental or empirical work that **tests psychological mechanisms**, even if conducted in applied domains (e.g., finance, portfolios, markets).



        ### Treat a paper as NOT RELEVANT if it primarily focuses on:

        - Market-level, firm-level, or population-level outcomes **without modeling individual decision processes**.
        - Purely normative optimization, algorithmic methods, or policy design **without psychological or cognitive interpretation**.
        - Large-scale behavioral patterns inferred from field data **without a cognitive or process-level account**.
        - Pricing, supply chains, operations, contracts, or strategic interactions **unless individual choice under risk or uncertainty is the central object of study**.

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

def get_abstract_with_fallback(it, issn):
    title = it.get("title", [""])[0]
    doi = it.get("DOI")

    # 1) Elsevier FIRST
    if doi and issn in ELSEVIER_ISSNS:
        meta = fetch_elsevier_metadata(doi)
        if meta and meta.get("abstract"):
            log(f"Abstract source: Elsevier — {title[:60]}")
            return meta["abstract"], "elsevier", meta

    # 2) Crossref
    crossref_abs = clean_abstract_text(it.get("abstract", ""))
    if crossref_abs:
        log(f"Abstract source: Crossref — {title[:60]}")
        return crossref_abs, "crossref", None

    # 3) Semantic Scholar
    ss_abs = clean_abstract_text(fetch_semantic_scholar_abstract(title))
    if ss_abs:
        log(f"Abstract source: Semantic Scholar — {title[:60]}")
        return ss_abs, "semantic_scholar", None

    log(f"No abstract found — {title[:60]}")
    return "", "none", None



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

def find_recent_relevant_papers():
    today = datetime.today().date()
    run_mode = os.getenv("DIGEST_RUN_MODE", "scheduled")

    # Decide date window
    if today.day == 15:
        # 1st → 15th of current month
        start_day = today.replace(day=1)
        end_day = today

    elif today.day == 1:
        # 16th → end of previous month
        last_month_end = today - timedelta(days=1)
        start_day = last_month_end.replace(day=16)
        end_day = last_month_end

    elif run_mode == "manual":
        # Manual / test run: last 14 days
        start_day = today - timedelta(days=14)
        end_day = today

    else:
        log("Not 1st or 15th — skipping digest generation.")
        return [], None, None

    log(f"Run mode: {run_mode}")
    log(f"Digest window: {start_day} → {end_day}")

    all_results = []

    for j_idx, j in enumerate(JOURNALS, start=1):
        log(f"[{j_idx}/{len(JOURNALS)}] Fetching {j['name']} ({j['issn']})")

        try:
            items = fetch_range_for_journal(j["issn"], start_day, end_day)
        except Exception as e:
            log(f"  ✗ Fetch failed: {e}")
            continue

        log(f"  → {len(items)} raw items fetched")

        valid_items = [
            it for it in items
            if parse_pub_datetime(it)
            and start_day <= parse_pub_datetime(it) <= end_day
        ]

        log(f"  → {len(valid_items)} items in date range")

        if not valid_items:
            log("  → Skipping (no valid items)")
            continue

        texts = []
        abstracts = []
        sources = []
        metas = []

        for it in valid_items:
            abs_text, source, elsevier_meta = get_abstract_with_fallback(it, j["issn"])
            title = it.get("title", [""])[0] if it.get("title") else ""

            texts.append(
                sanitize_for_embedding(f"{title}\n\n{abs_text or ''}")
            )
            abstracts.append(sanitize_for_embedding(abs_text or ""))
            sources.append(source)
            metas.append(elsevier_meta)   # NEW

        log(f"  → Creating embeddings for {len(texts)} papers")
        embeds = embed_text_batch(texts)

        if not embeds:
            log("  ✗ Embedding failed / empty")
            continue

        sims = [cosine_sim(e, topic_emb) for e in embeds]
        ranked_idx = np.argsort(sims)[::-1][:TOP_K_PER_JOURNAL]

        log(f"  → Running GPT relevance on top {len(ranked_idx)} papers")

        kept = 0
        for idx in ranked_idx:
            title = valid_items[idx].get("title", [""])[0]
            relevant, reason, summary = gpt_relevance_and_summary(
                title,
                abstracts[idx]
            )

            if relevant:
                kept += 1
                authors = format_authors(valid_items[idx].get("author", []))
                published = format_pub_date(valid_items[idx])

                meta = metas[idx]
                if meta:
                    if meta.get("authors"):
                        authors = format_authors(meta["authors"])
                    if meta.get("year"):
                        published = str(meta["year"])

                all_results.append({
                    "title": title,
                    "authors": authors,
                    "published": published,
                    "journal": j["name"],
                    "relevance_score": round(sims[idx], 2),
                    "doi": valid_items[idx].get("DOI"),
                    "abstract": abstracts[idx],
                    "abstract_source": sources[idx],
                    "metadata_source": sources[idx] if sources[idx] == "elsevier" else "crossref",
                    "summary": summary
                })

        log(f"  ✓ Kept {kept} relevant papers from {j['name']}")

    log(f"Finished. Total relevant papers: {len(all_results)}")

    return sorted(all_results, key=lambda x: x["relevance_score"], reverse=True), start_day, end_day

def format_email_body_html(results, start_day, end_day):

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
              <b>Date range:</b> {start_day} → {end_day}<br>
              <b>Total relevant papers:</b> {len(results)}
            </p>
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
                {title} {badge}
              </h3>

              {"<p style='margin:0 0 12px;font-size:14px;color:#374151;line-height:1.6;'>" + summary + "</p>" if summary else ""}

              <div style="font-size:13px;color:#4b5563;line-height:1.7;">
                <div><b>Authors:</b> {authors}</div>
                <div><b>Published:</b> {published}</div>
                <div><b>Relevance Score:</b> {score}</div>
                <div style="margin-top:10px;">
                  <b>DOI:</b>
                  {"<a href='" + esc(doi_url) + "' target='_blank' rel='noopener noreferrer' "
                    "style='color:#2563eb;text-decoration:none;'>" + esc(doi_url) + "</a>"
                    if doi_url else "Not available."}
                  
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
    today = datetime.today().date().isoformat()
    filename = f"{today}.html"
    path = DIGEST_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    return filename, today


def update_digest_index(date_str, filename, results, start_day, end_day):
    entry = {
        "date": date_str,
        "title": f"Research Digest · {start_day} → {end_day}",
        "papers": len(results),
        "file": filename
    }

    index = []

    if DIGEST_INDEX.exists():
        try:
            with open(DIGEST_INDEX, "r", encoding="utf-8") as f:
                index = json.load(f)
        except json.JSONDecodeError:
            # File exists but is empty or corrupted
            print("Warning: digests.json was empty or invalid. Reinitializing.")

    # Avoid duplicate entries
    if any(d["date"] == date_str for d in index):
        print("Digest already exists in index — skipping index update.")
        return

    index.append(entry)

    with open(DIGEST_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

def main():
    results, start_day, end_day = find_recent_relevant_papers()

    if not results:
        print("No digest generated (outside scheduled window).")
        return

    html = format_email_body_html(results, start_day, end_day)

    filename, date_str = save_digest_html(html)
    update_digest_index(date_str, filename, results, start_day, end_day)

    print(f"Saved digest: research-digest/{filename}")




if __name__ == "__main__":
    main()
