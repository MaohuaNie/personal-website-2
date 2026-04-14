import os
import json
import requests
import numpy as np
from datetime import datetime, timedelta, date
from openai import OpenAI
from anthropic import Anthropic
import re
from pathlib import Path
from urllib.parse import quote

# =========================================================
# CONFIG & CONSTANTS    
# =========================================================

ROOT = Path(__file__).parent
DIGEST_DIR = ROOT / "research-digest"
DIGEST_DIR.mkdir(exist_ok=True)
LOG_DIR = ROOT / "research-digest" / "logs"
LOG_DIR.mkdir(exist_ok=True)

DIGEST_INDEX = DIGEST_DIR / "digests.json"
FEED_PATH = DIGEST_DIR / "feed.xml"
FEED_MAX_ITEMS = 20
SITE_URL = "https://maohuanie.com"

# MailerLite subscriber broadcast
MAILERLITE_GROUP_ID = "184760257614972692"  # "Decision Science Digest" group
MAILERLITE_FROM_EMAIL = "niemaohua@gmail.com"  # must be a verified sender in MailerLite
MAILERLITE_FROM_NAME = "Maohua Nie"

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
    {"name": "Psychological Bulletin", "issn": "0033-2909"},
    {"name": "Psychological Science", "issn": "0956-7976"},
    {"name": "Psychological Methods", "issn": "1082-989X"},
    {"name": "Journal of Mathematical Psychology", "issn": "0022-2496"},
    {"name": "Current Directions in Psychological Science", "issn": "0963-7214"},
    {"name": "Perspectives on Psychological Science", "issn": "1745-6916"},
    {"name": "Behavior Research Methods", "issn": "1554-351X"},
    {"name": "Psychonomic Bulletin & Review", "issn": "1069-9384"},
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

TOP_K_PER_JOURNAL = 30
SIM_THRESHOLD = 0.30

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

# =========================================================
# ENV CHECKS
# =========================================================

ELSEVIER_API_KEY = os.getenv("ELSEVIER_API_KEY")
if not ELSEVIER_API_KEY:
    raise RuntimeError("Missing ELSEVIER_API_KEY environment variable")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("Missing ANTHROPIC_API_KEY environment variable")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Pre-compute topic embedding
topic_emb = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=[TOPIC_TEXT]
).data[0].embedding
topic_emb = np.array(topic_emb)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def log(msg):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def clean_abstract_text(text):
    """Remove JATS/XML tags and normalize whitespace."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_correction_item(it):
    """Return True if the item is a correction / erratum / retraction."""
    bad_types = {
        "correction", "erratum", "retraction", "retracted-article",
        "expression-of-concern", "addendum"
    }
    if it.get("type") in bad_types:
        return True

    relation = it.get("relation", {})
    if isinstance(relation, dict):
        for k in relation.keys():
            if any(x in k.lower() for x in ["correction", "erratum", "retraction", "update", "expression"]):
                return True

    title = it.get("title", [""])[0].lower() if it.get("title") else ""
    correction_markers = [
        "correction to", "publisher correction", "author correction",
        "erratum", "corrigendum", "retraction", "expression of concern", "addendum"
    ]
    if any(marker in title for marker in correction_markers):
        return True

    return False

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def embed_text_batch(texts):
    if not texts:
        return []
    clean_texts = []
    for t in texts:
        clean_texts.append(str(t) if t is not None else "")
    
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=clean_texts
    )
    return [np.array(x.embedding) for x in resp.data]

def sanitize_for_embedding(text):
    if not isinstance(text, str):
        return ""
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore").strip()

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

# =========================================================
# FETCHING & PROCESSING
# =========================================================

def fetch_elsevier_metadata(doi):
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
            return None
        data = r.json()
    except Exception:
        return None

    # Extract Abstract
    abstract = None
    head = data.get("abstracts-retrieval-response", {}).get("item", {}).get("bibrecord", {}).get("head", {})
    abstracts = head.get("abstracts")

    if isinstance(abstracts, str):
        abstract = abstracts
    elif isinstance(abstracts, dict):
        a = abstracts.get("abstract")
        if isinstance(a, str):
            abstract = a
        elif isinstance(a, dict):
            para = a.get("ce:para")
            if isinstance(para, list):
                abstract = " ".join(para)
            elif isinstance(para, str):
                abstract = para
            
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

    # Extract Authors
    authors = []
    ag = head.get("author-group", None)
    if isinstance(ag, dict): ag_list = [ag]
    elif isinstance(ag, list): ag_list = ag
    else: ag_list = []

    for group in ag_list:
        auths = group.get("author", [])
        if isinstance(auths, dict): auths = [auths]
        for a in auths:
            authors.append({
                "given": a.get("ce:given-name", ""),
                "family": a.get("ce:surname", ""),
            })

    # Extract Year
    pub_year = None
    pubdate = data.get("abstracts-retrieval-response", {}).get("item", {}).get("bibrecord", {}).get("item-info", {}).get("history", {}).get("publication-date", {})
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

def fetch_semantic_scholar_abstract(title):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": title, "limit": 1, "fields": "abstract"}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if data.get("data") and data["data"][0].get("abstract"):
                return data["data"][0]["abstract"]
    except:
        pass
    return ""

def fetch_openalex_abstract_by_doi(doi):
    if not doi: return ""
    doi = doi.lower().strip()
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    params = {"mailto": "maohua.nie@unibas.ch"}

    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200: return ""
        data = r.json()
        inverted = data.get("abstract_inverted_index")
        if not inverted: return ""
        words = []
        for word, positions in inverted.items():
            for p in positions:
                words.append((p, word))
        words.sort(key=lambda x: x[0])
        return " ".join(w for _, w in words).strip()
    except:
        return ""

def get_abstract_with_fallback(it, issn):
    title = it.get("title", [""])[0]
    doi = it.get("DOI")

    # 1) Elsevier
    if doi and issn in ELSEVIER_ISSNS:
        meta = fetch_elsevier_metadata(doi)
        if meta and meta.get("abstract"):
            return meta["abstract"], "elsevier", meta

    # 2) Crossref
    crossref_abs = clean_abstract_text(it.get("abstract", ""))
    if crossref_abs:
        return crossref_abs, "crossref", None

    # 3) OpenAlex
    if doi:
        oa_abs = clean_abstract_text(fetch_openalex_abstract_by_doi(doi))
        if oa_abs:
            return oa_abs, "openalex", None

    # 4) Semantic Scholar
    ss_abs = clean_abstract_text(fetch_semantic_scholar_abstract(title))
    if ss_abs:
        return ss_abs, "semantic_scholar", None

    return "", "none", None

def fetch_range_for_journal(issn, start_date, end_date):
    if hasattr(start_date, "isoformat"): start_date = start_date.isoformat()
    if hasattr(end_date, "isoformat"): end_date = end_date.isoformat()

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

def gpt_relevance_and_summary(title, abstract):
    prompt = f"""
        You are assisting a PhD student in economic psychology who studies **individual-level decision making**.
        His research focuses on how **people mentally represent, evaluate, and compare choice options**, particularly under **risk, uncertainty, and complexity**, and how these processes generate observable behavior.

        Your task is to evaluate whether the paper is relevant to **this cognitive and process-oriented research agenda**.

        ### Treat a paper as RELEVANT if it substantially concerns:
        - Individual decision making under risk or uncertainty.
        - Choice complexity, cognitive load, or informational complexity.
        - Gain–loss asymmetries, loss aversion, ambiguity, probability weighting.
        - Process-level evidence (response times, attention, memory, eye tracking).
        - Computational or formal cognitive models (Bayesian, drift diffusion, evidence accumulation).

        ### Treat a paper as NOT RELEVANT if it primarily focuses on:
        - Market-level, firm-level, or population-level outcomes without modeling individual processes.
        - Purely normative optimization or policy design without psychological interpretation.
        - Field data without a cognitive account.

        Return ONLY this JSON:
        {{
        "relevant": true/false,
        "reason": "1–2 sentences explaining why",
        "summary": "1–2 sentence plain-language summary of the paper"
        }}

        Title: {title}
        Abstract: {abstract}
        """
    try:
        resp = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        content = resp.content[0].text
        # Strip markdown code blocks if present
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        j = json.loads(content)
        return bool(j.get("relevant", False)), j.get("reason", ""), j.get("summary", "")
    except Exception as e:
        log(f"  LLM parse error for '{title[:60]}': {e}")
        log(f"  Raw response: {content[:300] if 'content' in dir() else 'no response'}")
        return False, "Parse error", ""

# =========================================================
# CORE LOGIC
# =========================================================

def get_latest_interval(run_date):
    """
    Determine the last valid digest interval based on the run date.
    
    - If run_date is after the 15th (e.g., Feb 16):
      Returns current month 1st -> 15th.
    
    - If run_date is on or before the 15th (e.g., Feb 2):
      Returns previous month 16th -> End of previous month.
    """
    if run_date.day > 15:
        # Window: 1st -> 15th of current month
        start_date = run_date.replace(day=1)
        end_date = run_date.replace(day=15)
    else:
        # Window: 16th -> End of previous month
        # Calculate last day of previous month
        first_of_current = run_date.replace(day=1)
        end_date = first_of_current - timedelta(days=1)
        # 16th of previous month
        start_date = end_date.replace(day=16)
        
    return start_date, end_date

def find_relevant_papers(start_day, end_day):
    """
    Search and filter papers for the given date window.
    """
    log(f"Searching papers for window: {start_day} → {end_day}")

    all_results = []
    run_log = []

    for j_idx, j in enumerate(JOURNALS, start=1):
        log(f"[{j_idx}/{len(JOURNALS)}] Fetching {j['name']} ({j['issn']})")

        try:
            items = fetch_range_for_journal(j["issn"], start_day, end_day)
        except Exception as e:
            log(f"  ✗ Fetch failed: {e}")
            continue

        valid_items = [
            it for it in items
            if parse_pub_datetime(it)
            and start_day <= parse_pub_datetime(it) <= end_day
            and not is_correction_item(it)
        ]

        if not valid_items:
            continue

        texts, abstracts, sources, metas = [], [], [], []

        for it in valid_items:
            abs_text, source, meta = get_abstract_with_fallback(it, j["issn"])
            title = it.get("title", [""])[0] if it.get("title") else ""

            texts.append(sanitize_for_embedding(f"{title}\n\n{abs_text or ''}"))
            abstracts.append(sanitize_for_embedding(abs_text or ""))
            sources.append(source)
            metas.append(meta)

        embeds = embed_text_batch(texts)
        if not embeds:
            continue

        sims = [cosine_sim(e, topic_emb) for e in embeds]
        ranked_idx = np.argsort(sims)[::-1][:TOP_K_PER_JOURNAL]

        for idx in ranked_idx:
            title = valid_items[idx].get("title", [""])[0]
            doi = valid_items[idx].get("DOI")
            score = round(sims[idx], 2)

            # Log every paper that made it to top-K, regardless of threshold
            entry = {
                "journal": j["name"],
                "title": title,
                "doi": doi,
                "embedding_score": score,
                "passed_threshold": score >= SIM_THRESHOLD,
                "llm_relevant": None,
                "llm_reason": None,
                "final_included": False,
            }

            if sims[idx] < SIM_THRESHOLD:
                run_log.append(entry)
                continue

            relevant, reason, summary = gpt_relevance_and_summary(title, abstracts[idx])
            entry["llm_relevant"] = relevant
            entry["llm_reason"] = reason

            if relevant:
                entry["final_included"] = True
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
                    "relevance_score": score,
                    "doi": doi,
                    "abstract": abstracts[idx],
                    "abstract_source": sources[idx],
                    "summary": summary
                })

            run_log.append(entry)

    # Save log
    log_file = LOG_DIR / f"log_{start_day}_{end_day}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2, ensure_ascii=False)
    log(f"Saved pipeline log to {log_file.name} ({len(run_log)} entries)")

    return sorted(all_results, key=lambda x: x["relevance_score"], reverse=True)

def format_email_body_html(results, start_day, end_day):
    journal_counts = {}
    for r in results:
        j = r.get("journal", "Unknown journal")
        journal_counts[j] = journal_counts.get(j, 0) + 1

    def esc(s):
        if s is None: return ""
        s = str(s)
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <link rel="stylesheet" href="../assets/site.css">
      <style>
        .star-rating {{
          display: inline-flex;
          flex-direction: row-reverse;
          gap: 4px;
        }}
        .star-rating input {{ display: none; }}
        .star-rating label {{
          font-size: 24px;
          color: #e5e7eb;
          cursor: pointer;
          transition: color 0.1s;
        }}
        .star-rating input:checked ~ label,
        .star-rating label:hover,
        .star-rating label:hover ~ label {{
          color: #fbbf24;
        }}
        .admin-controls {{
          position: fixed;
          top: 20px;
          right: 20px;
          z-index: 1000;
          display: none; /* Only show locally */
        }}
        .save-btn {{
          background: #1e3a8a;
          color: white;
          border: none;
          padding: 10px 20px;
          border-radius: 8px;
          cursor: pointer;
          font-weight: 600;
          box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        @media print {{ .admin-controls, .star-rating {{ display: none !important; }} }}
      </style>
    </head>
    <body style="background:#f6f7fb;padding:20px;color:#1f2937;">
      <div class="admin-controls" id="admin-ui">
        <button class="save-btn" onclick="savePage()">💾 Save Ratings</button>
      </div>
      <a id="top"></a>
      <div style="max-width:900px;margin:0 auto;background:#ffffff;border-radius:14px;padding:26px;box-shadow:0 6px 18px rgba(0,0,0,0.08);border:1px solid #eef0f4;">
        <div style="display:flex;align-items:flex-end;justify-content:space-between;gap:14px;flex-wrap:wrap;">
          <div>
            <h1 style="margin:0;font-size:26px;line-height:1.2;color:#111827;">Bi-weekly Research Digest</h1>
            <p style="margin:8px 0 0;font-size:14px;color:#4b5563;">
              <b>Date range:</b> {start_day} → {end_day}<br>
              <b>Total relevant papers:</b> {len(results)}
            </p>
          </div>
          <div style="display:flex; gap:10px; align-items:center;">
            <button id="filter-btn" onclick="toggleTopPicks()" style="background:#ffffff; border:1px solid #e5e7eb; padding:8px 16px; border-radius:10px; font-size:14px; font-weight:600; cursor:pointer; color:#374151; transition:all 0.2s;">
              ⭐ Show Top Picks (4★+)
            </button>
          </div>
        </div>
        <h2 style="margin:26px 0 10px;font-size:18px;border-bottom:1px solid #e5e7eb;padding-bottom:8px;color:#111827;">Summary by Journal</h2>
        <ul style="margin:10px 0 0;padding-left:18px;color:#374151;line-height:1.7;">
    """

    def journal_anchor(journal_name):
        return "journal-" + re.sub(r"[^a-z0-9]+", "-", journal_name.lower()).strip("-")

    for j, c in journal_counts.items():
        html += f"<li><a href='#{journal_anchor(j)}' style='text-decoration:none;color:#2563eb;'><b>{esc(j)}</b></a>: {c}</li>"

    html += "</ul>"

    for journal in journal_counts.keys():
        anchor = journal_anchor(journal)
        html += f"""<a id="{anchor}"></a><h2 class="journal-header" data-journal="{journal}" style="margin:34px 0 10px;font-size:24px;color:#b91c1c;">{esc(journal)}</h2>"""
        
        for idx, r in enumerate(results):
            if r.get("journal") != journal: continue
            
            title = esc(r.get("title", "Untitled"))
            summary = esc(r.get("summary", "")).strip()
            authors = esc(r.get("authors", "Unknown"))
            published = esc(r.get("published", "n.d."))
            score = esc(r.get("relevance_score", ""))
            doi = r.get('doi', '')
            doi_url = f"https://doi.org/{doi}" if doi else ""
            abstract_html = esc(r.get("abstract")) if r.get("abstract") else "Not available."
            paper_id = f"paper-{re.sub(r'[^a-z0-9]', '-', (doi or title).lower())}"

            html += f"""
            <div id="{paper_id}" class="paper-item" data-journal="{journal}" style="background:#ffffff;border:1px solid #e5e7eb;border-radius:14px;padding:18px;margin:14px 0 18px;box-shadow:0 2px 10px rgba(0,0,0,0.04);">
              <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:20px;">
                <h3 style="margin:0 0 10px;font-size:20px;color:#111827;flex:1;">{title}</h3>
                <div class="star-rating" data-paper-id="{paper_id}">
                  <input type="radio" id="star5-{paper_id}" name="rating-{paper_id}" value="5"><label for="star5-{paper_id}">★</label>
                  <input type="radio" id="star4-{paper_id}" name="rating-{paper_id}" value="4"><label for="star4-{paper_id}">★</label>
                  <input type="radio" id="star3-{paper_id}" name="rating-{paper_id}" value="3"><label for="star3-{paper_id}">★</label>
                  <input type="radio" id="star2-{paper_id}" name="rating-{paper_id}" value="2"><label for="star2-{paper_id}">★</label>
                  <input type="radio" id="star1-{paper_id}" name="rating-{paper_id}" value="1"><label for="star1-{paper_id}">★</label>
                </div>
              </div>
              {"<p style='margin:0 0 12px;font-size:14px;color:#374151;line-height:1.6;'>" + summary + "</p>" if summary else ""}
              <div style="font-size:13px;color:#4b5563;line-height:1.7;">
                <div><b>Authors:</b> {authors}</div>
                <div><b>Published:</b> {published}</div>
                <div><b>Relevance Score:</b> {score}</div>
                <div style="margin-top:10px;"><b>DOI:</b> {"<a href='" + esc(doi_url) + "' target='_blank' rel='noopener noreferrer' style='color:#2563eb;text-decoration:none;'>" + esc(doi_url) + "</a>" if doi_url else "Not available."}</div>
                <div style="margin-top:12px;padding:12px;background:#f9fafb;border:1px solid #eef0f4;border-radius:12px;">
                  <b>Abstract:</b><br><span style="color:#374151;">{abstract_html}</span>
                </div>
              </div>
            </div>
            """

    html += """
      </div>
      <a href="#top" class="back-to-top" aria-label="Back to top">↑</a>
      <script>
        let isFilterActive = false;

        // Only show admin controls and allow editing if running locally (file://)
        if (window.location.protocol === 'file:') {
          document.getElementById('admin-ui').style.display = 'block';
        } else {
          // Disable all radio buttons if not local
          document.querySelectorAll('input[type="radio"]').forEach(radio => {
            radio.disabled = true;
          });
          // Add a style to make the stars non-interactive
          const style = document.createElement('style');
          style.innerHTML = '.star-rating { pointer-events: none; }';
          document.head.appendChild(style);
        }

        function toggleTopPicks() {
          isFilterActive = !isFilterActive;
          const btn = document.getElementById('filter-btn');
          const papers = document.querySelectorAll('.paper-item');
          const headers = document.querySelectorAll('.journal-header');

          if (isFilterActive) {
            btn.style.background = '#1e3a8a';
            btn.style.color = '#ffffff';
            btn.innerHTML = '✨ Showing Top Picks (4★+)';

            headers.forEach(header => {
              const journal = header.getAttribute('data-journal');
              const journalPapers = document.querySelectorAll(`.paper-item[data-journal="${journal}"]`);
              let hasTopPick = false;

              journalPapers.forEach(paper => {
                const rating = paper.querySelector('input[type="radio"]:checked')?.value || 0;
                if (parseInt(rating) >= 4) {
                  paper.style.display = 'block';
                  hasTopPick = true;
                } else {
                  paper.style.display = 'none';
                }
              });

              header.style.display = hasTopPick ? 'block' : 'none';
            });
          } else {
            btn.style.background = '#ffffff';
            btn.style.color = '#374151';
            btn.innerHTML = '⭐ Show Top Picks (4★+)';
            
            papers.forEach(p => p.style.display = 'block');
            headers.forEach(h => h.style.display = 'block');
          }
        }

        function savePage() {
          // Reset filter before saving so all papers are visible in the source
          if (isFilterActive) toggleTopPicks();

          // Remove the Save button and admin UI from the saved HTML
          const adminUI = document.getElementById('admin-ui');
          adminUI.style.display = 'none';
          
          // Update the checked attribute of radio buttons based on their current state
          document.querySelectorAll('input[type="radio"]').forEach(radio => {
            if (radio.checked) {
              radio.setAttribute('checked', 'checked');
            } else {
              radio.removeAttribute('checked');
            }
          });

          const htmlContent = document.documentElement.outerHTML;
          const blob = new Blob([htmlContent], { type: 'text/html' });
          const a = document.createElement('a');
          a.href = URL.createObjectURL(blob);
          a.download = window.location.pathname.split('/').pop() || 'digest.html';
          a.click();
          
          // Show admin UI again
          adminUI.style.display = 'block';
        }
      </script>
    </body>
    </html>
    """
    return html
    return html

def save_digest_html(html, run_date):
    """
    Saves the HTML file using the actual run date as the filename.
    Example: 2026-02-02.html
    """
    date_str = run_date.isoformat()
    filename = f"{date_str}.html"
    path = DIGEST_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    return filename, date_str

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
            print("Warning: digests.json was empty or invalid. Reinitializing.")

    # Remove existing entry for the same date if it exists (overwrite behavior)
    index = [d for d in index if d["date"] != date_str]
    index.append(entry)
    
    # Sort by date descending
    index.sort(key=lambda x: x["date"], reverse=True)

    with open(DIGEST_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    # Regenerate the RSS feed so subscribers (via MailerLite etc.) get the new digest
    update_rss_feed()


def _rfc822(date_obj):
    """Format a date/datetime as RFC 822 string, required by the RSS 2.0 spec."""
    if isinstance(date_obj, str):
        date_obj = datetime.strptime(date_obj, "%Y-%m-%d")
    elif isinstance(date_obj, date) and not isinstance(date_obj, datetime):
        date_obj = datetime(date_obj.year, date_obj.month, date_obj.day)
    return date_obj.strftime("%a, %d %b %Y 00:00:00 GMT")


def _xml_escape(text):
    """Escape XML-unsafe characters so the feed validates."""
    if not isinstance(text, str):
        text = str(text)
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;"))


def update_rss_feed():
    """
    Generate /research-digest/feed.xml (RSS 2.0) from digests.json.

    MailerLite (and any other RSS-to-email service) polls this feed and
    automatically emails subscribers whenever a new <item> appears.
    The feed keeps only the most recent FEED_MAX_ITEMS digests.
    """
    if not DIGEST_INDEX.exists():
        return

    try:
        with open(DIGEST_INDEX, "r", encoding="utf-8") as f:
            index = json.load(f)
    except json.JSONDecodeError:
        print("Warning: digests.json could not be read for RSS feed generation.")
        return

    if not index:
        return

    items = sorted(index, key=lambda x: x["date"], reverse=True)[:FEED_MAX_ITEMS]

    channel_title = "Decision Science Digest | Maohua Nie"
    channel_desc = (
        "A bi-weekly reading list of interesting new papers on "
        "decision making, risk, behavioral economics, and cognitive modeling "
        "— automatically curated from 25+ leading journals."
    )
    channel_link = f"{SITE_URL}/research-digest/"
    feed_self = f"{SITE_URL}/research-digest/feed.xml"
    last_build = _rfc822(datetime.utcnow())

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">',
        '  <channel>',
        f'    <title>{_xml_escape(channel_title)}</title>',
        f'    <link>{_xml_escape(channel_link)}</link>',
        f'    <description>{_xml_escape(channel_desc)}</description>',
        '    <language>en-us</language>',
        f'    <atom:link href="{_xml_escape(feed_self)}" rel="self" type="application/rss+xml" />',
        f'    <lastBuildDate>{last_build}</lastBuildDate>',
    ]

    for entry in items:
        item_url = f"{SITE_URL}/research-digest/{entry['file']}"
        paper_count = entry.get("papers", 0)
        plural = "" if paper_count == 1 else "s"
        item_title = f"{entry.get('title', 'Research Digest')} ({paper_count} paper{plural})"
        item_desc = (
            f"This digest features {paper_count} curated new paper{plural} on "
            f"decision making, behavioral economics, and cognitive modeling. "
            f"Read the full digest at {item_url}"
        )
        lines.extend([
            '    <item>',
            f'      <title>{_xml_escape(item_title)}</title>',
            f'      <link>{_xml_escape(item_url)}</link>',
            f'      <guid isPermaLink="true">{_xml_escape(item_url)}</guid>',
            f'      <pubDate>{_rfc822(entry["date"])}</pubDate>',
            f'      <description>{_xml_escape(item_desc)}</description>',
            '    </item>',
        ])

    lines.extend(['  </channel>', '</rss>'])

    with open(FEED_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"RSS feed updated: {FEED_PATH.name} ({len(items)} items)")


def _html_to_email_body(html, digest_url, start_day, end_day, paper_count):
    """
    Convert a saved digest HTML file into an email-safe body.

    Emails can't execute JS, load external stylesheets, or render form
    inputs — so we strip <script>, <link rel="stylesheet">, the star
    rating radio inputs, the admin controls, and the filter button.
    A branded header and a "read online" link are wrapped around the
    remaining content.
    """
    # Strip <script>...</script> blocks
    html = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Strip <link rel="stylesheet" ...>
    html = re.sub(r"<link\s+[^>]*rel=[\"']stylesheet[\"'][^>]*/?>", "", html, flags=re.IGNORECASE)
    # Strip the admin-controls div (buttons only visible locally)
    html = re.sub(r"<div\s+class=[\"']admin-controls[\"'][^>]*>.*?</div>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Strip the interactive "Show Top Picks" filter button
    html = re.sub(r"<button\s+id=[\"']filter-btn[\"'][^>]*>.*?</button>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Strip the whole star-rating block (radio inputs don't render in email)
    html = re.sub(r"<div\s+class=[\"']star-rating[\"'][^>]*>.*?</div>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Strip the floating "back-to-top" link
    html = re.sub(r"<a\s+href=[\"']#top[\"']\s+class=[\"']back-to-top[\"'][^>]*>.*?</a>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Extract just the body content so we can wrap our own header around it
    m = re.search(r"<body[^>]*>(.*?)</body>", html, flags=re.DOTALL | re.IGNORECASE)
    body_inner = m.group(1) if m else html

    header = (
        "<div style=\"text-align:center;padding:28px 24px 18px;background:#ffffff;\">"
        "<h1 style=\"color:#004b7a;margin:0;font-size:24px;font-weight:700;letter-spacing:-0.01em;\">"
        "Decision Science Digest</h1>"
        "<p style=\"color:#6b7280;margin:4px 0 0;font-size:14px;\">by Maohua Nie</p>"
        f"<p style=\"color:#9ca3af;margin:14px 0 6px;font-size:13px;letter-spacing:0.04em;\">"
        f"{start_day} &nbsp;→&nbsp; {end_day}</p>"
        "</div>"
    )

    archive_url = f"{SITE_URL}/research-digest/"
    footer = (
        "<div style=\"text-align:center;padding:20px 24px 30px;background:#ffffff;\">"
        f"<a href=\"{archive_url}\" style=\"display:inline-block;padding:10px 22px;"
        "background:#004b7a;color:#fff;text-decoration:none;border-radius:8px;"
        "font-weight:600;font-size:14px;\">Browse the archive →</a>"
        "</div>"
    )

    wrapper_open = (
        "<!DOCTYPE html><html><body style=\"margin:0;padding:0;background:#f6f7fb;"
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;\">"
    )
    wrapper_close = "</body></html>"

    return wrapper_open + header + body_inner + footer + wrapper_close


def send_digest_email(date_str, filename, results, start_day, end_day):
    """
    Broadcast the new digest to MailerLite subscribers.

    Creates a one-off campaign via MailerLite's API and sends it immediately
    to the Decision Science Digest group. Requires MAILERLITE_API_TOKEN —
    if absent (e.g., local run), the function logs and returns without
    failing the pipeline.
    """
    api_token = os.getenv("MAILERLITE_API_TOKEN")
    if not api_token:
        print("MAILERLITE_API_TOKEN not set — skipping subscriber email broadcast.")
        return

    paper_count = len(results)
    plural = "" if paper_count == 1 else "s"
    digest_url = f"{SITE_URL}/research-digest/{filename}"
    subject = f"Decision Science Digest · {start_day} → {end_day} ({paper_count} paper{plural})"

    # Load the already-saved digest HTML and adapt it for email rendering.
    digest_path = DIGEST_DIR / filename
    if not digest_path.exists():
        print(f"MailerLite: digest file {digest_path} not found; skipping email.")
        return
    with open(digest_path, "r", encoding="utf-8") as f:
        digest_html = f.read()

    html_body = _html_to_email_body(digest_html, digest_url, start_day, end_day, paper_count)

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # 1. Create the campaign (draft state)
    # NOTE: Per MailerLite's official SDK tests, the create-campaign payload
    # uses name/language_id/type/emails. Recipients (groups) are NOT set at
    # creation — the campaign defaults to all subscribers, which is fine
    # while the "Decision Science Digest" group is the only one collecting
    # signups. To target a specific group, set it via the update endpoint
    # before scheduling.
    # Match the exact fields MailerLite's official SDK sends (name/language_id/
    # type/emails with subject/from_name/from/content). Extra fields like
    # plain_text cause the API to reject the whole emails[0] entry with a
    # vague "must be an array" error; MailerLite auto-generates plain text.
    create_payload = {
        "name": f"Decision Science Digest · {date_str}",
        "language_id": 1,
        "type": "regular",
        "emails": [{
            "subject": subject,
            "from_name": MAILERLITE_FROM_NAME,
            "from": MAILERLITE_FROM_EMAIL,
            "content": html_body,
        }],
    }

    try:
        r = requests.post(
            "https://connect.mailerlite.com/api/campaigns",
            headers=headers, json=create_payload, timeout=30,
        )
        r.raise_for_status()
        campaign_id = r.json()["data"]["id"]
        print(f"MailerLite campaign created: id={campaign_id}")
    except requests.HTTPError as e:
        body = e.response.text if e.response is not None else ""
        print(f"MailerLite: failed to create campaign ({e}). Response: {body}")
        return
    except Exception as e:
        print(f"MailerLite: unexpected error creating campaign: {e}")
        return

    # 2. Schedule for immediate delivery
    try:
        r2 = requests.post(
            f"https://connect.mailerlite.com/api/campaigns/{campaign_id}/schedule",
            headers=headers, json={"delivery": "instant"}, timeout=30,
        )
        r2.raise_for_status()
        print(f"MailerLite campaign {campaign_id} scheduled for instant send.")
    except requests.HTTPError as e:
        body = e.response.text if e.response is not None else ""
        print(f"MailerLite: failed to schedule campaign {campaign_id} ({e}). Response: {body}")
    except Exception as e:
        print(f"MailerLite: unexpected error scheduling campaign {campaign_id}: {e}")


# =========================================================
# MAIN & CATCH-UP LOGIC
# =========================================================

def get_report_date(end_day):
    """
    Returns the date used for the filename.
    - If the digest ends on the 15th, filename date is the 15th.
    - If the digest ends on the last day of the month, filename date is the 1st of the next month.
    """
    if end_day.day == 15:
        return end_day
    else:
        # End of the month, set filename to the 1st of the next month
        return end_day + timedelta(days=1)

def get_last_generated_report_date():
    """Reads digests.json to find the most recently generated report date."""
    if DIGEST_INDEX.exists():
        try:
            with open(DIGEST_INDEX, "r", encoding="utf-8") as f:
                index = json.load(f)
                if index:
                    # ISO format dates sort alphabetically correctly
                    latest_str = max([entry["date"] for entry in index])
                    return datetime.strptime(latest_str, "%Y-%m-%d").date()
        except (json.JSONDecodeError, ValueError):
            pass
    return None

def get_interval_from_report_date(r_date):
    """Reconstructs the start_day and end_day from a given report date."""
    if r_date.day == 15:
        return r_date.replace(day=1), r_date
    elif r_date.day == 1:
        end_day = r_date - timedelta(days=1)
        start_day = end_day.replace(day=16)
        return start_day, end_day
    else:
        raise ValueError(f"Unexpected report date: {r_date}")

def get_next_report_date(r_date):
    """Calculates the chronologically next report date."""
    if r_date.day == 15:
        # Move to the 1st of the next month
        next_month = r_date.replace(day=28) + timedelta(days=5)
        return next_month.replace(day=1)
    elif r_date.day == 1:
        # Move to the 15th of the current month
        return r_date.replace(day=15)
    else:
        raise ValueError(f"Unexpected report date: {r_date}")

def main():
    today = datetime.today().date()
    
    # 1. Determine what the CURRENT latest valid interval should be
    target_start, target_end = get_latest_interval(today)
    target_report_date = get_report_date(target_end)
    
    # 2. Find where we left off in the JSON
    last_generated_date = get_last_generated_report_date()
    
    intervals_to_run = []
    
    if not last_generated_date:
        # Fallback: if no JSON is found, just run the current target interval
        intervals_to_run.append((target_start, target_end, target_report_date))
    else:
        # Step forward half a month at a time until we reach the target
        current_report_date = get_next_report_date(last_generated_date)
        while current_report_date <= target_report_date:
            s_day, e_day = get_interval_from_report_date(current_report_date)
            intervals_to_run.append((s_day, e_day, current_report_date))
            current_report_date = get_next_report_date(current_report_date)

    # 3. Execution
    if not intervals_to_run:
        print("✅ Everything is up to date! No missing digests to generate.")
        return

    for start_day, end_day, report_date in intervals_to_run:
        print(f"\n==============================================")
        print(f"Generating digest for interval: {start_day} → {end_day}")
        print(f"Target Filename Date: {report_date}")
        print(f"==============================================")
        
        results = find_relevant_papers(start_day, end_day)

        # Note: Even if 0 papers are found, we still save the empty digest. 
        # If we didn't, it wouldn't be logged in the JSON, and the script 
        # would keep trying to generate this exact empty interval on every run.
        if not results:
            print(f"No relevant papers found for {start_day} → {end_day}. Creating an empty digest.")

        html = format_email_body_html(results, start_day, end_day)
        filename, date_str = save_digest_html(html, report_date)

        update_digest_index(date_str, filename, results, start_day, end_day)
        print(f"Success! Saved digest to: {DIGEST_DIR.name}/{filename}")

        # Broadcast to MailerLite subscribers (no-op if MAILERLITE_API_TOKEN is unset)
        send_digest_email(date_str, filename, results, start_day, end_day)

    print("\n🎉 All missing digests have been generated and the index is up to date!")

if __name__ == "__main__":
    main()