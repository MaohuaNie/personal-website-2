import os
import json
import requests
import numpy as np
from datetime import datetime, timedelta
import re
from pathlib import Path
from urllib.parse import quote

# OpenAI / Anthropic are imported lazily in init_clients() so this module can be
# imported (e.g. to reuse format_email_body_html) without those packages or keys.

# =========================================================
# CONFIG & CONSTANTS    
# =========================================================

ROOT = Path(__file__).parent
DIGEST_DIR = ROOT / "research-digest"
DIGEST_DIR.mkdir(exist_ok=True)
LOG_DIR = ROOT / "research-digest" / "logs"
LOG_DIR.mkdir(exist_ok=True)
WECHAT_DIR = DIGEST_DIR / "wechat"
WECHAT_DIR.mkdir(exist_ok=True)

DIGEST_INDEX = DIGEST_DIR / "digests.json"
SITE_URL = "https://maohuanie.com"

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
    {"name": "Journal of Economic Behavior & Organization", "issn": "0167-2681"},
    {"name": "Experimental Economics", "issn": "1386-4157"},
    {"name": "Journal of Behavioral and Experimental Economics", "issn": "2214-8043"},
    {"name": "Theory and Decision", "issn": "0040-5833"},
    {"name": "Nature Neuroscience", "issn": "1097-6256"},
    {"name": "Neuron", "issn": "0896-6273"},
    {"name": "Nature Communications", "issn": "2041-1723"},
    {"name": "PNAS", "issn": "0027-8424"},
    {"name": "eLife", "issn": "2050-084X"},
    {"name": "PLOS Computational Biology", "issn": "1553-7358"},
    {"name": "Attention, Perception, & Psychophysics", "issn": "1943-3921"},
]

ELSEVIER_ISSNS = {
    "0010-0277",  # Cognition
    "0749-5978",  # OBHDP
    "0010-0285",  # Cognitive Psychology
    "0167-4870",  # Journal of Economic Psychology
    "0022-2496",  # Journal of Mathematical Psychology
    "0167-2681",  # Journal of Economic Behavior & Organization
    "2214-8043",  # Journal of Behavioral and Experimental Economics
    "0896-6273",  # Neuron
}

TOP_K_PER_JOURNAL = 30
SIM_THRESHOLD = 0.30

# Papers are grouped in the digest by field. The LLM assigns each paper one of
# these based on the paper's own content (not its journal), so interdisciplinary
# journals (PNAS, Nature Human Behaviour, ...) no longer need a "General" bucket.
FIELD_ORDER = ["Economics", "Psychology", "Neuroscience"]
DEFAULT_FIELD = "Psychology"  # fallback when the model returns something unexpected

def normalize_field(value):
    """Map a raw model field string onto one of FIELD_ORDER (default: Psychology)."""
    v = (value or "").strip().lower()
    if v.startswith("econ"):
        return "Economics"
    if v.startswith("neuro"):
        return "Neuroscience"
    if v.startswith("psych"):
        return "Psychology"
    return DEFAULT_FIELD

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

attention allocation during decision making,
gaze, eye movements, and fixations while evaluating and comparing choice options,
gaze-weighted evidence accumulation, attentional drift diffusion model (aDDM),
how attention allocation relates to choice and response times,
how prior preferences bias attention during deliberation,

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Clients and the precomputed topic embedding are initialised lazily by
# init_clients() so that importing this module (e.g. to reuse
# format_email_body_html for previews / regeneration) does not require API keys
# or make network calls. A live digest run calls init_clients() from main().
openai_client = None
anthropic_client = None
topic_emb = None

def init_clients():
    """Validate keys, create API clients, and precompute the topic embedding."""
    global openai_client, anthropic_client, topic_emb
    if topic_emb is not None:
        return
    if not ELSEVIER_API_KEY:
        raise RuntimeError("Missing ELSEVIER_API_KEY environment variable")
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Missing ANTHROPIC_API_KEY environment variable")

    from openai import OpenAI
    from anthropic import Anthropic
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    topic_emb = np.array(
        openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[TOPIC_TEXT],
        ).data[0].embedding
    )


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

def _dedup_key(it):
    """Stable identity for a paper, used to collapse duplicate records.

    eLife (and similar) deposit a versioned DOI (…/elife.104684.5) alongside the
    canonical one (…/elife.104684), and Crossref returns both as journal-articles.
    Key on the canonical DOI so versions collapse; fall back to a normalized title
    when a DOI is missing.
    """
    doi = (it.get("DOI") or "").lower().strip()
    if doi:
        m = re.match(r"(10\.7554/elife\.\d+)\.\d+$", doi)
        return m.group(1) if m else doi
    title = (it.get("title") or [""])[0].lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", title)).strip()

def _is_versioned_doi(doi):
    return bool(re.match(r"10\.7554/elife\.\d+\.\d+$", (doi or "").lower().strip()))

def dedup_items(items):
    """Keep one record per paper, preferring the canonical (non-versioned) DOI."""
    # Stable sort puts canonical DOIs before versioned ones, so the first
    # occurrence we keep is the canonical record.
    ordered = sorted(items, key=lambda it: _is_versioned_doi(it.get("DOI", "")))
    seen, out = set(), []
    for it in ordered:
        k = _dedup_key(it)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out

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
    for key in ["published-online", "published-print", "created"]:
        if key in item:
            d = item[key].get("date-parts", [[]])[0]
            if len(d) >= 2:
                return f"{d[0]}-{str(d[1]).zfill(2)}"
            if len(d) == 1:
                return str(d[0])
    return "n.d."

def parse_pub_datetime(item):
    for key in ["published-online", "published-print", "created"]:
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
        You are assisting a PhD student in economic psychology whose **main field is decision making under risk and uncertainty** — how people mentally represent, evaluate, and compare choice options (including under complexity), and how these cognitive processes generate observable behavior such as choices and response times. He also has a **growing, secondary interest in how attention is allocated during these decisions** — how gaze and eye movements relate to choice and response times, and how prior preferences may bias attention during deliberation.

        Your task is to evaluate whether the paper is relevant to **this research agenda, which centers on risky / uncertain decision making and additionally values attention-allocation work that is tied to choice**.

        ### Treat a paper as RELEVANT if it substantially concerns:
        - Individual decision making under risk, uncertainty, or complexity.
        - Choice complexity, cognitive load, or informational complexity.
        - Gain–loss asymmetries, loss aversion, ambiguity, probability weighting.
        - Process-level evidence (response times, attention, memory, eye tracking).
        - Computational or formal cognitive models (Bayesian, drift diffusion, evidence accumulation, sequential sampling).
        - Additionally (a growing interest): how attention, gaze, eye movements, or fixations are allocated during decisions, judgments, or value-based / risky choice, and how this relates to choice, preference, or response times (e.g., gaze-weighted evidence accumulation, attentional drift diffusion / aDDM, gaze-cascade effects) — including cognitive-neuroscience work, provided it still connects to choice, value, preference, judgment, or response-time behavior.

        ### Treat a paper as NOT RELEVANT if it primarily focuses on:
        - Market-level, firm-level, or population-level outcomes without modeling individual processes.
        - Purely normative optimization or policy design without psychological interpretation.
        - Field data without a cognitive account.
        - Low-level perception, clinical, or neurobiological questions with no link to decision making, value, preference, attention allocation, or choice / response-time behavior.

        ### Also classify the paper into ONE field, based on the paper's own
        ### content, methods, and framing — NOT on the journal it appeared in:
        - "Economics": economic decision-making, behavioral / experimental economics,
          risk & uncertainty framed in an economic tradition, markets, incentives,
          preference elicitation, finance, game-theoretic behavior.
        - "Psychology": cognitive / experimental psychology, judgment and decision
          making, cognitive modeling, attention, memory, perception, response times.
        - "Neuroscience": neural mechanisms, brain imaging / recording, computational
          neuroscience, neural correlates of value, choice, or attention.
        Pick the single BEST fit. Interdisciplinary papers (e.g. in Nature Human
        Behaviour, PNAS, Nature Communications, eLife) must still be assigned to the
        one field their core approach most resembles.

        Return ONLY this JSON:
        {{
        "relevant": true/false,
        "reason": "1–2 sentences explaining why",
        "summary": "1–2 sentence plain-language summary of the paper",
        "field": "Economics" | "Psychology" | "Neuroscience"
        }}

        Title: {title}
        Abstract: {abstract}
        """
    try:
        resp = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=350,
            messages=[{"role": "user", "content": prompt}]
        )
        content = resp.content[0].text
        # Strip markdown code blocks if present
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        j = json.loads(content)
        return (
            bool(j.get("relevant", False)),
            j.get("reason", ""),
            j.get("summary", ""),
            normalize_field(j.get("field", "")),
        )
    except Exception as e:
        log(f"  LLM parse error for '{title[:60]}': {e}")
        log(f"  Raw response: {content[:300] if 'content' in dir() else 'no response'}")
        return False, "Parse error", "", DEFAULT_FIELD

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

        before = len(valid_items)
        valid_items = dedup_items(valid_items)
        if before != len(valid_items):
            log(f"  Deduped {before - len(valid_items)} duplicate record(s) (e.g. eLife versions)")

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

            relevant, reason, summary, field = gpt_relevance_and_summary(title, abstracts[idx])
            entry["llm_relevant"] = relevant
            entry["llm_reason"] = reason
            entry["llm_field"] = field

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
                    "field": field,
                    "relevance_score": score,
                    "doi": doi,
                    "abstract": abstracts[idx],
                    "abstract_source": sources[idx],
                    "summary": summary,
                    "reason": reason
                })

            run_log.append(entry)

    # Save log
    log_file = LOG_DIR / f"log_{start_day}_{end_day}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2, ensure_ascii=False)
    log(f"Saved pipeline log to {log_file.name} ({len(run_log)} entries)")

    return sorted(all_results, key=lambda x: x["relevance_score"], reverse=True)

FIELD_META = {
    "Economics":    {"color": "#1c5d54"},   # deep teal
    "Psychology":   {"color": "#8a2b3a"},   # muted burgundy
    "Neuroscience": {"color": "#3d3a78"},   # muted indigo
}
FIELD_DEFAULT_COLOR = "#495159"
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]

def format_email_body_html(results, start_day, end_day):
    def esc(s):
        if s is None: return ""
        s = str(s)
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")

    def slug(*parts):
        return re.sub(r"[^a-z0-9]+", "-", "-".join(parts).lower()).strip("-")

    def field_anchor(field_name):
        return "field-" + slug(field_name)

    def journal_anchor(field_name, journal_name):
        return "journal-" + slug(field_name, journal_name)

    def field_color(field_name):
        return FIELD_META.get(field_name, {}).get("color", FIELD_DEFAULT_COLOR)

    def paper_field(r):
        return r.get("field") or DEFAULT_FIELD

    # Order fields by FIELD_ORDER, then any unexpected ones; keep only fields
    # that actually have papers this issue.
    present = [f for f in FIELD_ORDER if any(paper_field(r) == f for r in results)]
    for r in results:
        f = paper_field(r)
        if f not in present:
            present.append(f)

    def journals_in_field(field):
        seen = []
        for r in results:
            if paper_field(r) != field:
                continue
            j = r.get("journal", "Unknown journal")
            if j not in seen:
                seen.append(j)
        return seen

    range_label = f"{start_day:%d %b %Y} – {end_day:%d %b %Y}"

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
      <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;0,8..60,700;1,8..60,400&display=swap">
      <link rel="stylesheet" href="../assets/site.css">
      <style>
        :root {{
          --paper:#fffdf9; --page:#efece5; --ink:#23272c; --ink-soft:#4c525a;
          --muted:#8b9099; --rule:#e7e2d7; --rule-strong:#cfc8b9; --accent:#0b4a6f;
        }}
        html, body {{ margin:0; }}
        body {{
          background:var(--page); color:var(--ink);
          font-family:'Source Serif 4','Iowan Old Style','Palatino Linotype',Palatino,Georgia,'Times New Roman',serif;
          font-size:17px; line-height:1.6;
          -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
          padding:28px 18px 90px;
        }}
        .sans {{ font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif; }}
        .sheet {{
          max-width:800px; margin:0 auto; background:var(--paper);
          padding:60px 68px 76px;
          border:1px solid var(--rule);
          box-shadow:0 1px 2px rgba(20,20,30,.04), 0 24px 60px rgba(30,30,45,.07);
        }}

        /* ---- Masthead ---- */
        .masthead {{ text-align:center; border-bottom:3px double var(--rule-strong); padding-bottom:24px; }}
        .masthead .eyebrow {{
          font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
          font-size:11px; letter-spacing:.34em; text-transform:uppercase;
          color:var(--muted); margin:0 0 12px;
        }}
        .masthead h1 {{ margin:0; font-size:42px; font-weight:700; letter-spacing:.005em; line-height:1.08; color:#1a1d21; }}
        .masthead .meta {{
          font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
          font-size:13px; letter-spacing:.03em; color:var(--ink-soft); margin:14px 0 0;
        }}
        .masthead .meta .dot {{ color:var(--rule-strong); margin:0 9px; }}

        /* ---- Toolbar ---- */
        .toolbar {{ display:flex; justify-content:center; margin:18px 0 0; }}
        #filter-btn {{
          font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
          background:transparent; border:1px solid var(--rule-strong); color:var(--ink-soft);
          padding:7px 15px; border-radius:2px; font-size:12px; letter-spacing:.05em;
          cursor:pointer; transition:all .18s ease;
        }}
        #filter-btn:hover {{ border-color:var(--accent); color:var(--accent); }}

        /* ---- Contents (table of contents) ---- */
        .contents {{ margin:34px 0 6px; }}
        .contents-title {{
          font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
          text-align:center; font-size:12px; letter-spacing:.28em; text-transform:uppercase;
          color:var(--muted); margin:0 0 20px;
        }}
        .toc-field {{ margin:0 0 18px; }}
        .toc-fname {{ font-size:18px; font-weight:600; text-decoration:none; }}
        .toc-fname:hover {{ text-decoration:underline; }}
        .toc-fcount {{
          font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
          font-size:12px; color:var(--muted); margin-left:8px;
        }}
        .toc-list {{ list-style:none; margin:7px 0 0; padding:0; }}
        .toc-row {{ display:flex; align-items:baseline; padding:2px 0; font-size:15.5px; }}
        .toc-row a {{ color:var(--ink-soft); text-decoration:none; white-space:nowrap; }}
        .toc-row a:hover {{ color:var(--accent); }}
        .toc-lead {{ flex:1; border-bottom:1px dotted var(--rule-strong); margin:0 9px; position:relative; top:-4px; }}
        .toc-row .n {{
          font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
          font-size:12.5px; color:var(--muted);
        }}

        /* ---- Field section ---- */
        .field-header {{
          display:flex; align-items:baseline; gap:15px;
          margin:64px 0 6px; padding-bottom:10px; border-bottom:2px solid currentColor;
        }}
        .field-header .rn {{ font-size:20px; font-weight:600; opacity:.5; }}
        .field-header .ft {{ font-size:31px; font-weight:700; letter-spacing:.005em; }}
        .field-header .fc {{
          margin-left:auto; font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
          font-size:12px; letter-spacing:.06em; opacity:.6; align-self:center;
        }}

        /* ---- Journal sub-header ---- */
        .journal-header {{
          font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
          font-size:12px; font-weight:600; letter-spacing:.19em; text-transform:uppercase;
          color:var(--muted); margin:34px 0 2px;
        }}

        /* ---- Paper entry ---- */
        .paper {{ padding:22px 0 24px; border-bottom:1px solid var(--rule); }}
        .paper-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:22px; }}
        .paper-title {{ margin:0; font-size:20px; line-height:1.34; font-weight:600; color:#1c1f23; flex:1; }}
        .paper-title .pn {{
          font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
          font-size:13px; font-weight:600; color:var(--muted); margin-right:9px;
        }}
        .paper-authors {{ margin:8px 0 0; font-style:italic; font-size:15.5px; color:var(--ink-soft); }}
        .paper-summary {{ margin:11px 0 0; font-size:16px; line-height:1.58; color:var(--ink); }}
        .paper-meta {{
          font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
          font-size:12px; letter-spacing:.02em; color:var(--muted);
          margin:12px 0 0; display:flex; flex-wrap:wrap; gap:8px 18px;
        }}
        .paper-meta a {{ color:var(--accent); text-decoration:none; }}
        .paper-meta a:hover {{ text-decoration:underline; }}
        .paper-abstract {{
          margin:14px 0 0; padding:1px 0 1px 20px; border-left:2px solid var(--rule-strong);
          font-size:14.5px; line-height:1.66; color:var(--ink-soft);
          text-align:justify; hyphens:auto;
        }}
        .paper-abstract .lbl {{
          display:block; font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
          font-size:10.5px; letter-spacing:.16em; text-transform:uppercase;
          color:var(--muted); margin:0 0 6px;
        }}

        /* ---- Star rating ---- */
        .star-rating {{ display:inline-flex; flex-direction:row-reverse; gap:3px; flex-shrink:0; }}
        .star-rating input {{ display:none; }}
        .star-rating label {{ font-size:17px; color:#ded8ca; cursor:pointer; transition:color .1s; }}
        .star-rating input:checked ~ label,
        .star-rating label:hover,
        .star-rating label:hover ~ label {{ color:#c19a35; }}

        /* ---- Admin ---- */
        .admin-controls {{ position:fixed; top:20px; right:20px; z-index:1000; display:none; }}
        .save-btn {{
          background:var(--accent); color:#fff; border:none; padding:10px 20px;
          border-radius:4px; cursor:pointer; font-weight:600; box-shadow:0 4px 12px rgba(0,0,0,.15);
        }}

        @media print {{ .admin-controls, .star-rating, .toolbar {{ display:none !important; }}
          body {{ background:#fff; padding:0; }} .sheet {{ box-shadow:none; border:none; }} }}
        @media (max-width:640px) {{
          .sheet {{ padding:34px 22px 46px; }}
          .masthead h1 {{ font-size:31px; }}
          .field-header .ft {{ font-size:25px; }}
          .field-header .fc {{ display:none; }}
          .paper-head {{ flex-direction:column; gap:10px; }}
        }}
      </style>
    </head>
    <body>
      <div class="admin-controls" id="admin-ui">
        <button class="save-btn" onclick="savePage()">💾 Save Ratings</button>
      </div>
      <a id="top"></a>
      <div class="sheet">
        <header class="masthead">
          <p class="eyebrow">Maohua Nie · Research Digest</p>
          <h1>Bi-weekly Research Digest</h1>
          <p class="meta">{range_label}<span class="dot">◆</span>{len(results)} paper{"s" if len(results) != 1 else ""} across {len(present)} field{"s" if len(present) != 1 else ""}</p>
        </header>

        <div class="toolbar">
          <button id="filter-btn" onclick="toggleTopPicks()">★ Show top picks (4★+)</button>
        </div>

        <nav class="contents">
          <p class="contents-title">Contents</p>
    """

    # ---- Contents: field -> journals with dotted leaders ----
    for i, field in enumerate(present):
        color = field_color(field)
        field_total = sum(1 for r in results if paper_field(r) == field)
        html += (
            f'<div class="toc-field">'
            f'<a class="toc-fname" href="#{field_anchor(field)}" style="color:{color}">'
            f'{ROMAN[i]}. {esc(field)}</a>'
            f'<span class="toc-fcount">{field_total}</span>'
            f'<ul class="toc-list">'
        )
        for journal in journals_in_field(field):
            c = sum(1 for r in results if paper_field(r) == field and r.get("journal") == journal)
            html += (
                f'<li class="toc-row"><a href="#{journal_anchor(field, journal)}">{esc(journal)}</a>'
                f'<span class="toc-lead"></span><span class="n">{c}</span></li>'
            )
        html += "</ul></div>"
    html += "</nav>"

    # ---- Body: field sections -> journal sub-sections -> papers ----
    paper_no = 0
    for i, field in enumerate(present):
        color = field_color(field)
        f_slug = slug(field)
        field_total = sum(1 for r in results if paper_field(r) == field)
        html += (
            f'<a id="{field_anchor(field)}"></a>'
            f'<h2 class="field-header" data-field="{f_slug}" style="color:{color}">'
            f'<span class="rn">{ROMAN[i]}</span><span class="ft">{esc(field)}</span>'
            f'<span class="fc">{field_total} paper{"s" if field_total != 1 else ""}</span></h2>'
        )

        for journal in journals_in_field(field):
            g_slug = slug(field, journal)
            html += (
                f'<a id="{journal_anchor(field, journal)}"></a>'
                f'<h3 class="journal-header" data-group="{g_slug}">{esc(journal)}</h3>'
            )

            for r in results:
                if paper_field(r) != field or r.get("journal") != journal:
                    continue

                paper_no += 1
                title = esc(r.get("title", "Untitled"))
                summary = esc(r.get("summary", "")).strip()
                authors = esc(r.get("authors", "Unknown"))
                published = esc(r.get("published", "n.d."))
                score = esc(r.get("relevance_score", ""))
                doi = r.get('doi', '')
                doi_url = f"https://doi.org/{doi}" if doi else ""
                abstract_html = esc(r.get("abstract")) if r.get("abstract") else "Not available."
                paper_id = f"paper-{re.sub(r'[^a-z0-9]', '-', (doi or title).lower())}"

                doi_html = (
                    f"<a href='{esc(doi_url)}' target='_blank' rel='noopener noreferrer'>doi.org/{esc(doi)}</a>"
                    if doi_url else "DOI not available"
                )

                html += f"""
                <div id="{paper_id}" class="paper-item paper" data-field="{f_slug}" data-group="{g_slug}">
                  <div class="paper-head">
                    <h4 class="paper-title"><span class="pn">{paper_no}</span>{title}</h4>
                    <div class="star-rating" data-paper-id="{paper_id}">
                      <input type="radio" id="star5-{paper_id}" name="rating-{paper_id}" value="5"><label for="star5-{paper_id}">★</label>
                      <input type="radio" id="star4-{paper_id}" name="rating-{paper_id}" value="4"><label for="star4-{paper_id}">★</label>
                      <input type="radio" id="star3-{paper_id}" name="rating-{paper_id}" value="3"><label for="star3-{paper_id}">★</label>
                      <input type="radio" id="star2-{paper_id}" name="rating-{paper_id}" value="2"><label for="star2-{paper_id}">★</label>
                      <input type="radio" id="star1-{paper_id}" name="rating-{paper_id}" value="1"><label for="star1-{paper_id}">★</label>
                    </div>
                  </div>
                  <p class="paper-authors">{authors}</p>
                  {"<p class='paper-summary'>" + summary + "</p>" if summary else ""}
                  <p class="paper-meta"><span>{published}</span><span>Relevance {score}</span><span>{doi_html}</span></p>
                  <div class="paper-abstract"><span class="lbl">Abstract</span>{abstract_html}</div>
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
          const journalHeaders = document.querySelectorAll('.journal-header');
          const fieldHeaders = document.querySelectorAll('.field-header');

          if (isFilterActive) {
            btn.style.background = '#0b4a6f';
            btn.style.color = '#ffffff';
            btn.style.borderColor = '#0b4a6f';
            btn.innerHTML = '★ Showing top picks (4★+)';

            // Show/hide papers by rating.
            papers.forEach(paper => {
              const rating = paper.querySelector('input[type="radio"]:checked')?.value || 0;
              paper.style.display = parseInt(rating) >= 4 ? 'block' : 'none';
            });

            // Hide journal sub-headers whose papers are all filtered out.
            journalHeaders.forEach(header => {
              const group = header.getAttribute('data-group');
              const groupPapers = document.querySelectorAll(`.paper-item[data-group="${group}"]`);
              const anyVisible = Array.from(groupPapers).some(p => p.style.display !== 'none');
              header.style.display = anyVisible ? 'block' : 'none';
            });

            // Hide field headers whose whole section is filtered out.
            fieldHeaders.forEach(header => {
              const field = header.getAttribute('data-field');
              const fieldPapers = document.querySelectorAll(`.paper-item[data-field="${field}"]`);
              const anyVisible = Array.from(fieldPapers).some(p => p.style.display !== 'none');
              header.style.display = anyVisible ? 'block' : 'none';
            });
          } else {
            btn.style.background = 'transparent';
            btn.style.color = '#4c525a';
            btn.style.borderColor = '#cfc8b9';
            btn.innerHTML = '★ Show top picks (4★+)';

            papers.forEach(p => p.style.display = 'block');
            journalHeaders.forEach(h => h.style.display = 'block');
            fieldHeaders.forEach(h => h.style.display = 'block');
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


def wechat_push_title(start_day):
    """Per-issue push title, e.g. '文献汇总：6月上'.

    上 = first half (1st–15th), 下 = second half (16th–month end). Derived from
    the interval start so the second-half issue (filed on the 1st of the next
    month) is still labelled with its content month.
    """
    half = "上" if start_day.day < 16 else "下"
    return f"文献汇总：{start_day.month}月{half}"


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

def save_wechat_docx(results, start_day, end_day, run_date):
    """Write a .docx for WeChat's 文档导入 (document import) upload route.

    Organised into field sections (Economics / Psychology / Neuroscience),
    then journal sub-sections, with numbered entries — mirroring the website.
    Paper titles are real hyperlinks to their DOI — clickable when the
    .docx is opened in Word/Pages and on the website. NOTE: WeChat strips
    external links from the article body on import, so inside the published
    article the linked title becomes plain text; the reliable click-through for
    readers is the 「阅读原文」 link, which should point at this issue's page.
    Saved as research-digest/wechat/<date>.docx.
    """
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
    except ImportError:
        log("python-docx not installed; skipping .docx export (pip install python-docx)")
        return None

    BRAND, GREY, INK = "004B7A", "888888", "1A1A1A"

    doc = Document()

    def para(align=None, before=None, after=None):
        p = doc.add_paragraph()
        if align is not None:
            p.alignment = align
        if before is not None:
            p.paragraph_format.space_before = Pt(before)
        if after is not None:
            p.paragraph_format.space_after = Pt(after)
        return p

    def run(p, text, bold=False, size=11, color=None):
        r = p.add_run(text)
        r.bold = bold
        r.font.size = Pt(size)
        if color is not None:
            r.font.color.rgb = RGBColor.from_string(color)
        return r

    def link(p, url, text, color=BRAND, bold=True, size=13):
        """Append a real hyperlink run (clickable in Word / on the web)."""
        r_id = p.part.relate_to(
            url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
            is_external=True)
        h = OxmlElement("w:hyperlink")
        h.set(qn("r:id"), r_id)
        r = OxmlElement("w:r")
        rpr = OxmlElement("w:rPr")
        c = OxmlElement("w:color")
        c.set(qn("w:val"), color)
        rpr.append(c)
        u = OxmlElement("w:u")
        u.set(qn("w:val"), "single")
        rpr.append(u)
        if bold:
            rpr.append(OxmlElement("w:b"))
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), str(int(size * 2)))
        rpr.append(sz)
        r.append(rpr)
        t = OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = text
        r.append(t)
        h.append(r)
        p._p.append(h)

    push_title = wechat_push_title(start_day)
    run(para(align=WD_ALIGN_PARAGRAPH.CENTER, after=2), push_title, bold=True, size=20, color=BRAND)
    run(para(align=WD_ALIGN_PARAGRAPH.CENTER, after=10),
        f"{start_day} → {end_day} · 共 {len(results)} 篇", size=9, color=GREY)

    if not results:
        run(para(), "本期暂无符合主题的新论文。", color=GREY)
    else:
        # Mirror the website: group by field, then by journal within each field.
        FIELD_ZH = {"Economics": "经济学 · Economics",
                    "Psychology": "心理学 · Psychology",
                    "Neuroscience": "神经科学 · Neuroscience"}
        FIELD_HEX = {"Economics": "1C5D54", "Psychology": "8A2B3A", "Neuroscience": "3D3A78"}
        ROMAN_ZH = ["Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ"]

        def pfield(r):
            return r.get("field") or DEFAULT_FIELD

        present = [f for f in FIELD_ORDER if any(pfield(r) == f for r in results)]
        for r in results:
            f = pfield(r)
            if f not in present:
                present.append(f)

        idx = 0
        for fi, field in enumerate(present):
            fcolor = FIELD_HEX.get(field, BRAND)
            fcount = sum(1 for r in results if pfield(r) == field)
            run(para(before=24, after=2),
                f"{ROMAN_ZH[fi]}  {FIELD_ZH.get(field, field)}（{fcount}）",
                bold=True, size=15, color=fcolor)

            seen = []
            for r in results:
                if pfield(r) == field and r.get("journal", "Unknown journal") not in seen:
                    seen.append(r.get("journal", "Unknown journal"))

            for j in seen:
                run(para(before=12, after=6), f"▎ {j}", bold=True, size=11, color=GREY)
                first = True
                for r in results:
                    if pfield(r) != field or r.get("journal") != j:
                        continue
                    if not first:  # light separator between papers in the same journal
                        run(para(align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=12),
                            "· · ·", size=10, color=GREY)
                    first = False
                    idx += 1
                    tp = para(after=2)
                    run(tp, f"{idx}. ", bold=True, size=13, color=INK)
                    doi = (r.get("doi") or "").strip()
                    title = r.get("title", "Untitled")
                    if doi:
                        link(tp, f"https://doi.org/{doi}", title, color=BRAND, bold=True, size=13)
                    else:
                        run(tp, title, bold=True, size=13, color=INK)
                    run(para(after=5), r.get("authors", "Unknown"), size=9, color=GREY)
                    if r.get("summary"):
                        ps = para(after=4)
                        run(ps, "内容简介 · ", bold=True, color=INK)
                        run(ps, r["summary"])

    run(para(align=WD_ALIGN_PARAGRAPH.CENTER, before=16, after=0),
        "点击文末「阅读原文」查看全部论文与可点击的 DOI 链接。", size=9, color=GREY)
    run(para(align=WD_ALIGN_PARAGRAPH.CENTER),
        "由自动化文献管线生成；摘要与推荐理由由模型自动生成，可能不完整，请以原文为准。",
        size=8, color=GREY)

    filename = f"{run_date.isoformat()}.docx"
    doc.save(str(WECHAT_DIR / filename))
    return filename

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
    init_clients()
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

        # Website: the HTML digest page (listed in digests.json, shown in the viewer).
        html = format_email_body_html(results, start_day, end_day)
        filename, date_str = save_digest_html(html, report_date)
        # WeChat: the .docx to drag into 文档导入.
        docx_filename = save_wechat_docx(results, start_day, end_day, report_date)

        update_digest_index(date_str, filename, results, start_day, end_day)
        print(f"Success! Website digest: {DIGEST_DIR.name}/{filename}")
        if docx_filename:
            print(f"         WeChat .docx:    {WECHAT_DIR.relative_to(ROOT)}/{docx_filename}  →  标题「{wechat_push_title(start_day)}」")
            print(f"         阅读原文 →        {SITE_URL}/{DIGEST_DIR.name}/{filename}")

    print("\n🎉 All missing digests have been generated and the index is up to date!")

if __name__ == "__main__":
    main()