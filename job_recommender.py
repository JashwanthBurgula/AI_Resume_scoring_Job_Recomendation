"""
Enhanced Job Recommender — India-First Edition
────────────────────────────────────────────────────────────────────────────────
Free APIs (zero keys, zero registration, zero cost — forever):
  1. Jobicy        https://jobicy.com/api/v2/remote-jobs   ?geo=india
  2. Himalayas     https://himalayas.app/jobs/api/search   ?country=India
  3. Remotive      https://remotive.com/api/remote-jobs    ?search=<query>
  4. Adzuna        https://api.adzuna.com/v1/api/jobs/in/  (free key, India endpoint)

Optional paid/freemium APIs (only used when keys are set):
  5. JSearch       https://jsearch.p.rapidapi.com/search   (RapidAPI, 500 req/month free)

Location priority:
  - If user sets a preferred location, jobs are filtered / ranked to prefer it.
  - India is always the primary region even when no location is specified.
  - "Worldwide" / "Anywhere" remote jobs are accepted as India-compatible.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import requests

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.cross_encoder import CrossEncoder
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

# ──────────────────────────── CONFIG ─────────────────────────────────────────

# Optional paid / freemium keys (leave blank → those sources are skipped)
JSEARCH_API_KEY  = os.environ.get("JSEARCH_API_KEY", "")
JSEARCH_HOST     = "jsearch.p.rapidapi.com"
JSEARCH_BASE_URL = "https://jsearch.p.rapidapi.com"

ADZUNA_APP_ID  = os.environ.get("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY", "")

# Sentence-Transformer models
SBERT_MODEL = os.environ.get("SBERT_MODEL", "all-MiniLM-L6-v2")
CE_MODEL    = os.environ.get("CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Local corpus fallback paths
CORPUS_INDEX = os.environ.get("JOBREC_INDEX", "jobrec_index.jsonl")
CORPUS_EMB   = os.environ.get("JOBREC_EMB",   "jobrec_embeddings.npy")
LOCAL_MODEL  = os.environ.get("JOBREC_MODEL",  "jobrec_model")

# ── India detection helpers ───────────────────────────────────────────────────
_INDIA_CITIES = {
    "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad", "chennai",
    "pune", "kolkata", "ahmedabad", "noida", "gurugram", "gurgaon",
    "jaipur", "surat", "lucknow", "kanpur", "nagpur", "indore", "bhopal",
    "chandigarh", "kochi", "coimbatore", "visakhapatnam", "vizag",
    "thiruvananthapuram", "trivandrum", "mysore", "mysuru", "vadodara",
    "baroda", "patna", "agra", "nashik", "faridabad", "meerut", "rajkot",
    "varanasi", "srinagar", "amritsar", "allahabad", "prayagraj",
    "jabalpur", "gwalior", "vijayawada", "jodhpur", "madurai",
    "raipur", "kota", "guwahati", "ranchi", "bhubaneswar",
    "remote india", "work from home india", "india", "in",
}

def _is_india_location(loc: str) -> bool:
    """Return True if the location string refers to India or a city within it."""
    if not loc:
        return False
    low = loc.lower().strip()
    if "india" in low or low in {"in", "ind"}:
        return True
    return any(city in low for city in _INDIA_CITIES)

def _is_worldwide(loc: str) -> bool:
    """Return True for 'worldwide', 'anywhere', 'global' remote tags."""
    low = loc.lower().strip()
    return any(kw in low for kw in (
        "worldwide", "anywhere", "global", "remote", "work from home",
        "wfh", "fully remote", "no restriction", "all countries", ""
    ))

# ──────────────────────────── DATA CLASSES ───────────────────────────────────

@dataclass
class JobResult:
    title:          str
    company:        str
    location:       str
    source:         str          # "jobicy" | "himalayas" | "remotive" | "adzuna" | "jsearch" | "local"
    url:            str
    description:    str
    date_posted:    str
    is_remote:      bool
    job_type:       str          # full_time | internship | contract | part_time
    salary_range:   str
    similarity:     float        # 0-1 cosine / CE score
    skill_match:    float        # % of resume skills matched
    matched_skills: List[str]
    missing_skills: List[str]
    relevance_note: str

# ──────────────────────────── SKILL MATCHING ─────────────────────────────────

from ats_analyzer import find_skills

def _skill_match_score(resume_skills: List[str], jd_text: str) -> Dict:
    jd_sk   = set(find_skills(jd_text))
    res_set = set(resume_skills)
    matched = sorted(res_set & jd_sk)
    missing = sorted(jd_sk - res_set)
    score   = len(matched) / max(1, len(jd_sk)) * 100
    return {"score": round(score, 1), "matched": matched, "missing": missing[:10]}

# ──────────────────────────── FREE API #1 — JOBICY ───────────────────────────
# https://jobicy.com/api/v2/remote-jobs
# No key needed. Supports ?geo=india for India-only jobs.

def fetch_jobicy_jobs(query: str, prefer_india: bool = True, top_k: int = 30) -> List[Dict]:
    """
    Jobicy free API — zero auth, no key, no registration.
    geo=india returns jobs open to Indian candidates.
    """
    params: Dict[str, Any] = {
        "count": min(top_k, 100),
        "tag":   query[:100] if query else "",
    }
    if prefer_india:
        params["geo"] = "india"

    try:
        r = requests.get(
            "https://jobicy.com/api/v2/remote-jobs",
            params=params, timeout=12,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("jobs", [])
    except Exception as e:
        print(f"[Jobicy] error: {e}")
        return []


def parse_jobicy_job(raw: Dict) -> Optional[Dict]:
    try:
        loc = raw.get("jobGeo", "") or "Remote / India"
        return {
            "title":       raw.get("jobTitle", ""),
            "company":     raw.get("companyName", ""),
            "location":    loc,
            "url":         raw.get("url", ""),
            "description": raw.get("jobExcerpt", "") or raw.get("jobDescription", ""),
            "date_posted": (raw.get("pubDate", "") or "")[:10],
            "is_remote":   True,
            "job_type":    (raw.get("jobType", "full-time") or "full-time").replace("-", "_").lower(),
            "salary":      raw.get("annualSalaryMin", "Not specified")
                            and f"${raw['annualSalaryMin']:,}–${raw['annualSalaryMax']:,}/yr"
                            if raw.get("annualSalaryMin") else "Not specified",
            "source":      "jobicy",
        }
    except Exception:
        return None

# ──────────────────────────── FREE API #2 — HIMALAYAS ───────────────────────
# https://himalayas.app/jobs/api/search
# No key needed. country=India to restrict to India-available roles.

def fetch_himalayas_jobs(query: str, prefer_india: bool = True, top_k: int = 20) -> List[Dict]:
    """
    Himalayas free API — no auth, no signup, no limits stated.
    Max 20 per request; paginate if needed.
    """
    params: Dict[str, Any] = {
        "q":    query[:120] if query else "",
        "sort": "relevant",
    }
    if prefer_india:
        params["country"] = "India"

    all_jobs: List[Dict] = []
    for page in range(1, 3):          # up to 2 pages = 40 jobs
        params["page"] = page
        try:
            r = requests.get(
                "https://himalayas.app/jobs/api/search",
                params=params, timeout=12,
            )
            r.raise_for_status()
            batch = r.json().get("jobs", [])
            all_jobs.extend(batch)
            if len(batch) < 20 or len(all_jobs) >= top_k:
                break
            time.sleep(0.4)
        except Exception as e:
            print(f"[Himalayas] error: {e}")
            break

    return all_jobs[:top_k]


def parse_himalayas_job(raw: Dict) -> Optional[Dict]:
    try:
        loc_parts = []
        for r in raw.get("locationRestrictions", []):
            loc_parts.append(r)
        if not loc_parts:
            loc_parts = ["Remote / Worldwide"]
        loc_str = ", ".join(loc_parts[:3])

        sal_min = raw.get("minSalary")
        sal_max = raw.get("maxSalary")
        cur     = raw.get("currency", "USD")
        if sal_min and sal_max:
            salary = f"{cur} {int(sal_min):,}–{int(sal_max):,}/yr"
        elif sal_min:
            salary = f"{cur} {int(sal_min):,}+/yr"
        else:
            salary = "Not specified"

        return {
            "title":       raw.get("title", ""),
            "company":     raw.get("companyName", ""),
            "location":    loc_str,
            "url":         raw.get("applicationLink") or raw.get("url", ""),
            "description": raw.get("description", "") or raw.get("shortDescription", ""),
            "date_posted": (raw.get("postedAt", "") or "")[:10],
            "is_remote":   True,
            "job_type":    (raw.get("employmentType", "Full Time") or "Full Time").replace(" ", "_").lower(),
            "salary":      salary,
            "source":      "himalayas",
        }
    except Exception:
        return None

# ──────────────────────────── FREE API #3 — REMOTIVE ────────────────────────
# https://remotive.com/api/remote-jobs
# No key. ?search= filters title/description. Results include candidate_required_location.

def fetch_remotive_jobs(query: str, prefer_india: bool = True, top_k: int = 30) -> List[Dict]:
    """
    Remotive free API — no auth required.
    We fetch with a search query and then post-filter for India / Worldwide.
    """
    params: Dict[str, Any] = {"search": query[:200] if query else ""}
    try:
        r = requests.get(
            "https://remotive.com/api/remote-jobs",
            params=params, timeout=15,
        )
        r.raise_for_status()
        all_jobs = r.json().get("jobs", [])
    except Exception as e:
        print(f"[Remotive] error: {e}")
        return []

    if not prefer_india:
        return all_jobs[:top_k]

    # Filter: keep jobs open to India or Worldwide
    india_jobs, world_jobs = [], []
    for j in all_jobs:
        req_loc = (j.get("candidate_required_location", "") or "").lower()
        if _is_india_location(req_loc):
            india_jobs.append(j)
        elif _is_worldwide(req_loc) or req_loc == "":
            world_jobs.append(j)

    # Prefer India-specific, then fill with Worldwide
    combined = india_jobs + world_jobs
    return combined[:top_k]


def parse_remotive_job(raw: Dict) -> Optional[Dict]:
    try:
        loc = raw.get("candidate_required_location", "") or "Remote / Worldwide"
        return {
            "title":       raw.get("title", ""),
            "company":     raw.get("company_name", ""),
            "location":    loc,
            "url":         raw.get("url", ""),
            "description": raw.get("description", ""),
            "date_posted": (raw.get("publication_date", "") or "")[:10],
            "is_remote":   True,
            "job_type":    (raw.get("job_type", "full_time") or "full_time").lower(),
            "salary":      raw.get("salary", "Not specified") or "Not specified",
            "source":      "remotive",
        }
    except Exception:
        return None

# ──────────────────────────── OPTIONAL — ADZUNA (free key) ───────────────────
# Free key from https://developer.adzuna.com — 1,000 calls/month.
# We force country="in" (India) always regardless of user input.

def fetch_adzuna_jobs(query: str, top_k: int = 20) -> List[Dict]:
    if not (ADZUNA_APP_ID and ADZUNA_APP_KEY):
        return []
    url = "https://api.adzuna.com/v1/api/jobs/in/search/1"   # "in" = India
    params = {
        "app_id":           ADZUNA_APP_ID,
        "app_key":          ADZUNA_APP_KEY,
        "what":             query,
        "results_per_page": str(top_k),
        "content-type":     "application/json",
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        print(f"[Adzuna-IN] error: {e}")
        return []


def parse_adzuna_job(raw: Dict) -> Optional[Dict]:
    try:
        sal_min = raw.get("salary_min")
        sal_max = raw.get("salary_max")
        salary  = (f"₹{sal_min:,.0f}–₹{sal_max:,.0f}/yr"
                   if sal_min and sal_max else "Not specified")
        loc = raw.get("location", {}).get("display_name", "India")
        # Ensure "India" appears in location string
        if "india" not in loc.lower():
            loc = f"{loc}, India"
        return {
            "title":       raw.get("title", ""),
            "company":     raw.get("company", {}).get("display_name", ""),
            "location":    loc,
            "url":         raw.get("redirect_url", ""),
            "description": raw.get("description", ""),
            "date_posted": (raw.get("created", "") or "")[:10],
            "is_remote":   False,
            "job_type":    (raw.get("contract_type", "") or "full_time").lower(),
            "salary":      salary,
            "source":      "adzuna",
        }
    except Exception:
        return None

# ──────────────────────────── OPTIONAL — JSEARCH (RapidAPI) ──────────────────
# 500 free requests/month. Only used when JSEARCH_API_KEY is set.
# We append "India" to the location to bias results.

def _jsearch_query(query: str, location: str, page: int) -> List[Dict]:
    if not JSEARCH_API_KEY:
        return []
    headers = {
        "X-RapidAPI-Key":  JSEARCH_API_KEY,
        "X-RapidAPI-Host": JSEARCH_HOST,
    }
    # Ensure location always references India
    if location and "india" not in location.lower():
        location = f"{location}, India"
    elif not location:
        location = "India"

    params = {
        "query":            query,
        "page":             str(page),
        "num_pages":        "1",
        "date_posted":      "month",
        "employment_types": "FULLTIME,PARTTIME,CONTRACTOR,INTERN",
        "location":         location,
    }
    try:
        r = requests.get(
            f"{JSEARCH_BASE_URL}/search",
            headers=headers, params=params, timeout=12,
        )
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        print(f"[JSearch] error: {e}")
        return []


def fetch_jsearch_jobs(query: str, location: str = "India", top_k: int = 30) -> List[Dict]:
    results = []
    for page in range(1, 3):
        results += _jsearch_query(query, location, page)
        if len(results) >= top_k:
            break
        time.sleep(0.3)
    return results[:top_k]


def parse_jsearch_job(raw: Dict) -> Optional[Dict]:
    try:
        city    = raw.get("job_city", "") or ""
        state   = raw.get("job_state", "") or ""
        country = raw.get("job_country", "") or ""
        loc_str = ", ".join(filter(None, [city, state, country])) or "India"
        if "india" not in loc_str.lower():
            loc_str = f"{loc_str}, India"

        lo  = raw.get("job_min_salary")
        hi  = raw.get("job_max_salary")
        cur = raw.get("job_salary_currency", "INR")
        per = raw.get("job_salary_period", "YEAR")
        if lo and hi:
            salary = f"{cur} {int(lo):,}–{int(hi):,}/{per.lower()}"
        elif lo:
            salary = f"{cur} {int(lo):,}/{per.lower()}"
        else:
            salary = "Not specified"

        return {
            "title":       raw.get("job_title", ""),
            "company":     raw.get("employer_name", ""),
            "location":    loc_str,
            "url":         raw.get("job_apply_link", ""),
            "description": raw.get("job_description", ""),
            "date_posted": (raw.get("job_posted_at_datetime_utc", "") or "")[:10],
            "is_remote":   bool(raw.get("job_is_remote")),
            "job_type":    (raw.get("job_employment_type") or "").lower(),
            "salary":      salary,
            "source":      "jsearch",
        }
    except Exception:
        return None

# ──────────────────────────── SEMANTIC RANKER ────────────────────────────────

class SemanticRanker:
    def __init__(self):
        self.bi_encoder = None
        self.ce         = None
        if not HAS_SBERT:
            return
        try:
            self.bi_encoder = SentenceTransformer(SBERT_MODEL)
        except Exception as e:
            print(f"[SBERT] bi-encoder load error: {e}")
        try:
            self.ce = CrossEncoder(CE_MODEL)
        except Exception as e:
            print(f"[SBERT] cross-encoder load error: {e}")

    def rank(self, query: str, jobs: List[Dict], top_k: int = 10) -> List[Dict]:
        if not jobs:
            return []
        descs  = [j.get("description", "") or j.get("title", "") for j in jobs]
        scores = np.zeros(len(jobs))

        if self.bi_encoder:
            q_emb  = self.bi_encoder.encode([query], normalize_embeddings=True)[0]
            d_embs = self.bi_encoder.encode(descs, normalize_embeddings=True, batch_size=32)
            scores = d_embs @ q_emb

        if self.ce and len(jobs) <= 60:
            pairs    = [(query, d[:512]) for d in descs]
            ce_raw   = np.array(self.ce.predict(pairs))
            ce_norm  = (ce_raw - ce_raw.min()) / (np.ptp(ce_raw) + 1e-9)
            scores   = 0.5 * scores + 0.5 * ce_norm

        order  = scores.argsort()[::-1]
        ranked = []
        for idx in order[:top_k]:
            j = dict(jobs[idx])
            j["similarity"] = float(round(scores[idx], 4))
            ranked.append(j)
        return ranked

# ──────────────────────────── LOCAL CORPUS FALLBACK ──────────────────────────

class LocalCorpus:
    def __init__(self):
        self.docs, self.embs = [], None
        if not (os.path.isfile(CORPUS_INDEX) and os.path.isfile(CORPUS_EMB)):
            return
        with open(CORPUS_INDEX, encoding="utf-8") as f:
            self.docs = [json.loads(l) for l in f if l.strip()]
        self.embs = np.load(CORPUS_EMB).astype(np.float32)
        print(f"[LocalCorpus] loaded {len(self.docs)} docs")

    def search(self, query_emb: np.ndarray, top_k: int = 20) -> List[Dict]:
        if self.embs is None or not self.docs:
            return []
        sims  = self.embs @ query_emb
        order = sims.argsort()[::-1][:top_k]
        results = []
        for i in order:
            d = dict(self.docs[i])
            d.update({"similarity": float(round(sims[i], 4)), "source": "local"})
            d.setdefault("url", "")
            d.setdefault("date_posted", "")
            d.setdefault("is_remote", False)
            d.setdefault("job_type", "full_time")
            d.setdefault("salary", "Not specified")
            results.append(d)
        return results

# ──────────────────────────── LOCATION SCORE BOOST ───────────────────────────

def _location_score(job_loc: str, preferred_loc: str) -> float:
    """
    Returns a 0.0–1.0 bonus to be added to ranking score.
    1.0  → job is in the exact preferred city/state
    0.7  → job is in India (any city)
    0.4  → job is worldwide / remote (acceptable)
    0.0  → job is in a foreign country (not India, not worldwide)
    """
    jl = (job_loc or "").lower()

    # Prefer exact preferred location match
    if preferred_loc:
        pl = preferred_loc.lower()
        if pl in jl:
            return 1.0

    # India-generic match
    if _is_india_location(jl):
        return 0.7

    # Worldwide remote (acceptable for India-based candidates)
    if _is_worldwide(jl):
        return 0.4

    # Foreign location penalty
    return 0.0

# ──────────────────────────── MAIN RECOMMENDER ───────────────────────────────

class JobRecommender:
    def __init__(self):
        self.ranker = SemanticRanker()
        self.corpus = LocalCorpus()

    def recommend(
        self,
        resume_text:          str,
        job_query:            str  = "",
        location:             str  = "",
        include_internships:  bool = True,
        top_k:                int  = 10,
    ) -> List[JobResult]:

        resume_skills = find_skills(resume_text)

        # ── Build search query ────────────────────────────────────────────────
        top_skills = " ".join(resume_skills[:5]) if resume_skills else "software engineer"
        query      = job_query.strip() or top_skills

        # ── Determine India preference ────────────────────────────────────────
        # Default to India unless the user explicitly specifies a foreign country
        prefer_india = True
        if location:
            pl = location.lower()
            # If they gave a foreign location explicitly, respect it
            if (
                not _is_india_location(pl)
                and not _is_worldwide(pl)
                and len(pl) > 2            # ignore short codes like "in"
            ):
                prefer_india = False

        raw_jobs: List[Dict] = []

        # ── SOURCE 1: Jobicy (free, no key) ──────────────────────────────────
        print("[Jobicy] fetching…")
        for raw in fetch_jobicy_jobs(query, prefer_india=prefer_india, top_k=40):
            j = parse_jobicy_job(raw)
            if j:
                raw_jobs.append(j)

        # ── SOURCE 2: Himalayas (free, no key) ───────────────────────────────
        print("[Himalayas] fetching…")
        for raw in fetch_himalayas_jobs(query, prefer_india=prefer_india, top_k=40):
            j = parse_himalayas_job(raw)
            if j:
                raw_jobs.append(j)

        # ── SOURCE 3: Remotive (free, no key) ────────────────────────────────
        print("[Remotive] fetching…")
        for raw in fetch_remotive_jobs(query, prefer_india=prefer_india, top_k=40):
            j = parse_remotive_job(raw)
            if j:
                raw_jobs.append(j)

        # ── SOURCE 4: Adzuna India (free key) ────────────────────────────────
        if ADZUNA_APP_ID and ADZUNA_APP_KEY:
            print("[Adzuna-IN] fetching…")
            for raw in fetch_adzuna_jobs(query, top_k=25):
                j = parse_adzuna_job(raw)
                if j:
                    raw_jobs.append(j)

        # ── SOURCE 5: JSearch (optional, RapidAPI key) ───────────────────────
        if JSEARCH_API_KEY:
            jsearch_loc = location if location else "India"
            print(f"[JSearch] fetching for location='{jsearch_loc}'…")
            for raw in fetch_jsearch_jobs(query, location=jsearch_loc, top_k=30):
                j = parse_jsearch_job(raw)
                if j:
                    raw_jobs.append(j)

        # ── Local corpus fallback ─────────────────────────────────────────────
        if not raw_jobs and self.corpus.embs is not None and self.ranker.bi_encoder:
            print("[LocalCorpus] using local fallback…")
            q_emb    = self.ranker.bi_encoder.encode(
                [resume_text[:800] + "\n" + query], normalize_embeddings=True
            )[0]
            raw_jobs = self.corpus.search(q_emb, top_k=40)

        if not raw_jobs:
            print("[JobRecommender] No jobs found from any source.")
            return []

        # ── Deduplicate by URL ────────────────────────────────────────────────
        seen_urls:  set = set()
        deduped:    List[Dict] = []
        for j in raw_jobs:
            url = j.get("url", "")
            key = url if url else f"{j.get('title','')}__{j.get('company','')}"
            if key and key in seen_urls:
                continue
            seen_urls.add(key)
            deduped.append(j)

        # ── Semantic reranking ────────────────────────────────────────────────
        full_query = resume_text[:800] + "\n\n" + query
        ranked     = self.ranker.rank(full_query, deduped, top_k=top_k * 3)

        # ── Location-aware scoring + final assembly ───────────────────────────
        results: List[JobResult] = []
        for j in ranked:
            sm = _skill_match_score(
                resume_skills,
                (j.get("description", "") + " " + j.get("title", "")).strip(),
            )

            jtype = j.get("job_type", "").lower()

            # Relevance note
            if sm["score"] >= 70:
                note = "Strong match — apply now!"
            elif sm["score"] >= 40:
                note = "Good match — tailor your resume before applying."
            else:
                note = "Partial match — consider upskilling in missing areas."

            results.append(JobResult(
                title          = j.get("title", ""),
                company        = j.get("company", ""),
                location       = j.get("location", "").strip(", "),
                source         = j.get("source", ""),
                url            = j.get("url", ""),
                description    = (j.get("description", "")[:600] + "…")
                                  if len(j.get("description", "")) > 600
                                  else j.get("description", ""),
                date_posted    = j.get("date_posted", ""),
                is_remote      = bool(j.get("is_remote", False)),
                job_type       = jtype,
                salary_range   = j.get("salary", "Not specified"),
                similarity     = j.get("similarity", 0.0),
                skill_match    = sm["score"],
                matched_skills = sm["matched"],
                missing_skills = sm["missing"],
                relevance_note = note,
            ))
            if len(results) >= top_k * 2:
                break

        # ── Sort: prefer India matches, then skill+similarity ─────────────────
        results.sort(
            key=lambda x: (
                _location_score(x.location, location),   # 0–1 India bonus
                x.skill_match,                            # 0–100
                x.similarity * 50,                        # 0–50
            ),
            reverse=True,
        )
        return results[:top_k]


# ──────────────────────────── CLI ────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from ats_analyzer import extract_pdf_text

    ap = argparse.ArgumentParser(description="India-first Job Recommender")
    ap.add_argument("--resume",       required=True,  help="Path to resume PDF")
    ap.add_argument("--query",        default="",     help="Job search query")
    ap.add_argument("--location",     default="",     help="Preferred location (e.g. 'Bangalore', 'Mumbai')")
    ap.add_argument("--top_k",        type=int, default=10)
    ap.add_argument("--internships",  action="store_true")
    args = ap.parse_args()

    resume_text = extract_pdf_text(args.resume)
    rec  = JobRecommender()
    jobs = rec.recommend(
        resume_text, args.query, args.location, args.internships, args.top_k
    )

    print(f"\n=== TOP {len(jobs)} JOB RECOMMENDATIONS (India-first) ===\n")
    for i, j in enumerate(jobs, 1):
        print(f"{i}. {j.title} @ {j.company}  [{j.location}]")
        print(f"   Source: {j.source} | Type: {j.job_type} | Remote: {j.is_remote}")
        print(f"   Skill match: {j.skill_match}% | Similarity: {j.similarity:.3f}")
        print(f"   Salary: {j.salary_range} | Posted: {j.date_posted}")
        print(f"   {j.relevance_note}")
        if j.url:
            print(f"   Apply: {j.url}")
        print()
