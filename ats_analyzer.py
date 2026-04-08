"""
Enhanced ATS Resume Analyzer
- Best-in-class ATS scoring formula (multi-signal weighted)
- Deep keyword/skill gap analysis
- Intelligent rewrite suggestions
- Role recommendation engine
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import PyPDF2
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# ─────────────── CONFIG ───────────────
PHRASING_MODEL_DIR = os.environ.get("PHRASING_MODEL_DIR", "phrasing_model")
ATS_MODEL_PATH     = os.environ.get("ATS_MODEL_PATH",     "ats_model.joblib")
ATS_VEC_PATH       = os.environ.get("ATS_VECTORIZER_PATH","ats_vectorizer.joblib")
DEFAULT_REPORT_DIR = os.environ.get("ATS_REPORT_DIR",     "reports")

# ─────────────── SKILL TAXONOMY ───────────────
SKILLS: Dict[str, List[str]] = {
    "programming": [
        "python","java","c++","c","javascript","typescript","golang","rust","scala",
        "kotlin","swift","r","matlab","perl","php","ruby","sql","bash","shell",
    ],
    "ml_ai": [
        "machine learning","deep learning","nlp","natural language processing",
        "computer vision","reinforcement learning","transformers","llm","bert","gpt",
        "t5","pytorch","torch","tensorflow","keras","scikit-learn","xgboost",
        "lightgbm","catboost","huggingface","diffusion models","rag","langchain",
        "llama","openai api","stable diffusion","object detection","yolo",
        "neural network","convolutional neural network","recurrent neural network",
        "lstm","attention mechanism","fine-tuning","transfer learning","embedding",
    ],
    "data": [
        "pandas","numpy","matplotlib","seaborn","plotly","data analysis",
        "data visualization","statistics","etl","data pipeline","data wrangling",
        "feature engineering","a/b testing","hypothesis testing","sql","nosql",
        "data modeling","data warehousing","dbt","apache spark","pyspark","hadoop",
    ],
    "backend": [
        "rest api","fastapi","flask","django","spring boot","node.js","express",
        "graphql","microservices","jwt","oauth","grpc","websocket","celery",
        "message queue","kafka","rabbitmq","redis","nginx","gunicorn",
    ],
    "frontend": [
        "react","vue","angular","next.js","nuxt","html","css","tailwind",
        "bootstrap","typescript","webpack","vite","figma","ui/ux",
    ],
    "cloud_devops": [
        "aws","azure","gcp","docker","kubernetes","linux","git","github",
        "gitlab","ci/cd","terraform","ansible","jenkins","github actions",
        "prometheus","grafana","helm","argo","cloud functions","lambda",
        "s3","ec2","sagemaker","vertex ai","databricks",
    ],
    "databases": [
        "mysql","postgresql","mongodb","sqlite","redis","elasticsearch",
        "cassandra","dynamodb","bigquery","snowflake","clickhouse","neo4j",
    ],
    "security": [
        "cybersecurity","owasp","threat modeling","penetration testing",
        "cryptography","soc","siem","vulnerability assessment","iam","zero trust",
    ],
    "soft_skills": [
        "agile","scrum","jira","confluence","leadership","communication",
        "problem solving","teamwork","project management","mentoring",
    ],
}

# Flat lookup for fast matching
ALL_SKILLS: List[str] = [s for grp in SKILLS.values() for s in grp]

JOB_CATALOG: List[Dict] = [
    {"role":"Data Scientist (Entry)",    "skills":["python","pandas","numpy","scikit-learn","statistics","data analysis","sql","machine learning"]},
    {"role":"ML Engineer (Entry)",       "skills":["python","pytorch","tensorflow","machine learning","deep learning","docker","aws","git","ci/cd"]},
    {"role":"NLP Engineer",              "skills":["python","nlp","transformers","bert","t5","huggingface","pytorch","llm","langchain"]},
    {"role":"Data Analyst",              "skills":["sql","data analysis","data visualization","python","pandas","statistics","tableau","power bi"]},
    {"role":"Backend Developer",         "skills":["rest api","python","fastapi","flask","sql","docker","git","postgresql","redis"]},
    {"role":"Cloud/DevOps Engineer",     "skills":["aws","docker","kubernetes","linux","ci/cd","git","terraform","github actions"]},
    {"role":"AI/ML Research Engineer",   "skills":["python","pytorch","tensorflow","deep learning","transformers","llm","fine-tuning","embedding"]},
    {"role":"Data Engineer",             "skills":["python","sql","apache spark","etl","data pipeline","aws","databricks","airflow","snowflake"]},
    {"role":"Full Stack Developer",      "skills":["react","node.js","python","javascript","typescript","postgresql","docker","rest api"]},
    {"role":"Computer Vision Engineer",  "skills":["python","computer vision","pytorch","yolo","opencv","deep learning","tensorflow","c++"]},
    {"role":"Security Analyst",          "skills":["cybersecurity","owasp","penetration testing","linux","python","siem","vulnerability assessment"]},
    {"role":"BI / Analytics Engineer",   "skills":["sql","tableau","power bi","python","pandas","data visualization","etl","statistics"]},
]

ACTION_VERBS = [
    "developed","implemented","designed","engineered","optimized","built",
    "led","created","improved","deployed","automated","architected","delivered",
    "reduced","increased","scaled","migrated","integrated","launched","achieved",
    "collaborated","mentored","trained","analyzed","established","streamlined",
]

WEAK_PHRASES = {
    "worked on":         "developed",
    "helped with":       "contributed to",
    "responsible for":   "owned",
    "handled":           "executed",
    "participated in":   "contributed to",
    "was involved in":   "drove",
    "assisted in":       "supported",
    "tried to":          "successfully",
    "basically":         "",
    "just":              "",
}

SECTION_HEADERS = re.compile(
    r"(experience|education|skills|projects|certifications|achievements|"
    r"summary|objective|work history|internship|publications|awards)",
    re.IGNORECASE,
)

# ─────────────── DATA CLASSES ───────────────
@dataclass
class RewriteSuggestion:
    original:  str
    suggested: str
    reason:    str

@dataclass
class SectionScore:
    name:    str
    score:   float
    details: str

@dataclass
class AnalysisReport:
    resume_path:              str
    generated_at:             str
    ats_score:                float
    similarity_score:         float
    score_breakdown:          Dict[str, float]
    section_scores:           List[SectionScore]
    found_skills_resume:      List[str]
    found_skills_job:         List[str]
    missing_skills_for_job:   List[str]
    missing_keywords_for_ats: List[str]
    mismatched_keywords:      List[str]
    extra_skills_not_in_jd:   List[str]
    rewrite_suggestions:      List[RewriteSuggestion]
    recommended_roles:        List[Dict]
    resume_word_count:        int
    has_quantified_impact:    bool
    action_verb_density:      float
    keyword_density:          float
    improvement_tips:         List[str]

# ─────────────── TEXT UTILITIES ───────────────
def _norm(s: str) -> str:
    s = s.replace("\u00ad", "").replace("\u2019", "'")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def extract_pdf_text(path: str) -> str:
    reader = PyPDF2.PdfReader(path)
    pages  = [p.extract_text() or "" for p in reader.pages]
    return _norm("\n".join(pages))

def split_bullets(text: str) -> List[str]:
    lines = re.split(r"[\n\r]+", text)
    out   = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        ln = re.sub(r"^[\-\*\u2022\u25cf\u25aa\u2013\u2014•]+\s*", "", ln).strip()
        if len(ln.split()) >= 5:
            out.append(ln)
    return out

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]

def detect_sections(text: str) -> Dict[str, str]:
    """Split resume text into named sections."""
    sections: Dict[str, str] = {}
    current = "header"
    buf: List[str] = []
    for line in text.splitlines():
        m = SECTION_HEADERS.match(line.strip())
        if m and len(line.strip()) < 50:
            sections[current] = "\n".join(buf)
            current = line.strip().lower()
            buf = []
        else:
            buf.append(line)
    sections[current] = "\n".join(buf)
    return sections

# ─────────────── KEYWORD / SKILL EXTRACTION ───────────────
def find_skills(text: str) -> List[str]:
    t = text.lower()
    found = []
    for skill in ALL_SKILLS:
        if " " in skill or "/" in skill or "." in skill:
            if skill in t:
                found.append(skill)
        else:
            if re.search(rf"\b{re.escape(skill)}\b", t):
                found.append(skill)
    return sorted(set(found))

def extract_tfidf_keywords(text: str, top_n: int = 40) -> List[str]:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return []
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=3000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\+\#\./-]{1,}\b",
    )
    X  = vec.fit_transform([t])
    w  = X.toarray().ravel()
    fs = vec.get_feature_names_out()
    order = w.argsort()[::-1]
    seen, out = set(), []
    for i in order[:top_n]:
        k = fs[i].strip().lower()
        if w[i] > 0 and k and k not in seen and len(k) > 2:
            seen.add(k)
            out.append(k)
    return out

def missing_keywords(resume: str, job: str, limit: int = 25) -> List[str]:
    job_sk  = find_skills(job)
    job_kws = extract_tfidf_keywords(job, top_n=50)
    res_l   = resume.lower()

    merged, seen = [], set()
    for k in job_sk + job_kws:
        if k not in seen and k not in res_l:
            seen.add(k)
            merged.append(k)
        if len(merged) >= limit:
            break
    return merged

def mismatched_kws(resume: str, job: str, limit: int = 20) -> List[str]:
    """Resume-prominent keywords absent from the JD (may dilute relevance)."""
    res_kws = extract_tfidf_keywords(resume, top_n=60)
    job_set = set(extract_tfidf_keywords(job, top_n=60)) | set(find_skills(job))
    return [k for k in res_kws if k not in job_set and len(k) > 2][:limit]

def extra_skills_vs_jd(resume: str, job: str) -> List[str]:
    return sorted(set(find_skills(resume)) - set(find_skills(job)))

# ─────────────── QUALITY SIGNALS ───────────────
def action_verb_density(text: str) -> float:
    sents = split_sentences(text) or [text]
    hits  = sum(1 for s in sents if any(v in s.lower() for v in ACTION_VERBS))
    return round(hits / max(1, len(sents)), 4)

def has_quantified_impact(text: str) -> bool:
    return bool(re.search(r"\b\d+(\.\d+)?[\s]?(%|x|times|users|records|ms|gb|tb|k\b|million)\b", text, re.I))

def keyword_density_score(resume: str, job: str) -> float:
    jd_kws = extract_tfidf_keywords(job, top_n=30)
    if not jd_kws:
        return 0.0
    res_l  = resume.lower()
    hits   = sum(1 for k in jd_kws if k in res_l)
    return round((hits / len(jd_kws)) * 100, 2)

def section_completeness(text: str) -> Tuple[List[SectionScore], float]:
    important = {
        "experience":       0.30,
        "education":        0.20,
        "skills":           0.20,
        "projects":         0.15,
        "certifications":   0.05,
        "summary":          0.10,
    }
    tl = text.lower()
    scores, total = [], 0.0
    for sec, weight in important.items():
        present = bool(re.search(rf"\b{sec}\b", tl))
        s = 100.0 if present else 0.0
        total += s * weight
        scores.append(SectionScore(
            name=sec,
            score=s,
            details="Present" if present else "Missing — add this section",
        ))
    return scores, round(total, 2)

# ─────────────── ATS SCORING ───────────────
class AtsScorer:
    """Optional trained ML prior; graceful fallback if models absent."""
    def __init__(self, model_path: str, vec_path: str):
        self.available = False
        if os.path.exists(model_path) and os.path.exists(vec_path):
            try:
                self.model = joblib.load(model_path)
                self.vec   = joblib.load(vec_path)
                self.available = True
            except Exception:
                pass

    def cosine_sim(self, a: str, b: str) -> Optional[float]:
        if not self.available:
            return None
        vs = self.vec.transform([a, b])
        return float(cosine_similarity(vs[0], vs[1])[0][0] * 100)

    def ml_score(self, resume: str) -> Optional[float]:
        if not self.available:
            return None
        X = self.vec.transform([resume])
        return float(np.clip(self.model.predict(X).ravel()[0], 0, 100))


def compute_ats_score(
    resume: str,
    job:    str,
    scorer: AtsScorer,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Returns (final_score, similarity, breakdown_dict).

    Multi-signal ATS scoring formula (weights tuned for recruiter-grade ATS):
      • Cosine similarity (TF-IDF)    35 %
      • Skill coverage                25 %
      • Keyword coverage (adaptive)   20 %
      • Section completeness          10 %
      • Writing quality (actions+metrics) 10 %
      (+ optional ML prior replaces 10 % if available)
    """
    sim = scorer.cosine_sim(resume, job)

    res_sk = set(find_skills(resume))
    job_sk = set(find_skills(job))
    skill_cov = (len(res_sk & job_sk) / max(1, len(job_sk))) * 100

    jd_kws   = extract_tfidf_keywords(job, top_n=40)
    res_l    = resume.lower()
    kw_cov   = (sum(1 for k in jd_kws if k and k in res_l) / max(1, len(jd_kws))) * 100

    _, sec_score = section_completeness(resume)

    sents   = split_sentences(resume)
    av_hits = sum(1 for s in sents if any(v in s.lower() for v in ACTION_VERBS))
    met_hits= sum(1 for s in sents if re.search(r"\b\d+(\.\d+)?%?\b", s))
    av_sc   = min(100, (av_hits / max(1, len(sents))) * 120)
    mt_sc   = min(100, (met_hits / max(1, len(sents))) * 120)
    quality = (av_sc + mt_sc) / 2

    if sim is None:
        sim = (skill_cov * 0.6 + kw_cov * 0.4)

    ml = scorer.ml_score(resume)

    if ml is None:
        final = (
            0.35 * sim
          + 0.25 * skill_cov
          + 0.20 * kw_cov
          + 0.10 * sec_score
          + 0.10 * quality
        )
    else:
        final = (
            0.30 * sim
          + 0.22 * skill_cov
          + 0.18 * kw_cov
          + 0.10 * sec_score
          + 0.10 * quality
          + 0.10 * ml
        )

    breakdown = {
        "cosine_similarity":   round(sim, 2),
        "skill_coverage":      round(skill_cov, 2),
        "keyword_coverage":    round(kw_cov, 2),
        "section_completeness":round(sec_score, 2),
        "writing_quality":     round(quality, 2),
    }
    if ml is not None:
        breakdown["ml_prior"] = round(ml, 2)

    return round(float(np.clip(final, 0, 100)), 2), round(float(sim), 2), breakdown

# ─────────────── REWRITE SUGGESTIONS ───────────────
class PhraseModel:
    def __init__(self, model_dir: str):
        self.available = False
        if not HAS_TORCH or not os.path.isdir(model_dir):
            return
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.model     = T5ForConditionalGeneration.from_pretrained(model_dir).to(self.device)
        self.available = True

    def improve(self, bullet: str, max_len: int = 72) -> Optional[str]:
        if not self.available:
            return None
        import torch
        prompt = "improve resume bullet: " + bullet.strip()
        enc  = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                               padding="max_length", max_length=64)
        ids  = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        out  = self.model.generate(ids, attention_mask=mask, max_length=max_len,
                                   num_beams=4, no_repeat_ngram_size=3)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip() or None


def _looks_sane(s: str) -> bool:
    w = s.split()
    if len(w) < 5 or len(s) > 250:
        return False
    toks = re.findall(r"[a-zA-Z]+", s.lower())
    return (len(set(toks)) / max(1, len(toks))) >= 0.55


def generate_rewrites(
    bullets:  List[str],
    miss_kws: List[str],
    pm:       PhraseModel,
    limit:    int = 15,
) -> List[RewriteSuggestion]:
    suggs = []
    miss_set = set(miss_kws)

    for b in bullets:
        if len(suggs) >= limit:
            break
        b_low  = b.lower()
        parts  = []
        revised = b

        for weak, strong in WEAK_PHRASES.items():
            if weak in b_low:
                revised = re.sub(re.escape(weak), strong, revised, flags=re.IGNORECASE).strip()
                if strong:
                    parts.append(f"replace '{weak}' → '{strong}'")
                else:
                    parts.append(f"remove filler word '{weak}'")

        if not re.search(r"\b\d+(\.\d+)?%?\b", revised):
            if any(v in revised.lower() for v in ACTION_VERBS):
                revised = revised.rstrip(".") + " (quantify: add %, count, or time saved)."
                parts.append("add measurable metric")

        injectables = [k for k in ["python","sql","pytorch","tensorflow","aws","docker","kubernetes"]
                       if k in miss_set and k not in revised.lower()]
        if injectables and any(v in revised.lower() for v in ACTION_VERBS):
            kw = injectables[0]
            revised = revised.rstrip(".") + f" (if applicable, mention: {kw})."
            parts.append(f"consider adding keyword '{kw}'")

        t5_out = pm.improve(b) if pm.available else None
        if t5_out and _looks_sane(t5_out):
            revised = t5_out
            parts   = ["T5 rewrite for clarity/impact"] + parts

        if revised != b:
            suggs.append(RewriteSuggestion(
                original=b,
                suggested=revised,
                reason=", ".join(parts) or "improve ATS phrasing",
            ))
    return suggs

# ─────────────── ROLE RECOMMENDATION ───────────────
def recommend_roles(resume: str, job: str, top_k: int = 6) -> List[Dict]:
    res_sk = set(find_skills(resume))
    job_sk = set(find_skills(job))
    recs   = []
    for item in JOB_CATALOG:
        role_sk = set(item["skills"])
        ov_res  = len(res_sk & role_sk)
        ov_job  = len(job_sk & role_sk)
        denom   = max(1, len(role_sk))
        score   = 0.65 * (ov_res / denom) + 0.35 * (ov_job / denom)
        recs.append({
            "role":           item["role"],
            "match_percent":  round(score * 100, 1),
            "matched_skills": sorted((res_sk | job_sk) & role_sk),
            "missing_skills": sorted(role_sk - res_sk),
        })
    recs.sort(key=lambda x: x["match_percent"], reverse=True)
    return recs[:top_k]

# ─────────────── IMPROVEMENT TIPS ───────────────
def generate_tips(report_data: Dict) -> List[str]:
    tips = []
    score = report_data.get("ats_score", 0)
    if score < 40:
        tips.append("Your ATS score is low. Tailor this resume specifically to the job description.")
    if not report_data.get("has_quantified_impact"):
        tips.append("Add quantified achievements (%, numbers, scale) to stand out — e.g. 'reduced latency by 40%'.")
    if report_data.get("action_verb_density", 0) < 0.4:
        tips.append("Start more bullet points with strong action verbs: developed, engineered, optimized, led, deployed.")
    miss = report_data.get("missing_skills_for_job", [])
    if miss:
        top3 = miss[:3]
        tips.append(f"Priority skills to add or learn: {', '.join(top3)}.")
    breakdown = report_data.get("score_breakdown", {})
    if breakdown.get("section_completeness", 100) < 70:
        tips.append("Add missing resume sections (summary, skills, certifications) to improve ATS parsing.")
    if breakdown.get("keyword_coverage", 100) < 50:
        tips.append("Mirror more language from the job description in your resume — ATS systems look for exact matches.")
    wc = report_data.get("resume_word_count", 500)
    if wc < 300:
        tips.append("Your resume seems short. Aim for 400–600 words for junior roles, 600–900 for senior roles.")
    if wc > 900:
        tips.append("Resume may be too long. Trim to 1–2 pages; focus on impact over exhaustive history.")
    if not tips:
        tips.append("Great resume! Ensure formatting is clean for ATS parsing and proofread one more time.")
    return tips

# ─────────────── MAIN ANALYSIS ───────────────
def analyze(
    resume_pdf_path: str,
    job_description: str,
    use_t5:          bool = False,
) -> AnalysisReport:
    resume_text = extract_pdf_text(resume_pdf_path)
    bullets     = split_bullets(resume_text)
    pm          = PhraseModel(PHRASING_MODEL_DIR) if use_t5 else PhraseModel("__none__")
    scorer      = AtsScorer(ATS_MODEL_PATH, ATS_VEC_PATH)

    ats, sim, breakdown = compute_ats_score(resume_text, job_description, scorer)
    sec_scores, _       = section_completeness(resume_text)
    res_sk  = find_skills(resume_text)
    job_sk  = find_skills(job_description)
    miss_sk = sorted(set(job_sk) - set(res_sk))
    miss_kw = missing_keywords(resume_text, job_description)
    mm_kw   = mismatched_kws(resume_text, job_description)
    extra   = extra_skills_vs_jd(resume_text, job_description)
    rewrites= generate_rewrites(bullets, miss_kw, pm)
    roles   = recommend_roles(resume_text, job_description)
    av_d    = action_verb_density(resume_text)
    quant   = has_quantified_impact(resume_text)
    kd      = keyword_density_score(resume_text, job_description)
    wc      = len(resume_text.split())

    partial = dict(
        ats_score=ats, score_breakdown=breakdown,
        missing_skills_for_job=miss_sk,
        has_quantified_impact=quant,
        action_verb_density=av_d,
        resume_word_count=wc,
    )
    tips = generate_tips(partial)

    return AnalysisReport(
        resume_path              = resume_pdf_path,
        generated_at             = datetime.now().isoformat(timespec="seconds"),
        ats_score                = ats,
        similarity_score         = sim,
        score_breakdown          = breakdown,
        section_scores           = sec_scores,
        found_skills_resume      = res_sk,
        found_skills_job         = job_sk,
        missing_skills_for_job   = miss_sk,
        missing_keywords_for_ats = miss_kw,
        mismatched_keywords      = mm_kw,
        extra_skills_not_in_jd   = extra,
        rewrite_suggestions      = rewrites,
        recommended_roles        = roles,
        resume_word_count        = wc,
        has_quantified_impact    = quant,
        action_verb_density      = av_d,
        keyword_density          = kd,
        improvement_tips         = tips,
    )

# ─────────────── REPORT SAVING ───────────────
def save_report(report: AnalysisReport, out_dir: str = DEFAULT_REPORT_DIR) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(out_dir, f"ats_report_{stamp}.json")
    tpath = os.path.join(out_dir, f"ats_report_{stamp}.txt")

    payload = asdict(report)
    payload["rewrite_suggestions"] = [asdict(r) for r in report.rewrite_suggestions]
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = [
        "=" * 55,
        "   ATS RESUME ASSISTANT — FULL ANALYSIS REPORT",
        "=" * 55,
        f"Generated : {report.generated_at}",
        f"Resume    : {report.resume_path}",
        f"Word count: {report.resume_word_count}",
        "",
        f"  ATS Score         : {report.ats_score} / 100",
        f"  Similarity        : {report.similarity_score}",
        f"  Has quantified ?  : {'Yes' if report.has_quantified_impact else 'No'}",
        f"  Action-verb density: {report.action_verb_density:.0%}",
        f"  Keyword density   : {report.keyword_density}%",
        "",
        "─── Score Breakdown ──────────────────────────────",
    ]
    for k, v in report.score_breakdown.items():
        lines.append(f"  {k:<28}: {v}")
    lines += [
        "",
        "─── Section Completeness ─────────────────────────",
    ]
    for s in report.section_scores:
        lines.append(f"  {s.name:<20}: {s.details}")
    lines += [
        "",
        "─── Skills in Resume ─────────────────────────────",
        (", ".join(report.found_skills_resume) or "(none detected)"),
        "",
        "─── Skills in Job Description ────────────────────",
        (", ".join(report.found_skills_job) or "(none)"),
        "",
        "─── Skills to Learn (Job Gap) ────────────────────",
        (", ".join(report.missing_skills_for_job) or "(none)"),
        "",
        "─── Missing ATS Keywords ─────────────────────────",
        (", ".join(report.missing_keywords_for_ats) or "(none)"),
        "",
        "─── Keywords in Resume Not in JD ─────────────────",
        (", ".join(report.mismatched_keywords) or "(none)"),
        "",
        "─── Rewrite Suggestions ──────────────────────────",
    ]
    for i, s in enumerate(report.rewrite_suggestions, 1):
        lines += [
            f"{i}. [{s.reason}]",
            f"   Before: {s.original}",
            f"   After : {s.suggested}",
            "",
        ]
    lines += ["─── Improvement Tips ─────────────────────────────"]
    for tip in report.improvement_tips:
        lines.append(f"  • {tip}")
    lines += [
        "",
        "─── Recommended Roles ────────────────────────────",
    ]
    for r in report.recommended_roles:
        lines += [
            f"  {r['role']}  ({r['match_percent']}% match)",
            f"    Matched : {', '.join(r['matched_skills']) or '—'}",
            f"    Missing : {', '.join(r['missing_skills']) or '—'}",
            "",
        ]

    with open(tpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return tpath, jpath


if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser(description="ATS Resume Analyzer CLI")
    ap.add_argument("--resume",      required=True,  help="Path to resume PDF")
    ap.add_argument("--job",         required=True,  help="Job description text or .txt file path")
    ap.add_argument("--out",         default=DEFAULT_REPORT_DIR)
    ap.add_argument("--use_t5",      action="store_true")
    args = ap.parse_args()

    job_text = open(args.job, encoding="utf-8").read() if os.path.isfile(args.job) else args.job
    rpt  = analyze(args.resume, job_text, use_t5=args.use_t5)
    tp, jp = save_report(rpt, args.out)
    print(f"\nATS Score : {rpt.ats_score} / 100")
    print(f"Reports   : {tp}\n            {jp}")
