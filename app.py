"""
ATS Resume Assistant — Streamlit Web App
Run with:  streamlit run app.py
"""

import os
import io
import json
import tempfile

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="ATS Resume Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

  .main { background: #0D0F1A; }
  .block-container { padding-top: 2rem; max-width: 1200px; }

  /* Score ring card */
  .score-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #12142a 100%);
    border: 1px solid #2e3258;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(99,102,241,0.15);
  }
  .score-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 4.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8 0%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
  }
  .score-label { color: #94a3b8; font-size: 0.85rem; margin-top: 0.4rem; letter-spacing: 0.1em; text-transform: uppercase; }

  /* Skill chip */
  .chip {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px;
    letter-spacing: 0.02em;
  }
  .chip-green  { background: #052e16; color: #4ade80; border: 1px solid #166534; }
  .chip-red    { background: #2d0a0a; color: #f87171; border: 1px solid #7f1d1d; }
  .chip-blue   { background: #0c1a3a; color: #60a5fa; border: 1px solid #1e3a6e; }
  .chip-yellow { background: #2d1e00; color: #fbbf24; border: 1px solid #78350f; }

  /* Tip card */
  .tip-card {
    background: #12142a;
    border-left: 3px solid #818cf8;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #cbd5e1;
    font-size: 0.9rem;
  }

  /* Job card */
  .job-card {
    background: #12142a;
    border: 1px solid #1e2147;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
    transition: border-color 0.2s;
  }
  .job-card:hover { border-color: #818cf8; }
  .job-title  { font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 600; color: #e2e8f0; }
  .job-meta   { font-size: 0.82rem; color: #64748b; margin-top: 2px; }
  .match-bar-bg { background: #1e2147; border-radius: 99px; height: 6px; margin: 8px 0; }
  .match-bar    { background: linear-gradient(90deg, #818cf8, #38bdf8); border-radius: 99px; height: 6px; }

  /* Section header */
  .section-hdr {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: #e2e8f0;
    border-bottom: 1px solid #1e2147;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
  }

  /* Rewrite card */
  .rw-original { background: #2d0a0a; border-radius: 8px; padding: 0.6rem 0.9rem; font-size: 0.85rem; color: #fca5a5; font-style: italic; }
  .rw-suggested { background: #052e16; border-radius: 8px; padding: 0.6rem 0.9rem; font-size: 0.85rem; color: #86efac; margin-top: 4px; }
  .rw-reason  { font-size: 0.75rem; color: #818cf8; font-weight: 500; margin-top: 4px; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #0a0c18 !important; }
  [data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

  /* Upload area */
  [data-testid="stFileUploader"] { background: #0d0f1a; border: 2px dashed #2e3258; border-radius: 12px; }

  /* Buttons */
  .stButton button {
    background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
  }
  .stButton button:hover { opacity: 0.88 !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab"] { color: #64748b; font-weight: 500; }
  .stTabs [aria-selected="true"] { color: #818cf8 !important; border-bottom-color: #818cf8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Lazy imports (so app starts even if optional deps missing) ──
@st.cache_resource(show_spinner=False)
def load_analyzer():
    from ats_analyzer import analyze, AtsScorer
    return analyze

@st.cache_resource(show_spinner=False)
def load_recommender():
    try:
        from job_recommender import JobRecommender
        return JobRecommender()
    except Exception as e:
        return None


# ── Helpers ─────────────────────────────────────────────────
def chips(skills: list, kind: str) -> str:
    return " ".join(f'<span class="chip chip-{kind}">{s}</span>' for s in skills)


def gauge_chart(score: float) -> go.Figure:
    color = "#4ade80" if score >= 70 else "#fbbf24" if score >= 45 else "#f87171"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 40, "color": color, "family": "Space Grotesk"}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#475569"}},
            "bar":   {"color": color, "thickness": 0.28},
            "bgcolor": "#0d1117",
            "steps": [
                {"range": [0, 40],  "color": "#1a0a0a"},
                {"range": [40, 70], "color": "#1a160a"},
                {"range": [70, 100],"color": "#0a1a0f"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": score},
        },
    ))
    fig.update_layout(
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=10, b=10),
        font_color="#94a3b8",
    )
    return fig


def radar_chart(breakdown: dict) -> go.Figure:
    labels = list(breakdown.keys())
    values = list(breakdown.values())
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(99,102,241,0.18)",
        line=dict(color="#818cf8", width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#475569"), gridcolor="#1e2147"),
            angularaxis=dict(tickfont=dict(color="#94a3b8"), gridcolor="#1e2147"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=20, b=20),
        showlegend=False,
        height=300,
    )
    return fig


def bar_breakdown(breakdown: dict) -> go.Figure:
    labels = [k.replace("_", " ").title() for k in breakdown.keys()]
    values = list(breakdown.values())
    colors = ["#4ade80" if v >= 70 else "#fbbf24" if v >= 40 else "#f87171" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition="auto",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 100], gridcolor="#1e2147", tickfont=dict(color="#475569")),
        yaxis=dict(tickfont=dict(color="#94a3b8")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=250,
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 ATS Resume Assistant")
    st.markdown("<p style='color:#64748b;font-size:0.85rem;'>Upload your resume and paste a job description to get an instant ATS score, gap analysis, and real-time job recommendations.</p>", unsafe_allow_html=True)
    st.divider()

    uploaded_pdf = st.file_uploader("📄 Upload Resume PDF", type=["pdf"])
    st.divider()

    st.markdown("**⚙️ Settings**")
    use_t5       = st.toggle("Use T5 rewrite model", value=False, help="Requires local T5 model directory")
    include_intern= st.toggle("Include internships", value=True)
    location_pref = st.text_input("Preferred location", placeholder="e.g. Bangalore, India")
    job_query_override = st.text_input("Custom job search query", placeholder="e.g. NLP Engineer Python")
    st.divider()
    st.markdown("""
    <p style='color:#334155;font-size:0.75rem;'>
    <b style='color:#60a5fa;'>Free APIs (no key needed):</b><br>
    🟢 Jobicy (India filter) — auto-enabled<br>
    🟢 Himalayas — auto-enabled<br>
    🟢 Remotive — auto-enabled<br><br>
    <b style='color:#818cf8;'>Optional (add key for more results):</b><br>
    🔑 ADZUNA_APP_ID + ADZUNA_APP_KEY (India, 1K/mo free)<br>
    🔑 JSEARCH_API_KEY (LinkedIn/Indeed, 500/mo free)
    </p>
    """, unsafe_allow_html=True)


# ── Main layout ──────────────────────────────────────────────
st.markdown("<h1 style='color:#e2e8f0;margin-bottom:0.2rem;'>🎯 ATS Resume Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748b;margin-bottom:1.5rem;'>AI-powered ATS scoring · Gap analysis · Real-time job recommendations</p>", unsafe_allow_html=True)

tab_analyze, tab_jobs = st.tabs(["📊 Resume Analysis", "💼 Job Recommendations"])

# ═══════════════════════════════════════════════════════════════
#  TAB 1 — RESUME ANALYSIS
# ═══════════════════════════════════════════════════════════════
with tab_analyze:
    job_desc = st.text_area(
        "📋 Paste Job Description",
        height=220,
        placeholder="Paste the full job description here — the more detail, the better the ATS analysis...",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_analysis = st.button("⚡ Analyze Resume", use_container_width=True)

    if run_analysis:
        if not uploaded_pdf:
            st.error("Please upload a resume PDF first.")
        elif not job_desc.strip():
            st.error("Please paste a job description.")
        else:
            with st.spinner("Analyzing your resume…"):
                # Save PDF to temp file
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded_pdf.read())
                    tmp_path = tmp.name

                analyze_fn = load_analyzer()
                try:
                    report = analyze_fn(tmp_path, job_desc, use_t5=use_t5)
                    st.session_state["report"] = report
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.stop()
                finally:
                    os.unlink(tmp_path)

    report = st.session_state.get("report")

    if report:
        # ─── Row 1: Score + Breakdown ─────────────────────────────
        c1, c2, c3 = st.columns([1, 1.6, 1.4])

        with c1:
            st.plotly_chart(gauge_chart(report.ats_score), use_container_width=True)
            score_color = "green" if report.ats_score >= 70 else "yellow" if report.ats_score >= 45 else "red"
            label = "Strong Match 🚀" if report.ats_score >= 70 else "Decent Match ✅" if report.ats_score >= 45 else "Needs Work ⚠️"
            st.markdown(f"<div style='text-align:center;margin-top:-1rem;'><span class='chip chip-{score_color}'>{label}</span></div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='section-hdr'>Score Breakdown</div>", unsafe_allow_html=True)
            st.plotly_chart(bar_breakdown(report.score_breakdown), use_container_width=True)

        with c3:
            st.markdown("<div class='section-hdr'>Radar View</div>", unsafe_allow_html=True)
            st.plotly_chart(radar_chart(report.score_breakdown), use_container_width=True)

        st.divider()

        # ─── Row 2: Quick Stats ───────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Word Count",     report.resume_word_count)
        m2.metric("Skills Found",   len(report.found_skills_resume))
        m3.metric("Missing Skills", len(report.missing_skills_for_job))
        m4.metric("Keyword Density",f"{report.keyword_density}%")

        st.divider()

        # ─── Row 3: Skills ───────────────────────────────────────
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("<div class='section-hdr'>✅ Skills in Your Resume</div>", unsafe_allow_html=True)
            if report.found_skills_resume:
                st.markdown(chips(report.found_skills_resume, "green"), unsafe_allow_html=True)
            else:
                st.info("No recognized skills detected.")

            st.markdown("<div class='section-hdr' style='margin-top:1rem;'>❌ Missing Skills (Job Gap)</div>", unsafe_allow_html=True)
            if report.missing_skills_for_job:
                st.markdown(chips(report.missing_skills_for_job, "red"), unsafe_allow_html=True)
            else:
                st.success("No critical skill gaps!")

        with col_r:
            st.markdown("<div class='section-hdr'>🔑 Missing ATS Keywords</div>", unsafe_allow_html=True)
            if report.missing_keywords_for_ats:
                st.markdown(chips(report.missing_keywords_for_ats[:18], "yellow"), unsafe_allow_html=True)
            else:
                st.success("Great keyword coverage!")

            st.markdown("<div class='section-hdr' style='margin-top:1rem;'>📊 Section Completeness</div>", unsafe_allow_html=True)
            for sec in report.section_scores:
                icon = "✅" if sec.score == 100 else "⚠️"
                color = "#4ade80" if sec.score == 100 else "#f87171"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e2147;'>"
                    f"<span style='color:#94a3b8;text-transform:capitalize;'>{icon} {sec.name}</span>"
                    f"<span style='color:{color};font-size:0.8rem;'>{sec.details}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.divider()

        # ─── Row 4: Improvement Tips ─────────────────────────────
        st.markdown("<div class='section-hdr'>💡 Improvement Tips</div>", unsafe_allow_html=True)
        for tip in report.improvement_tips:
            st.markdown(f"<div class='tip-card'>💡 {tip}</div>", unsafe_allow_html=True)

        st.divider()

        # ─── Row 5: Rewrite Suggestions ──────────────────────────
        st.markdown("<div class='section-hdr'>✍️ Rewrite Suggestions</div>", unsafe_allow_html=True)
        if report.rewrite_suggestions:
            for i, s in enumerate(report.rewrite_suggestions, 1):
                with st.expander(f"Suggestion {i} — {s.reason[:60]}"):
                    st.markdown(f"<div class='rw-original'>❌ Before: {s.original}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='rw-suggested'>✅ After: {s.suggested}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='rw-reason'>📌 {s.reason}</div>", unsafe_allow_html=True)
        else:
            st.info("No rewrite suggestions generated. Your bullets look solid!")

        st.divider()

        # ─── Row 6: Recommended Roles ────────────────────────────
        st.markdown("<div class='section-hdr'>🧭 Best Matching Roles for You</div>", unsafe_allow_html=True)
        roles_df = pd.DataFrame(report.recommended_roles)
        if not roles_df.empty:
            fig_roles = px.bar(
                roles_df,
                x="match_percent", y="role", orientation="h",
                color="match_percent",
                color_continuous_scale=["#f87171", "#fbbf24", "#4ade80"],
                range_color=[0, 100],
                labels={"match_percent": "Match %", "role": "Role"},
            )
            fig_roles.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(tickfont=dict(color="#94a3b8")),
                xaxis=dict(range=[0, 100], gridcolor="#1e2147", tickfont=dict(color="#475569")),
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
            )
            st.plotly_chart(fig_roles, use_container_width=True)

        st.divider()

        # ─── Download Report ─────────────────────────────────────
        from ats_analyzer import save_report
        from dataclasses import asdict as _asdict
        import json as _json

        payload = _asdict(report)
        payload["rewrite_suggestions"] = [_asdict(r) for r in report.rewrite_suggestions]
        json_str = _json.dumps(payload, indent=2, ensure_ascii=False)
        st.download_button("📥 Download Full Report (JSON)", json_str, "ats_report.json", "application/json")


# ═══════════════════════════════════════════════════════════════
#  TAB 2 — JOB RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════

with tab_jobs:
    st.markdown("<p style='color:#94a3b8;'>Real-time job search powered by JSearch API (LinkedIn, Indeed, Glassdoor, Naukri) + Adzuna.</p>", unsafe_allow_html=True)

    jcol1, jcol2 = st.columns([3, 1])
    with jcol1:
        custom_jq = st.text_input("🔍 Job Search Query (optional)", value=job_query_override,
                                   placeholder="e.g. Data Scientist, ML Engineer, NLP Intern")
    with jcol2:
        top_k_jobs = st.number_input("# of jobs", min_value=5, max_value=30, value=10)

    run_jobs = st.button("🔎 Find Jobs Now", use_container_width=False)

    if run_jobs:
        if not uploaded_pdf:
            st.warning("Upload a resume PDF for personalized recommendations. Searching with query only.")
            resume_text_for_jobs = custom_jq
        else:
            from ats_analyzer import extract_pdf_text
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                uploaded_pdf.seek(0)
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name
            resume_text_for_jobs = extract_pdf_text(tmp_path)
            os.unlink(tmp_path)

        recommender = load_recommender()
        if recommender is None:
            st.error("Job recommender could not be loaded. Ensure job_recommender.py is in the same directory.")
            st.stop()

        with st.spinner("Fetching real-time jobs…"):
            try:
                jobs = recommender.recommend(
                    resume_text  = resume_text_for_jobs,
                    job_query    = custom_jq,
                    location     = location_pref,
                    include_internships = include_intern,
                    top_k        = int(top_k_jobs),
                )
                st.session_state["jobs"] = jobs
            except Exception as e:
                st.error(f"Job fetch failed: {e}")
                st.stop()

    jobs = st.session_state.get("jobs", [])

    if jobs:
        st.markdown(f"<div class='section-hdr'>Found {len(jobs)} relevant opportunities</div>", unsafe_allow_html=True)

        # Filter bar
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            type_filter = st.multiselect("Job type", ["full_time","internship","contract","part_time"], default=[])
        with fc2:
            remote_filter = st.checkbox("Remote only", value=False)
        with fc3:
            min_match = st.slider("Min skill match %", 0, 100, 0, 5)

        filtered = [j for j in jobs if
            (not type_filter or j.job_type in type_filter) and
            (not remote_filter or j.is_remote) and
            j.skill_match >= min_match
        ]

        if not filtered:
            st.info("No jobs match the current filters. Try relaxing them.")
        else:
            for j in filtered:
                match_w = int(j.skill_match)
                sim_pct = int(j.similarity * 100)
                note_color = "#4ade80" if j.skill_match >= 70 else "#fbbf24" if j.skill_match >= 40 else "#f87171"
                remote_badge = "🌐 Remote" if j.is_remote else ""
                intern_badge = "🎓 Internship" if "intern" in j.job_type.lower() else ""

                st.markdown(f"""
                <div class="job-card">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                      <div class="job-title">{j.title}</div>
                      <div class="job-meta">{j.company} &nbsp;·&nbsp; {j.location or 'Location N/A'} &nbsp;·&nbsp; {j.date_posted or ''} &nbsp; {remote_badge} &nbsp; {intern_badge}</div>
                    </div>
                    <div style="text-align:right;min-width:90px;">
                      <div style="font-size:1.3rem;font-weight:700;color:{note_color};font-family:'Space Grotesk',sans-serif;">{match_w}%</div>
                      <div style="font-size:0.72rem;color:#64748b;">skill match</div>
                    </div>
                  </div>
                  <div class="match-bar-bg"><div class="match-bar" style="width:{match_w}%;"></div></div>
                  <div style="font-size:0.82rem;color:#64748b;margin-top:2px;">
                    Salary: <span style="color:#94a3b8;">{j.salary_range}</span> &nbsp;·&nbsp;
                    Source: <span style="color:#818cf8;">{j.source}</span>
                  </div>
                  <div style="margin-top:8px;font-size:0.82rem;color:{note_color};">▸ {j.relevance_note}</div>
                  {'<div style="margin-top:6px;">' + chips(j.matched_skills[:6], "green") + '</div>' if j.matched_skills else ''}
                  {'<div style="margin-top:4px;">' + chips(j.missing_skills[:4], "red") + '</div>' if j.missing_skills else ''}
                  {'<a href="' + j.url + '" target="_blank" style="display:inline-block;margin-top:10px;background:linear-gradient(135deg,#4f46e5,#0ea5e9);color:white;padding:5px 16px;border-radius:8px;font-size:0.82rem;font-weight:600;text-decoration:none;">Apply Now →</a>' if j.url else ''}
                </div>
                """, unsafe_allow_html=True)

        # Download job results
        jobs_data = [
            {k: v for k, v in j.__dict__.items()}
            for j in filtered
        ]
        import json as _json2
        st.download_button(
            "📥 Download Job Results (JSON)",
            _json2.dumps(jobs_data, indent=2),
            "job_recommendations.json",
            "application/json",
        )

    elif not run_jobs:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#334155;">
          <div style="font-size:3rem;margin-bottom:1rem;">💼</div>
          <div style="font-size:1.1rem;color:#475569;">Click <strong style='color:#818cf8;'>Find Jobs Now</strong> to get real-time job recommendations tailored to your resume.</div>
          <div style="font-size:0.85rem;margin-top:0.5rem;color:#334155;">Powered by JSearch (LinkedIn · Indeed · Glassdoor · Naukri) + Adzuna</div>
        </div>
        """, unsafe_allow_html=True)
