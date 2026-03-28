import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from groq import Groq
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="MedRisk AI — Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background: #0d1117;
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: #161b22; }
[data-testid="stToolbar"] { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

/* ── Top nav bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 1.5rem;
    border-bottom: 1px solid #21262d;
    margin-bottom: 2rem;
}
.topbar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}
.logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #58a6ff, #1f6feb);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.logo-text {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: -0.02em;
}
.logo-sub {
    font-size: 0.72rem;
    color: #8b949e;
    font-weight: 400;
    margin-top: 1px;
}
.badge-row { display: flex; gap: 8px; }
.badge {
    font-size: 0.68rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-blue  { background: #1f3a5f; color: #58a6ff; border: 1px solid #1f6feb44; }
.badge-green { background: #1a3a2a; color: #3fb950; border: 1px solid #2ea04344; }
.badge-purple{ background: #2d1f4a; color: #bc8cff; border: 1px solid #8957e544; }

/* ── Section header ── */
.section-head {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b949e;
    margin: 0 0 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #21262d;
}

/* ── Input card ── */
.input-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.4rem 1.5rem 1rem;
    height: 100%;
    transition: border-color 0.2s;
}
.input-card:hover { border-color: #30363d; }

/* ── Sliders + selects ── */
[data-testid="stSlider"] { padding: 0.1rem 0; }
[data-testid="stSlider"] > div > div > div > div {
    background: #1f6feb !important;
}
[data-testid="stSlider"] > div > div > div {
    background: #21262d !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #8b949e !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #f0f6fc !important;
    font-size: 0.85rem !important;
}

/* ── Main predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.8rem 2rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px #1f6feb55 !important;
}

/* ── Risk score card ── */
.risk-card {
    border-radius: 14px;
    padding: 2rem;
    display: flex;
    align-items: center;
    gap: 1.8rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.risk-card::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 160px; height: 160px;
    border-radius: 50%;
    opacity: 0.07;
}
.risk-high   { background: #1a0a0a; border: 1px solid #6e1a1a; }
.risk-high::before { background: #f85149; }
.risk-medium { background: #191109; border: 1px solid #6e4a0e; }
.risk-medium::before { background: #e3b341; }
.risk-low    { background: #091319; border: 1px solid #0e4429; }
.risk-low::before { background: #3fb950; }

.risk-number {
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.04em;
    font-family: 'DM Mono', monospace;
}
.risk-high   .risk-number { color: #f85149; }
.risk-medium .risk-number { color: #e3b341; }
.risk-low    .risk-number { color: #3fb950; }

.risk-label {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f0f6fc;
    margin-bottom: 4px;
}
.risk-desc {
    font-size: 0.82rem;
    color: #8b949e;
    line-height: 1.5;
    max-width: 280px;
}
.risk-divider { width: 1px; height: 60px; background: #21262d; flex-shrink: 0; }

/* ── Stat tiles ── */
.stat-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 1.5rem;
}
.stat-tile {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    transition: border-color 0.2s;
}
.stat-tile:hover { border-color: #388bfd44; }
.stat-tile-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b949e;
    margin-bottom: 6px;
}
.stat-tile-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f0f6fc;
    font-family: 'DM Mono', monospace;
}
.stat-tile-sub {
    font-size: 0.68rem;
    color: #8b949e;
    margin-top: 2px;
}

/* ── SHAP section ── */
.shap-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1.2rem;
}

/* ── AI summary ── */
.ai-card {
    background: linear-gradient(135deg, #0c1a2e 0%, #0f2241 100%);
    border: 1px solid #1f6feb44;
    border-radius: 12px;
    padding: 1.6rem;
    position: relative;
    overflow: hidden;
}
.ai-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #1f6feb, #388bfd, #58a6ff);
}
.ai-card-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #388bfd;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.ai-card-text {
    font-size: 0.92rem;
    color: #c9d1d9;
    line-height: 1.75;
}

/* ── History table ── */
.history-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    margin-top: 1.5rem;
}
.history-row {
    display: grid;
    grid-template-columns: 60px 60px 80px 80px 100px auto;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid #21262d;
    font-size: 0.8rem;
    align-items: center;
}
.history-row:last-child { border-bottom: none; }
.history-head { color: #8b949e; font-weight: 600; font-size: 0.7rem; text-transform: uppercase; }
.chip {
    font-size: 0.68rem;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 20px;
    display: inline-block;
}
.chip-red    { background: #6e1a1a; color: #f85149; }
.chip-yellow { background: #6e4a0e; color: #e3b341; }
.chip-green  { background: #0e4429; color: #3fb950; }

/* ── Divider ── */
hr { border-color: #21262d !important; margin: 1.5rem 0 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] p { color: #8b949e !important; font-size: 0.85rem !important; }

/* ── Hide streamlit default elements ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load("models/xgboost_readmission.pkl")
    with open("models/feature_names.json") as f:
        features = json.load(f)
    return model, features

model, feature_names = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

# ── Top navigation bar ─────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-logo">
    <div class="logo-icon">🏥</div>
    <div>
      <div class="logo-text">MedRisk AI</div>
      <div class="logo-sub">Hospital Readmission Predictor</div>
    </div>
  </div>
  <div class="badge-row">
    <span class="badge badge-blue">XGBoost Model</span>
    <span class="badge badge-green">SHAP Explainability</span>
    <span class="badge badge-purple">LLaMA 3.1 Summaries</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Input section ──────────────────────────────────────────────────────
st.markdown('<div class="section-head">Patient Parameters</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("**Patient Info**", help="Basic demographic information")
    age               = st.slider("Age", 20, 95, 65)
    gender            = st.selectbox("Gender", ["Female", "Male"])
    time_in_hospital  = st.slider("Days in hospital", 1, 14, 4)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("**Clinical Details**", help="Lab and medication information")
    num_medications   = st.slider("Number of medications", 1, 40, 15)
    num_lab_procedures= st.slider("Lab procedures", 1, 100, 45)
    number_diagnoses  = st.slider("Number of diagnoses", 1, 9, 5)
    num_procedures    = st.slider("Procedures done", 0, 6, 1)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("**Visit History**", help="Prior hospital visit history")
    number_inpatient  = st.slider("Prior inpatient visits",  0, 15, 0)
    number_outpatient = st.slider("Prior outpatient visits", 0, 15, 0)
    number_emergency  = st.slider("Prior emergency visits",  0, 15, 0)
    insulin           = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍  Analyse Patient Risk", use_container_width=True)

# ── Prediction logic ───────────────────────────────────────────────────
if predict_btn:
    med_map = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
    total_visits       = number_inpatient + number_outpatient + number_emergency
    emergency_rate     = round(number_emergency / (total_visits + 1), 3)
    medication_density = round(num_medications / (time_in_hospital + 1), 3)
    is_senior          = 1 if age >= 65 else 0
    high_visit_risk    = 1 if (is_senior == 1 and number_emergency >= 3) else 0

    input_data = {f: 0 for f in feature_names}
    input_data.update({
        'age': age,
        'gender': 1 if gender == "Male" else 0,
        'time_in_hospital': time_in_hospital,
        'num_medications': num_medications,
        'num_lab_procedures': num_lab_procedures,
        'number_diagnoses': number_diagnoses,
        'num_procedures': num_procedures,
        'number_inpatient': number_inpatient,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'insulin': med_map[insulin],
        'total_visits': total_visits,
        'emergency_rate': emergency_rate,
        'medication_density': medication_density,
        'is_senior': is_senior,
        'high_visit_risk': high_visit_risk,
    })

    input_df   = pd.DataFrame([input_data])
    risk_proba = model.predict_proba(input_df)[0][1]
    risk_pct   = round(float(risk_proba) * 100, 1)   # ✅ always 1 decimal

    if risk_pct > 20:
        risk_cls, risk_label, risk_desc = (
            "risk-high", "🔴 HIGH RISK",
            "Immediate care coordination and follow-up plan required before discharge."
        )
        chip_cls = "chip-red"
    elif risk_pct > 12:
        risk_cls, risk_label, risk_desc = (
            "risk-medium", "🟡 MODERATE RISK",
            "Schedule follow-up within 7 days and review discharge medications."
        )
        chip_cls = "chip-yellow"
    else:
        risk_cls, risk_label, risk_desc = (
            "risk-low", "🟢 LOW RISK",
            "Patient appears stable. Standard discharge protocol applies."
        )
        chip_cls = "chip-green"

    st.markdown('<div class="section-head" style="margin-top:1.5rem;">Risk Assessment</div>',
                unsafe_allow_html=True)

    # ── Risk score banner ──
    st.markdown(f"""
    <div class="risk-card {risk_cls}">
      <div class="risk-number">{risk_pct}%</div>
      <div class="risk-divider"></div>
      <div>
        <div class="risk-label">{risk_label}</div>
        <div class="risk-desc">{risk_desc}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 4 stat tiles ──
    er_display  = f"{emergency_rate:.1%}"
    md_display  = f"{medication_density:.1f}"
    sen_display = "Yes" if is_senior else "No"
    hrv_display = "Yes" if high_visit_risk else "No"

    st.markdown(f"""
    <div class="stat-strip">
      <div class="stat-tile">
        <div class="stat-tile-label">Emergency Rate</div>
        <div class="stat-tile-value">{er_display}</div>
        <div class="stat-tile-sub">of all visits via ER</div>
      </div>
      <div class="stat-tile">
        <div class="stat-tile-label">Med Density</div>
        <div class="stat-tile-value">{md_display}</div>
        <div class="stat-tile-sub">medications per day</div>
      </div>
      <div class="stat-tile">
        <div class="stat-tile-label">Prior Visits</div>
        <div class="stat-tile-value">{total_visits}</div>
        <div class="stat-tile-sub">total hospital contacts</div>
      </div>
      <div class="stat-tile">
        <div class="stat-tile-label">High Visit Risk</div>
        <div class="stat-tile-value">{hrv_display}</div>
        <div class="stat-tile-sub">senior + 3+ ER visits</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SHAP chart ──────────────────────────────────────────────────────
    with st.spinner("Computing SHAP explainability..."):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_df)
        top_idx   = np.argsort(np.abs(shap_vals[0]))[::-1][:8]
        top_factors = "\n".join([
            f"{i+1}. {feature_names[idx]}: "
            f"{'+'if shap_vals[0][idx]>0 else ''}{shap_vals[0][idx]:.3f}"
            for i, idx in enumerate(top_idx[:5])
        ])

    st.markdown('<div class="section-head">SHAP Explainability</div>', unsafe_allow_html=True)

    top_vals  = shap_vals[0][top_idx]
    top_names = [feature_names[i] for i in top_idx]
    colors    = ['#f85149' if v > 0 else '#58a6ff' for v in top_vals]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')

    bars = ax.barh(
        range(len(top_vals)),
        top_vals[::-1] if False else top_vals,
        color=colors,
        height=0.58,
        edgecolor='none'
    )
    # clean feature names for display
    display_names = [n.replace('_', ' ').title() for n in top_names]
    ax.set_yticks(range(len(top_vals)))
    ax.set_yticklabels(display_names, fontsize=9.5,
                       color='#c9d1d9', fontfamily='monospace')

    ax.axvline(0, color='#30363d', linewidth=1)
    ax.set_xlabel("SHAP value", fontsize=9, color='#8b949e', labelpad=8)
    ax.tick_params(axis='x', colors='#8b949e', labelsize=8.5)
    ax.tick_params(axis='y', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='x', alpha=0.15, color='#8b949e', linestyle='--')

    # legend
    red_patch  = mpatches.Patch(color='#f85149', label='Increases risk')
    blue_patch = mpatches.Patch(color='#58a6ff', label='Decreases risk')
    ax.legend(handles=[red_patch, blue_patch], loc='lower right',
              fontsize=8.5, frameon=False,
              labelcolor='#8b949e')

    plt.tight_layout(pad=1.2)
    st.pyplot(fig)
    plt.close()

    # ── Groq AI summary ─────────────────────────────────────────────────
    with st.spinner("Generating clinical summary with LLaMA 3.1..."):
        try:
            client  = Groq(api_key=os.getenv("GROQ_API_KEY"))
            prompt  = f"""You are a clinical decision support assistant helping hospital doctors.
A machine learning model has assessed a diabetic patient for 30-day readmission risk.

Patient profile:
- Age: {age} | Gender: {gender} | Days in hospital: {time_in_hospital}
- Medications: {num_medications} | Lab procedures: {num_lab_procedures}
- Prior visits (inpatient/outpatient/emergency): {number_inpatient}/{number_outpatient}/{number_emergency}
- Insulin: {insulin} | Senior patient: {"Yes" if is_senior else "No"}
- Computed emergency rate: {emergency_rate:.1%}
- Computed medication density: {medication_density:.1f} meds/day

Model output: {risk_pct}% probability of readmission within 30 days ({risk_label})

Top SHAP risk factors:
{top_factors}

Instructions: Write exactly 3 sentences for the attending physician.
Sentence 1: Summarise the patient's overall risk with the key driving factors.
Sentence 2: Explain what the SHAP factors reveal about clinical vulnerability.
Sentence 3: Give ONE specific, actionable discharge recommendation.
Be clinical, precise, and direct. No bullet points."""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=280
            )
            summary = response.choices[0].message.content.strip()
            st.markdown(f"""
            <div class="ai-card">
              <div class="ai-card-label">⚡ AI Clinical Summary — LLaMA 3.1 via Groq</div>
              <div class="ai-card-text">{summary}</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"AI summary unavailable: {e}")

    # ── Save to history ──────────────────────────────────────────────────
    st.session_state.history.append({
        "age": age,
        "days": time_in_hospital,
        "meds": num_medications,
        "visits": total_visits,
        "risk": risk_pct,
        "label": risk_label,
        "chip": chip_cls,
    })

# ── Patient history table ────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<div class="section-head" style="margin-top:2rem;">Prediction History</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="history-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="history-row history-head">
      <span>Age</span><span>Days</span><span>Meds</span>
      <span>Visits</span><span>Risk</span><span>Status</span>
    </div>
    """, unsafe_allow_html=True)

    for h in reversed(st.session_state.history[-6:]):
        st.markdown(f"""
        <div class="history-row">
          <span style="color:#c9d1d9;">{h['age']}</span>
          <span style="color:#c9d1d9;">{h['days']}</span>
          <span style="color:#c9d1d9;">{h['meds']}</span>
          <span style="color:#c9d1d9;">{h['visits']}</span>
          <span style="font-family:'DM Mono',monospace;color:#f0f6fc;font-weight:700;">{h['risk']}%</span>
          <span><span class="chip {h['chip']}">{h['label']}</span></span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🗑  Clear History", use_container_width=False):
        st.session_state.history = []
        st.rerun()