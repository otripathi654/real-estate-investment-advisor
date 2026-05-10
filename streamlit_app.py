"""
=============================================================
 Real Estate Investment Advisor
 Step 4: Streamlit Application
=============================================================
 Usage:
   streamlit run streamlit_app.py

 Pages:
   🏠  Home            — Project overview
   🔍  Predict         — Investment classifier + price forecast
   📊  EDA Insights    — Browse all 20 EDA charts
   🤖  Model Report    — Metrics, feature importance, confusion matrix
   📁  Data Explorer   — Filter & browse the dataset
=============================================================
"""

import os
import glob
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# DARK MODE STATE  (must be before CSS injection)
# ─────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ─────────────────────────────────────────────
# CUSTOM CSS  (light / dark aware)
# ─────────────────────────────────────────────
dark = st.session_state.dark_mode

if dark:
    app_bg          = "#0F172A"
    sidebar_bg      = "#1E293B"
    card_bg         = "#1E293B"
    card_border     = "#334155"
    text_primary    = "#F1F5F9"
    text_secondary  = "#94A3B8"
    header_color    = "#93C5FD"
    accent          = "#3B82F6"
    plot_bg         = "#1E293B"
    paper_bg        = "#1E293B"
    plot_font       = "#F1F5F9"
    metric_shadow   = "rgba(0,0,0,0.4)"
    good_bg         = "linear-gradient(135deg, #064E3B, #065F46)"
    good_border     = "#10B981"
    good_text       = "#6EE7B7"
    bad_bg          = "linear-gradient(135deg, #7F1D1D, #991B1B)"
    bad_border      = "#EF4444"
    bad_text        = "#FCA5A5"
    price_bg        = "linear-gradient(135deg, #1E3A5F, #1e40af)"
    price_border    = "#3B82F6"
    price_text      = "#BFDBFE"
    input_bg        = "#1E293B"
    divider         = "#334155"
else:
    app_bg          = "#F0F4F8"
    sidebar_bg      = "#FFFFFF"
    card_bg         = "#FFFFFF"
    card_border     = "#E2E8F0"
    text_primary    = "#1E293B"
    text_secondary  = "#64748B"
    header_color    = "#1E3A5F"
    accent          = "#2563EB"
    plot_bg         = "#FFFFFF"
    paper_bg        = "#FFFFFF"
    plot_font       = "#1E293B"
    metric_shadow   = "rgba(0,0,0,0.08)"
    good_bg         = "linear-gradient(135deg, #D1FAE5, #A7F3D0)"
    good_border     = "#10B981"
    good_text       = "#065F46"
    bad_bg          = "linear-gradient(135deg, #FEE2E2, #FECACA)"
    bad_border      = "#EF4444"
    bad_text        = "#7F1D1D"
    price_bg        = "linear-gradient(135deg, #EFF6FF, #DBEAFE)"
    price_border    = "#2563EB"
    price_text      = "#1E3A5F"
    input_bg        = "#FFFFFF"
    divider         = "#E2E8F0"

st.markdown(f"""
<style>
  /* ── Base ── */
  .stApp {{ background-color: {app_bg} !important; }}
  .stApp, .stApp p, .stApp label, .stApp span, .stApp div {{
    color: {text_primary} !important;
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
    background-color: {sidebar_bg} !important;
  }}
  [data-testid="stSidebar"] * {{
    color: {text_primary} !important;
  }}

  /* ── Metric cards ── */
  div[data-testid="metric-container"] {{
    background: {card_bg} !important;
    border: 1px solid {card_border};
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px {metric_shadow};
  }}
  div[data-testid="metric-container"] label,
  div[data-testid="metric-container"] div {{
    color: {text_primary} !important;
  }}

  /* ── Section headers ── */
  .section-header {{
    font-size: 22px;
    font-weight: 700;
    color: {header_color} !important;
    border-left: 5px solid {accent};
    padding-left: 12px;
    margin: 24px 0 16px 0;
  }}

  /* ── Result box — green ── */
  .result-good {{
    background: {good_bg};
    border: 2px solid {good_border};
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    font-size: 20px;
    font-weight: 700;
    color: {good_text} !important;
  }}

  /* ── Result box — red ── */
  .result-bad {{
    background: {bad_bg};
    border: 2px solid {bad_border};
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    font-size: 20px;
    font-weight: 700;
    color: {bad_text} !important;
  }}

  /* ── Price forecast card ── */
  .price-card {{
    background: {price_bg};
    border: 2px solid {price_border};
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    font-size: 20px;
    font-weight: 700;
    color: {price_text} !important;
  }}

  /* ── Input widgets ── */
  .stTextInput input, .stNumberInput input, .stSelectbox select,
  div[data-baseweb="select"] > div,
  div[data-baseweb="input"] > div {{
    background-color: {input_bg} !important;
    color: {text_primary} !important;
    border-color: {card_border} !important;
  }}

  /* ── Dataframe / table ── */
  .stDataFrame, iframe {{
    background-color: {card_bg} !important;
  }}

  /* ── st.info / st.success boxes ── */
  div[data-testid="stInfo"], div[data-testid="stSuccess"] {{
    background-color: {card_bg} !important;
    border-color: {accent} !important;
    color: {text_primary} !important;
  }}

  /* ── Tabs ── */
  button[data-baseweb="tab"] {{
    color: {text_secondary} !important;
  }}
  button[data-baseweb="tab"][aria-selected="true"] {{
    color: {accent} !important;
    border-bottom-color: {accent} !important;
  }}

  /* ── Horizontal rule ── */
  hr {{ border-color: {divider} !important; }}

  /* ── Hide default header ── */
  header[data-testid="stHeader"] {{ background: transparent; }}
</style>
""", unsafe_allow_html=True)

# Helper: plot layout kwargs to apply dark/light theme to every Plotly chart
def plot_layout(**extra):
    return dict(
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font_color=plot_font,
        **extra,
    )


# ─────────────────────────────────────────────
# LOAD ARTEFACTS  (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models …")
def load_models():
    clf = joblib.load("models/best_classifier.pkl")   if os.path.exists("models/best_classifier.pkl")   else None
    reg = joblib.load("models/best_regressor.pkl")    if os.path.exists("models/best_regressor.pkl")    else None
    meta = joblib.load("models/model_metadata.pkl")   if os.path.exists("models/model_metadata.pkl")    else {}
    features = joblib.load("models/feature_names.pkl") if os.path.exists("models/feature_names.pkl")   else []
    return clf, reg, meta, features


@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    paths = ["data/processed_data.csv", "data/india_housing_prices.csv"]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return pd.DataFrame()


clf_model, reg_model, meta, feature_names = load_models()
df = load_data()


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/home.png", width=64)
    st.title("Real Estate\nAdvisor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Home", "🔍  Predict", "📊  EDA Insights",
         "🤖  Model Report", "📁  Data Explorer"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    # ── Dark mode toggle ──────────────────────
    toggle_label = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
    if st.button(toggle_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.markdown("---")
    st.caption("Built with Python · Scikit-learn · XGBoost · MLflow · Streamlit")


# ═══════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.title("🏠 Real Estate Investment Advisor")
    st.markdown("#### Predicting Property Profitability & Future Value")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Rows", f"{len(df):,}" if not df.empty else "—")
    with col2:
        st.metric("Features Used", len(feature_names) if feature_names else "—")
    with col3:
        best_clf = meta.get("best_classifier", "—")
        clf_f1   = meta.get("clf_metrics", {}).get(best_clf, {}).get("f1", None)
        st.metric("Best Clf F1", f"{clf_f1:.3f}" if clf_f1 else "—")
    with col4:
        best_reg = meta.get("best_regressor", "—")
        reg_r2   = meta.get("reg_metrics", {}).get(best_reg, {}).get("R2", None)
        st.metric("Best Reg R²", f"{reg_r2:.3f}" if reg_r2 else "—")

    st.markdown('<div class="section-header">Project Overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Problem Statement**

Develop an ML application to assist real estate investors:
- ✅ **Classify** whether a property is a *Good Investment*
- 📈 **Predict** the estimated property price after **5 years**

**Domain:** Real Estate / Investment / Financial Analytics
        """)
    with c2:
        st.markdown("""
**Tech Stack**

| Layer | Tools |
|---|---|
| Data Processing | Pandas, NumPy, Scikit-learn |
| ML Models | XGBoost, Random Forest |
| Experiment Tracking | MLflow |
| Visualisation | Plotly, Seaborn, Matplotlib |
| Deployment | Streamlit |
        """)

    st.markdown('<div class="section-header">Business Use Cases</div>', unsafe_allow_html=True)
    u1, u2, u3, u4 = st.columns(4)
    for col, icon, title, desc in [
        (u1, "💼", "Investors", "Identify high-return properties intelligently"),
        (u2, "🏡", "Buyers",    "Choose properties in high-appreciation areas"),
        (u3, "🏢", "Agencies",  "Automate investment analysis for listings"),
        (u4, "📊", "Platforms", "Build trust with data-backed price forecasts"),
    ]:
        with col:
            st.info(f"**{icon} {title}**\n\n{desc}")

    st.markdown('<div class="section-header">Pipeline</div>', unsafe_allow_html=True)
    steps = ["📥 Raw Data", "🔧 Preprocessing", "📊 EDA", "🤖 Model Training", "🚀 Streamlit App"]
    cols  = st.columns(len(steps))
    for i, (c, s) in enumerate(zip(cols, steps)):
        with c:
            st.success(f"**Step {i+1}**\n\n{s}")


# ═══════════════════════════════════════════════════════════
#  PAGE 2 — PREDICT
# ═══════════════════════════════════════════════════════════
elif page == "🔍  Predict":
    st.title("🔍 Property Investment Predictor")
    st.markdown("Enter property details below to get an investment recommendation and 5-year price forecast.")
    st.markdown("---")

    if clf_model is None or reg_model is None:
        st.error("⚠️  Models not found. Please run `train_models.py` first to generate `models/` folder.")
        st.stop()

    # ── Input Form ────────────────────────────────────────
    st.markdown('<div class="section-header">Property Details</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📍 Location**")
        state          = st.selectbox("State", sorted(df["State"].dropna().unique()) if "State" in df.columns else ["Maharashtra"])
        city_options   = sorted(df[df["State"] == state]["City"].dropna().unique()) if "City" in df.columns else ["Mumbai"]
        city           = st.selectbox("City", city_options if len(city_options) > 0 else ["Mumbai"])
        locality       = st.text_input("Locality", value="Bandra")

    with col2:
        st.markdown("**🏗️ Property Info**")
        property_type  = st.selectbox("Property Type", ["Apartment", "Villa", "House", "Studio", "Penthouse"])
        bhk            = st.slider("BHK", 1, 6, 3)
        size_sqft      = st.number_input("Size (SqFt)", min_value=100, max_value=10000, value=1200, step=50)
        price_lakhs    = st.number_input("Current Price (Lakhs ₹)", min_value=5.0, max_value=5000.0, value=85.0, step=5.0)
        year_built     = st.number_input("Year Built", min_value=1970, max_value=2024, value=2015, step=1)

    with col3:
        st.markdown("**🏙️ Amenities & Infrastructure**")
        furnished      = st.selectbox("Furnished Status", ["Fully", "Semi", "Unfurnished"])
        nearby_schools = st.slider("Nearby Schools", 0, 10, 4)
        nearby_hosp    = st.slider("Nearby Hospitals", 0, 10, 3)
        parking        = st.slider("Parking Spaces", 0, 5, 1)
        transport      = st.selectbox("Public Transport", ["High", "Medium", "Low"])
        floor_no       = st.number_input("Floor No.", min_value=0, max_value=50, value=5)
        total_floors   = st.number_input("Total Floors", min_value=1, max_value=60, value=12)
        security       = st.selectbox("Security", ["Gated", "CCTV", "Guard", "None"])
        facing         = st.selectbox("Facing", ["North", "South", "East", "West", "North-East", "North-West"])
        owner_type     = st.selectbox("Owner Type", ["Builder", "Individual", "Agent"])

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Investment", type="primary", use_container_width=True)

    if predict_btn:
        # ── Build feature row ─────────────────────────────
        age_of_property    = 2025 - int(year_built)
        price_per_sqft     = (price_lakhs * 100_000) / max(size_sqft, 1)
        floor_ratio        = floor_no / max(total_floors, 1)
        school_density     = nearby_schools / 10.0
        hospital_density   = nearby_hosp / 10.0
        transport_score    = {"High": 3, "Medium": 2, "Low": 1}[transport]
        infra_score        = school_density + hospital_density + transport_score / 3
        amenity_count      = 3   # default; would come from multi-select in full app
        is_new             = int(age_of_property <= 5)
        has_security       = int(security in ["Gated", "CCTV", "Guard"])
        is_furnished       = int(furnished == "Fully")

        # Encode state/city/locality (use median-encoded fallback)
        state_enc    = hash(state)    % 1000
        city_enc     = hash(city)     % 1000
        locality_enc = hash(locality) % 1000

        input_dict = {
            "BHK":                      bhk,
            "Size_in_SqFt":             size_sqft / 10000,   # scaled
            "Price_in_Lakhs":           price_lakhs,
            "Price_per_SqFt":           price_per_sqft / 100000,
            "Floor_No":                 floor_no,
            "Total_Floors":             total_floors,
            "Age_of_Property":          age_of_property / 55,
            "Nearby_Schools":           nearby_schools / 10,
            "Nearby_Hospitals":         nearby_hosp / 10,
            "Parking_Space":            parking / 5,
            "Floor_Ratio":              floor_ratio,
            "School_Density_Score":     school_density,
            "Hospital_Density_Score":   hospital_density,
            "Infrastructure_Score":     infra_score / 4,
            "Amenity_Count":            amenity_count / 10,
            "Is_New_Property":          is_new,
            "Has_Premium_Security":     has_security,
            "Is_Fully_Furnished":       is_furnished,
            "State_enc":                state_enc,
            "City_enc":                 city_enc,
            "Locality_enc":             locality_enc,
        }

        # Align to training feature names
        row = pd.DataFrame([input_dict])
        for col in feature_names:
            if col not in row.columns:
                row[col] = 0
        row = row[feature_names]

        # ── Run inference ─────────────────────────────────
        clf_pred    = clf_model.predict(row)[0]
        clf_proba   = clf_model.predict_proba(row)[0] if hasattr(clf_model, "predict_proba") else None
        reg_pred    = reg_model.predict(row)[0]
        future_price = price_lakhs * (1.08 ** 5)  # cross-check formula

        # ── Display results ───────────────────────────────
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)

        with r1:
            if clf_pred == 1:
                st.markdown(
                    f'<div class="result-good">✅ Good Investment<br>'
                    f'<span style="font-size:15px;font-weight:400;">Confidence: '
                    f'{clf_proba[1]*100:.1f}%</span></div>' if clf_proba is not None
                    else '<div class="result-good">✅ Good Investment</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-bad">❌ Not a Good Investment<br>'
                    f'<span style="font-size:15px;font-weight:400;">Confidence: '
                    f'{clf_proba[0]*100:.1f}%</span></div>' if clf_proba is not None
                    else '<div class="result-bad">❌ Not a Good Investment</div>',
                    unsafe_allow_html=True
                )

        with r2:
            st.markdown(
                f'<div class="price-card">📈 Estimated Price After 5 Years<br>'
                f'<span style="font-size:28px;">₹ {reg_pred:.1f} Lakhs</span><br>'
                f'<span style="font-size:14px;font-weight:400;">'
                f'(Formula check: ₹{future_price:.1f}L @ 8% p.a.)</span></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ── Confidence gauge ──────────────────────────────
        if clf_proba is not None:
            st.markdown('<div class="section-header">Confidence Breakdown</div>', unsafe_allow_html=True)
            g1, g2 = st.columns(2)

            with g1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=clf_proba[1] * 100,
                    title={"text": "Good Investment Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#10B981" if clf_pred == 1 else "#EF4444"},
                        "steps": [
                            {"range": [0,  40], "color": "#FEE2E2"},
                            {"range": [40, 60], "color": "#FEF9C3"},
                            {"range": [60, 100],"color": "#D1FAE5"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
                    },
                ))
                fig.update_layout(**plot_layout(height=280, margin=dict(t=40, b=10)))
                st.plotly_chart(fig, use_container_width=True)

            with g2:
                appreciation = ((reg_pred - price_lakhs) / price_lakhs) * 100
                fig2 = go.Figure(go.Indicator(
                    mode="number+delta",
                    value=reg_pred,
                    number={"prefix": "₹", "suffix": "L", "font": {"size": 48}},
                    delta={"reference": price_lakhs, "relative": True,
                           "valueformat": ".1%", "suffix": " appreciation"},
                    title={"text": "5-Year Price Forecast"},
                ))
                fig2.update_layout(**plot_layout(height=280, margin=dict(t=60, b=10)))
                st.plotly_chart(fig2, use_container_width=True)

        # ── Price growth chart ────────────────────────────
        st.markdown('<div class="section-header">Price Growth Projection</div>', unsafe_allow_html=True)
        years  = list(range(0, 11))
        prices = [price_lakhs * (1.08 ** y) for y in years]

        fig3 = px.line(
            x=years, y=prices,
            labels={"x": "Years from Now", "y": "Estimated Price (₹ Lakhs)"},
            title=f"10-Year Price Growth Projection — {city}",
            markers=True,
            color_discrete_sequence=["#2563EB"],
        )
        fig3.add_vline(x=5, line_dash="dash", line_color="#10B981",
                       annotation_text="5-yr mark")
        fig3.update_layout(**plot_layout(yaxis_tickprefix="₹", height=380))
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 3 — EDA INSIGHTS
# ═══════════════════════════════════════════════════════════
elif page == "📊  EDA Insights":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("All 20 EDA questions answered visually.")
    st.markdown("---")

    eda_dir = "eda_outputs"
    charts  = sorted(glob.glob(os.path.join(eda_dir, "Q*.png")))

    SECTIONS = {
        "Price & Size Analysis (Q1–Q5)":              ["Q01", "Q02", "Q03", "Q04", "Q05"],
        "Location-based Analysis (Q6–Q10)":           ["Q06", "Q07", "Q08", "Q09", "Q10"],
        "Feature Relationships & Correlation (Q11–Q15)": ["Q11", "Q12", "Q13", "Q14", "Q15"],
        "Investment / Amenities / Ownership (Q16–Q20)":  ["Q16", "Q17", "Q18", "Q19", "Q20"],
    }

    if not charts:
        st.warning("⚠️  No EDA charts found. Run `eda.py` first to generate `eda_outputs/`.")
        if not df.empty:
            st.markdown("**Quick inline charts from loaded dataset:**")
            c1, c2 = st.columns(2)
            if "Price_in_Lakhs" in df.columns:
                with c1:
                    fig = px.histogram(df, x="Price_in_Lakhs", nbins=50,
                                       title="Price Distribution",
                                       color_discrete_sequence=["#2563EB"])
                    fig.update_layout(**plot_layout())
                    st.plotly_chart(fig, use_container_width=True)
            if "Size_in_SqFt" in df.columns:
                with c2:
                    fig = px.histogram(df, x="Size_in_SqFt", nbins=50,
                                       title="Size Distribution",
                                       color_discrete_sequence=["#7C3AED"])
                    fig.update_layout(**plot_layout())
                    st.plotly_chart(fig, use_container_width=True)
    else:
        for section, qids in SECTIONS.items():
            st.markdown(f'<div class="section-header">{section}</div>', unsafe_allow_html=True)
            sec_charts = [c for c in charts if any(os.path.basename(c).startswith(q) for q in qids)]
            for i in range(0, len(sec_charts), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(sec_charts):
                        with col:
                            st.image(sec_charts[i + j], use_container_width=True)

    # ── Inline interactive chart ───────────────────────────
    if not df.empty and "City" in df.columns and "Price_in_Lakhs" in df.columns:
        st.markdown('<div class="section-header">Interactive: City-wise Price Distribution</div>',
                    unsafe_allow_html=True)
        top_cities = df["City"].value_counts().head(12).index
        sub = df[df["City"].isin(top_cities)]
        fig = px.box(sub, x="City", y="Price_in_Lakhs",
                     color="City", points=False,
                     title="Price Distribution — Top 12 Cities",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(**plot_layout(showlegend=False, height=420))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 4 — MODEL REPORT
# ═══════════════════════════════════════════════════════════
elif page == "🤖  Model Report":
    st.title("🤖 Model Performance Report")
    st.markdown("---")

    if not meta:
        st.error("⚠️  Model metadata not found. Run `train_models.py` first.")
        st.stop()

    # ── Classification metrics ────────────────────────────
    st.markdown('<div class="section-header">Classification — Good Investment</div>',
                unsafe_allow_html=True)
    clf_metrics = meta.get("clf_metrics", {})
    if clf_metrics:
        rows = []
        for model_name, m in clf_metrics.items():
            rows.append({
                "Model":     model_name,
                "Accuracy":  round(m.get("accuracy", 0), 4),
                "F1 Score":  round(m.get("f1", 0), 4),
                "Precision": round(m.get("precision", 0), 4),
                "Recall":    round(m.get("recall", 0), 4),
                "CV F1":     round(m.get("cv_f1", 0), 4),
            })
        clf_df = pd.DataFrame(rows)
        best   = meta.get("best_classifier", "")
        clf_df["Best"] = clf_df["Model"].apply(lambda x: "⭐" if x == best else "")
        st.dataframe(clf_df.set_index("Model"), use_container_width=True)

        # Bar chart comparison
        fig = px.bar(clf_df, x="Model", y=["Accuracy", "F1 Score", "Precision", "Recall"],
                     barmode="group", title="Classification Model Comparison",
                     color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(**plot_layout(height=380))
        st.plotly_chart(fig, use_container_width=True)

    # ── Regression metrics ────────────────────────────────
    st.markdown('<div class="section-header">Regression — Future Price (5 yrs)</div>',
                unsafe_allow_html=True)
    reg_metrics = meta.get("reg_metrics", {})
    if reg_metrics:
        rows = []
        for model_name, m in reg_metrics.items():
            rows.append({
                "Model": model_name,
                "RMSE":  round(m.get("RMSE", 0), 4),
                "MAE":   round(m.get("MAE", 0), 4),
                "R²":    round(m.get("R2", 0), 4),
                "MAPE":  round(m.get("MAPE", 0), 2),
                "CV R²": round(m.get("cv_r2", 0), 4),
            })
        reg_df = pd.DataFrame(rows)
        best   = meta.get("best_regressor", "")
        reg_df["Best"] = reg_df["Model"].apply(lambda x: "⭐" if x == best else "")
        st.dataframe(reg_df.set_index("Model"), use_container_width=True)

        fig2 = px.bar(reg_df, x="Model", y=["RMSE", "MAE", "R²"],
                      barmode="group", title="Regression Model Comparison",
                      color_discrete_sequence=px.colors.qualitative.Pastel1)
        fig2.update_layout(**plot_layout(height=380))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Model artefact images (confusion matrix, feature importance) ──
    model_dir = "models"
    img_files = glob.glob(os.path.join(model_dir, "*.png"))
    if img_files:
        st.markdown('<div class="section-header">Model Artefact Charts</div>',
                    unsafe_allow_html=True)
        tabs = st.tabs(["Confusion Matrices", "Feature Importance", "Actual vs Predicted"])
        cm_imgs  = [f for f in img_files if os.path.basename(f).startswith("cm_")]
        fi_imgs  = [f for f in img_files if os.path.basename(f).startswith("fi_")]
        avp_imgs = [f for f in img_files if os.path.basename(f).startswith("avp_")]

        for tab, imgs in zip(tabs, [cm_imgs, fi_imgs, avp_imgs]):
            with tab:
                if not imgs:
                    st.info("Charts will appear here after running train_models.py")
                for i in range(0, len(imgs), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(imgs):
                            with cols[j]:
                                st.image(imgs[i + j], use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 5 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════
elif page == "📁  Data Explorer":
    st.title("📁 Dataset Explorer")
    st.markdown("Filter, sort, and explore the India Housing Prices dataset.")
    st.markdown("---")

    if df.empty:
        st.error("⚠️  Dataset not found. Place `india_housing_prices.csv` in the `data/` folder.")
        st.stop()

    # ── Sidebar filters ───────────────────────────────────
    st.sidebar.markdown("### 🔎 Filters")

    if "State" in df.columns:
        states = st.sidebar.multiselect("State", sorted(df["State"].dropna().unique()),
                                        default=[])
    if "City" in df.columns:
        cities = st.sidebar.multiselect("City", sorted(df["City"].dropna().unique()),
                                        default=[])
    if "BHK" in df.columns:
        bhk_range = st.sidebar.slider("BHK Range", int(df["BHK"].min()),
                                      int(df["BHK"].max()), (1, 4))
    if "Price_in_Lakhs" in df.columns:
        price_range = st.sidebar.slider(
            "Price Range (Lakhs)",
            float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max()),
            (float(df["Price_in_Lakhs"].quantile(0.05)),
             float(df["Price_in_Lakhs"].quantile(0.95))),
        )
    if "Property_Type" in df.columns:
        ptypes = st.sidebar.multiselect("Property Type",
                                        sorted(df["Property_Type"].dropna().unique()),
                                        default=[])

    # Apply filters
    filtered = df.copy()
    if "State" in df.columns and states:
        filtered = filtered[filtered["State"].isin(states)]
    if "City" in df.columns and cities:
        filtered = filtered[filtered["City"].isin(cities)]
    if "BHK" in df.columns:
        filtered = filtered[(filtered["BHK"] >= bhk_range[0]) &
                            (filtered["BHK"] <= bhk_range[1])]
    if "Price_in_Lakhs" in df.columns:
        filtered = filtered[(filtered["Price_in_Lakhs"] >= price_range[0]) &
                            (filtered["Price_in_Lakhs"] <= price_range[1])]
    if "Property_Type" in df.columns and ptypes:
        filtered = filtered[filtered["Property_Type"].isin(ptypes)]

    # ── Summary stats ─────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filtered Rows", f"{len(filtered):,}")
    if "Price_in_Lakhs" in filtered.columns:
        c2.metric("Avg Price (L)", f"₹{filtered['Price_in_Lakhs'].mean():.1f}")
        c3.metric("Min Price (L)", f"₹{filtered['Price_in_Lakhs'].min():.1f}")
        c4.metric("Max Price (L)", f"₹{filtered['Price_in_Lakhs'].max():.1f}")

    # ── Inline charts ─────────────────────────────────────
    if not filtered.empty:
        ch1, ch2 = st.columns(2)
        with ch1:
            if "City" in filtered.columns and "Price_in_Lakhs" in filtered.columns:
                top = filtered.groupby("City")["Price_in_Lakhs"].mean().nlargest(10).reset_index()
                fig = px.bar(top, x="City", y="Price_in_Lakhs",
                             title="Top 10 Cities by Avg Price",
                             color="Price_in_Lakhs", color_continuous_scale="Blues")
                fig.update_layout(**plot_layout(showlegend=False, height=350))
                st.plotly_chart(fig, use_container_width=True)
        with ch2:
            if "Property_Type" in filtered.columns:
                vc = filtered["Property_Type"].value_counts().reset_index()
                vc.columns = ["Property_Type", "Count"]
                fig2 = px.pie(vc, names="Property_Type", values="Count",
                              title="Property Type Distribution",
                              color_discrete_sequence=px.colors.qualitative.Set2)
                fig2.update_layout(**plot_layout(height=350))
                st.plotly_chart(fig2, use_container_width=True)

        if "Good_Investment" in filtered.columns:
            ch3, ch4 = st.columns(2)
            with ch3:
                gi = filtered["Good_Investment"].value_counts().reset_index()
                gi.columns = ["Label", "Count"]
                gi["Label"] = gi["Label"].map({0: "Not Good", 1: "Good Investment"})
                fig3 = px.bar(gi, x="Label", y="Count",
                              color="Label",
                              color_discrete_map={"Good Investment": "#10B981", "Not Good": "#EF4444"},
                              title="Good Investment Distribution")
                fig3.update_layout(**plot_layout(showlegend=False, height=320))
                st.plotly_chart(fig3, use_container_width=True)
            with ch4:
                if "BHK" in filtered.columns and "Price_in_Lakhs" in filtered.columns:
                    fig4 = px.scatter(
                        filtered.sample(min(2000, len(filtered))),
                        x="Size_in_SqFt" if "Size_in_SqFt" in filtered.columns else "BHK",
                        y="Price_in_Lakhs",
                        color="Good_Investment",
                        color_discrete_map={0: "#EF4444", 1: "#10B981"},
                        opacity=0.5, title="Size vs Price (coloured by Investment Label)",
                        labels={"Good_Investment": "Good Inv."},
                    )
                    fig4.update_layout(**plot_layout(height=320))
                    st.plotly_chart(fig4, use_container_width=True)

    # ── Raw table ─────────────────────────────────────────
    st.markdown('<div class="section-header">Raw Data Table</div>', unsafe_allow_html=True)
    show_cols = [c for c in ["State", "City", "Locality", "Property_Type", "BHK",
                              "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
                              "Age_of_Property", "Good_Investment", "Future_Price_5yr"]
                 if c in filtered.columns]
    st.dataframe(
        filtered[show_cols].head(500).reset_index(drop=True),
        use_container_width=True, height=400
    )
    csv = filtered[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data (CSV)", data=csv,
                       file_name="filtered_properties.csv", mime="text/csv")
