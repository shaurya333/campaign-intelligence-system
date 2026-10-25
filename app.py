"""
Campaign Intelligence System - Streamlit web app.
Run locally:  streamlit run app.py
Reads outputs/model_params.json (written by train.py), so no training happens here.
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import cis_core as cis

HERE = Path(__file__).parent
st.set_page_config(page_title="Campaign Intelligence System", page_icon="📞", layout="wide")


@st.cache_data
def load_params():
    return json.loads((HERE / "outputs" / "model_params.json").read_text())


params = load_params()
F, P, M = params["fatigue"], params["propensity"], params["metrics"]

NICE = {
    "job": {"admin.": "Admin", "blue-collar": "Blue-collar", "entrepreneur": "Entrepreneur", "housemaid": "Housemaid",
            "management": "Management", "retired": "Retired", "self-employed": "Self-employed", "services": "Services",
            "student": "Student", "technician": "Technician", "unemployed": "Unemployed", "unknown": "Unknown"},
    "marital": {"divorced": "Divorced", "married": "Married", "single": "Single", "unknown": "Unknown"},
    "education": {"basic.4y": "Basic, 4 years", "basic.6y": "Basic, 6 years", "basic.9y": "Basic, 9 years",
                  "high.school": "High school", "illiterate": "Illiterate", "professional.course": "Professional course",
                  "university.degree": "University degree", "unknown": "Unknown"},
    "default": {"no": "No", "unknown": "Unknown", "yes": "Yes"},
    "poutcome": {"nonexistent": "Not in a past campaign", "failure": "Did not buy", "success": "Bought"},
    "month": {"mar": "March", "apr": "April", "may": "May", "jun": "June", "jul": "July", "aug": "August",
              "sep": "September", "oct": "October", "nov": "November", "dec": "December"},
}
LABELS = {"job": "Job", "marital": "Marital status", "education": "Education", "default": "Credit in default",
          "poutcome": "Last campaign result", "month": "Month of call"}
WHY = {
    "CONTACT NOW": "Good chance of a sale and the customer is not tired of calls. Call as planned.",
    "WAIT, THEN RETRY": "Worth contacting, but the call history says they are tired. Pause, then try again.",
    "LOW PRIORITY": "Not tired, but unlikely to buy. Include only in broad, low-cost campaigns.",
    "SUPPRESS": "Unlikely to buy and already tired. More calls would only add brush-offs.",
    "DO NOT CONTACT": "The customer opted out. An opt-out overrides both scores.",
}

st.title("Campaign Intelligence System")
st.caption("Decides whom to call and when to hold back. Fatigue model (main) + propensity model (supporting), "
           "both logistic regression, trained on 41,188 real bank phone calls (UCI Bank Marketing).")

demo, how, results = st.tabs(["Live demo", "How it works", "Model results"])

# ======================================================================= live demo
with demo:
    example = st.radio("Start from a worked example", list(cis.EXAMPLES), horizontal=True)
    c = dict(cis.EXAMPLES[example])

    left, right = st.columns([5, 7], gap="large")
    with left:
        st.subheader("Contact history")
        st.caption("Inputs to the fatigue model")
        c["calls_this_campaign"] = st.slider("Calls made in this campaign", 1, 30, c["calls_this_campaign"])
        c["earlier_contacts"] = st.slider("Contacts in earlier campaigns", 0, 7, c["earlier_contacts"])
        c["contact"] = st.radio("Channel", ["cellular", "telephone"], index=["cellular", "telephone"].index(c["contact"]),
                                format_func=lambda v: "Mobile" if v == "cellular" else "Landline", horizontal=True)
        c["opted_out"] = st.checkbox("Customer has opted out")

        st.subheader("Customer profile")
        st.caption("Inputs to the propensity model")
        c["age"] = st.slider("Age", 18, 95, c["age"])
        a, b = st.columns(2)
        for i, field in enumerate(["job", "marital", "education", "default", "poutcome", "month"]):
            options = list(NICE["month"]) if field == "month" else list(P["categorical"][field])
            with (a if i % 2 == 0 else b):
                c[field] = st.selectbox(LABELS[field], options, index=options.index(c[field]),
                                        format_func=lambda v, f=field: NICE[f].get(v, v))
        c["euribor3m"] = st.slider("Euribor 3-month interest rate (%)", 0.6, 5.1, float(c["euribor3m"]), 0.01)

    r = cis.score_from_params(params, c)
    with right:
        action = r["action"]
        box = {"CONTACT NOW": st.success, "WAIT, THEN RETRY": st.warning, "LOW PRIORITY": st.info,
               "SUPPRESS": st.error, "DO NOT CONTACT": st.error}[action]
        box(f"### {action.capitalize()}\n{WHY[action]}")

        m1, m2 = st.columns(2)
        m1.metric("Fatigue score (main model)", f"{r['fatigue']:.1f} / 100")
        m1.caption(f"**{r['band']} band** (Low 0–30, Medium 31–60, High 61–100)")
        m1.progress(min(r["fatigue"] / 100, 1.0))
        high = r["propensity"] >= P["high_cut"]
        m2.metric("Chance of subscribing (supporting)", f"{r['propensity']:.1%}")
        m2.caption(f"**{'High' if high else 'Low'} propensity** (high from {P['high_cut']:.1%})")
        m2.progress(min(r["propensity"], 1.0))

        st.markdown("**How the fatigue score was worked out**")
        land = int(c["contact"] == "telephone")
        st.code(
            f"z = {F['intercept']:.3f} + {F['coef']['calls_this_campaign']:.3f} x {c['calls_this_campaign']} calls"
            f" - {abs(F['coef']['earlier_contacts']):.3f} x {c['earlier_contacts']} earlier"
            f" + {F['coef']['landline']:.3f} x {land} landline = {r['fatigue_z']:.3f}\n"
            f"score = 100 / (1 + e^-z) = {r['fatigue']:.1f}  ->  {r['band']} band", language=None)

        st.markdown("**This customer's fatigue as calls go up**")
        curve = pd.DataFrame({"Calls": range(1, 31)})
        curve["Fatigue score"] = [cis.sigmoid(F["intercept"] + F["coef"]["calls_this_campaign"] * n
                                              + F["coef"]["earlier_contacts"] * c["earlier_contacts"]
                                              + F["coef"]["landline"] * land) * 100 for n in curve.Calls]
        actual = pd.DataFrame({"Calls": [int(k) for k in M["brush_off_by_calls"]],
                               "Actual brush-off rate": [v * 100 for v in M["brush_off_by_calls"].values()]})
        bands = pd.DataFrame({"from": [0, 30, 60], "to": [30, 60, 100], "Band": ["Low", "Medium", "High"]})
        band_layer = alt.Chart(bands).mark_rect(opacity=0.18).encode(
            y=alt.Y("from:Q", scale=alt.Scale(domain=[0, 100]), title="Score / rate (%)"), y2="to:Q",
            color=alt.Color("Band:N", scale=alt.Scale(domain=["Low", "Medium", "High"],
                                                      range=["#0F766E", "#EAB308", "#C2410C"]), legend=None))
        line = alt.Chart(curve).mark_line(color="#0F766E", strokeWidth=3).encode(x=alt.X("Calls:Q", title="Calls in this campaign"),
                                                                                 y="Fatigue score:Q")
        dots = alt.Chart(actual).mark_circle(color="#64748B", size=45).encode(x="Calls:Q", y="Actual brush-off rate:Q",
                                                                             tooltip=["Calls", "Actual brush-off rate"])
        now = alt.Chart(pd.DataFrame({"Calls": [min(c["calls_this_campaign"], 30)], "Fatigue score": [r["fatigue"]]})) \
            .mark_point(color="#12232A", size=160, filled=True).encode(x="Calls:Q", y="Fatigue score:Q")
        st.altair_chart((band_layer + line + dots + now).properties(height=280), width="stretch")
        st.caption("Line: fatigue score for this customer. Grey dots: real brush-off rate in the data. "
                   "Black dot: where the customer is now.")

        st.markdown("**Decision matrix**")
        st.table(pd.DataFrame({"Rested (fatigue ≤ 30)": ["CONTACT NOW", "LOW PRIORITY"],
                               "Tired (fatigue > 30)": ["WAIT, THEN RETRY", "SUPPRESS"]},
                              index=["High propensity", "Low propensity"]))

# ======================================================================= how it works
with how:
    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Pipeline")
        st.markdown(f"""
1. **Load real data:** {M['rows']:,} phone calls from a Portuguese bank (UCI Bank Marketing, Moro et al. 2014).
2. **Make the fatigue label:** a call is a *brush-off* if it lasted under 60 seconds and did not lead to a sale.
3. **Split:** 80% to train, 20% ({M['test_rows']:,} customers) held out to test.
4. **Train two logistic regressions:** fatigue (calls, earlier contacts, landline) and propensity (age, job,
   marital status, education, default, last result, month, channel, interest rate).
5. **Decide:** fatigue score banded 0–30 / 31–60 / 61–100, propensity "high" in the top 30%, decision matrix.
6. **Serve:** the learned weights are saved to `model_params.json`, and this app scores customers with them.
""")
        st.image(str(HERE / "outputs" / "architecture.png"))
    with b:
        st.subheader("Tech stack")
        st.table(pd.DataFrame({"Used for": ["All model code", "Data, features and the label",
                                            "LogisticRegression, OneHotEncoder, StandardScaler, AUC",
                                            "Charts", "This web app", "Code hosting and app hosting",
                                            "Run the notebook in the browser"]},
                              index=["Python 3", "pandas, NumPy", "scikit-learn", "matplotlib, Altair", "Streamlit",
                                     "GitHub + Streamlit Cloud", "Google Colab"]))
        st.subheader("Code map")
        st.table(pd.DataFrame({"What it does": ["Load data, make the label, train, score, decide",
                                                "Train both models, print results, save charts and weights",
                                                "Score the six worked examples", "This Streamlit app",
                                                "The whole project as one Colab notebook"]},
                              index=["cis_core.py", "train.py", "score_demo.py", "app.py", "CIS_Colab.ipynb"]))

# ======================================================================= results
with results:
    k = st.columns(4)
    k[0].metric("Fatigue model AUC", f"{M['fatigue_auc']:.2f}")
    k[1].metric("Propensity model AUC", f"{M['propensity_auc']:.2f}")
    k[2].metric("Top 10% subscribe", f"{M['top_decile_rate']:.1%}")
    k[2].caption(f"{M['top_decile_lift']:.1f}x the average of {M['base_rate']:.1%}")
    k[3].metric("Contact-now group subscribed", f"{M['decision']['CONTACT NOW']['subscribe_rate']:.1%}")
    k[3].caption(f"Suppress group: {M['decision']['SUPPRESS']['subscribe_rate']:.1%}")
    a, b = st.columns(2, gap="large")
    a.image(str(HERE / "outputs" / "fatigue_curve.png"), caption="Fatigue score follows the real brush-off rate")
    b.image(str(HERE / "outputs" / "decision_outcomes.png"), caption="Outcomes of each decision group (test data)")
    a.image(str(HERE / "outputs" / "propensity_lift.png"), caption="Subscription rate by propensity decile")
    b.image(str(HERE / "outputs" / "roc_curves.png"), caption="ROC curves of both models")

st.caption("Data: S. Moro, P. Cortez and P. Rita, Decision Support Systems 62 (2014) 22–31. "
           "DSE3170 Project Based Learning 3, Manipal University Jaipur.")
