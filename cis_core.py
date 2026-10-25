"""
Campaign Intelligence System (CIS) - core logic.

Two logistic regression models, no tree models:
  1. Fatigue model (main)        - how likely is the customer to brush off the next call?
  2. Propensity model (support)  - how likely is the customer to subscribe?
A decision matrix then combines the two scores into one action.

Data: UCI Bank Marketing (Moro, Cortez & Rita, 2014), 41,188 phone contacts.
"""
import json
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# ---------------------------------------------------------------- settings
# A "brush-off" is a call that lasted under 60 seconds and did not lead to a sale.
BRUSH_OFF_SECONDS = 60

# Fatigue model inputs: the customer's contact history only.
FATIGUE_FEATURES = ["calls_this_campaign", "earlier_contacts", "landline"]

# Propensity model inputs: who the customer is and the market at the time.
# Call duration is NOT used because it is only known after the call (leakage).
PROP_NUMERIC = ["age", "euribor3m"]
PROP_CATEGORICAL = ["job", "marital", "education", "default", "poutcome", "month", "contact"]

# Score bands (same as the report): 0-30 Low, 31-60 Medium, 61-100 High.
LOW_MAX, MEDIUM_MAX = 30, 60


# ---------------------------------------------------------------- data
def load_data(path="data/bank_marketing.csv"):
    """Read the raw UCI file and add the columns the models need."""
    df = pd.read_csv(path, sep=";")
    df.columns = [c.replace(".", "_") for c in df.columns]

    df["subscribed"] = (df["y"] == "yes").astype(int)
    df["calls_this_campaign"] = df["campaign"]          # includes the current call
    df["earlier_contacts"] = df["previous"]             # contacts in earlier campaigns
    df["landline"] = (df["contact"] == "telephone").astype(int)
    df["brush_off"] = ((df["duration"] < BRUSH_OFF_SECONDS) & (df["subscribed"] == 0)).astype(int)
    return df


def split(df):
    """80/20 split, stratified on the outcome, fixed seed so results repeat."""
    return train_test_split(df, test_size=0.2, random_state=RANDOM_STATE,
                            stratify=df["subscribed"])


# ---------------------------------------------------------------- models
def train_fatigue_model(train):
    """Plain logistic regression on raw (unscaled) counts, so each weight reads
    directly as 'change in log-odds per extra call / contact'."""
    model = LogisticRegression(max_iter=1000)
    model.fit(train[FATIGUE_FEATURES], train["brush_off"])
    return model


def train_propensity_model(train):
    """Logistic regression with one-hot encoded categories and scaled numbers."""
    prep = ColumnTransformer([
        ("num", StandardScaler(), PROP_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), PROP_CATEGORICAL),
    ])
    model = Pipeline([("prep", prep), ("lr", LogisticRegression(max_iter=2000, C=1.0))])
    model.fit(train[PROP_NUMERIC + PROP_CATEGORICAL], train["subscribed"])
    return model


# ---------------------------------------------------------------- scoring
def fatigue_score(model, X):
    """Fatigue score = P(brush-off) x 100, i.e. 1 / (1 + e^-z) x 100."""
    return model.predict_proba(X[FATIGUE_FEATURES])[:, 1] * 100


def propensity_score(model, X):
    return model.predict_proba(X[PROP_NUMERIC + PROP_CATEGORICAL])[:, 1]


def fatigue_band(score):
    if score <= LOW_MAX:
        return "Low"
    if score <= MEDIUM_MAX:
        return "Medium"
    return "High"


def decide(propensity, fatigue, high_prop_cut, opted_out=False):
    """Decision matrix: propensity (high/low) x fatigue (rested/tired)."""
    if opted_out:
        return "DO NOT CONTACT"
    high = propensity >= high_prop_cut
    tired = fatigue > LOW_MAX
    if high and not tired:
        return "CONTACT NOW"
    if high and tired:
        return "WAIT, THEN RETRY"
    if tired:
        return "SUPPRESS"
    return "LOW PRIORITY"


# ---------------------------------------------------------------- export
def export_params(fat_model, prop_model, high_prop_cut, metrics, path="outputs/model_params.json"):
    """Write every learned number to JSON, so the web demo can score customers
    with exactly the same maths as Python (no server needed)."""
    prep = prop_model.named_steps["prep"]
    lr = prop_model.named_steps["lr"]
    scaler = prep.named_transformers_["num"]
    encoder = prep.named_transformers_["cat"]
    coefs = lr.coef_[0]

    numeric = {}
    for i, name in enumerate(PROP_NUMERIC):
        numeric[name] = {"mean": float(scaler.mean_[i]), "std": float(scaler.scale_[i]),
                         "coef": float(coefs[i])}
    categorical, k = {}, len(PROP_NUMERIC)
    for name, levels in zip(PROP_CATEGORICAL, encoder.categories_):
        categorical[name] = {}
        for level in levels:
            categorical[name][str(level)] = float(coefs[k])
            k += 1

    params = {
        "fatigue": {
            "intercept": float(fat_model.intercept_[0]),
            "coef": {f: float(c) for f, c in zip(FATIGUE_FEATURES, fat_model.coef_[0])},
            "bands": {"low_max": LOW_MAX, "medium_max": MEDIUM_MAX},
        },
        "propensity": {
            "intercept": float(lr.intercept_[0]),
            "numeric": numeric,
            "categorical": categorical,
            "high_cut": float(high_prop_cut),
        },
        "metrics": metrics,
    }
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return params


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def score_from_params(params, customer):
    """Score one customer using only the saved JSON numbers (what app.py and the
    web demo do). customer is a dict with the propensity fields plus
    calls_this_campaign, earlier_contacts and contact."""
    f = params["fatigue"]
    landline = 1 if customer["contact"] == "telephone" else 0
    z_fat = (f["intercept"]
             + f["coef"]["calls_this_campaign"] * customer["calls_this_campaign"]
             + f["coef"]["earlier_contacts"] * customer["earlier_contacts"]
             + f["coef"]["landline"] * landline)
    fatigue = sigmoid(z_fat) * 100

    p = params["propensity"]
    z_prop = p["intercept"]
    for name, n in p["numeric"].items():
        z_prop += n["coef"] * (customer[name] - n["mean"]) / n["std"]
    for name, levels in p["categorical"].items():
        z_prop += levels.get(str(customer[name]), 0.0)
    propensity = sigmoid(z_prop)

    action = decide(propensity, fatigue, p["high_cut"], customer.get("opted_out", False))
    return {"fatigue_z": z_fat, "fatigue": fatigue, "band": fatigue_band(fatigue),
            "propensity": propensity, "action": action}


# Six worked examples used by score_demo.py, the Streamlit app and the web demo.
# E and F are the same person: F is E after 15 calls.
EXAMPLES = {
    "A - past buyer, first call": dict(age=35, job="admin.", marital="single", education="university.degree",
        default="no", poutcome="success", month="oct", contact="cellular", euribor3m=1.0,
        calls_this_campaign=1, earlier_contacts=2),
    "B - average customer": dict(age=45, job="blue-collar", marital="married", education="basic.9y",
        default="unknown", poutcome="nonexistent", month="may", contact="telephone", euribor3m=4.86,
        calls_this_campaign=3, earlier_contacts=0),
    "C - called 10 times on landline": dict(age=50, job="technician", marital="married", education="professional.course",
        default="no", poutcome="nonexistent", month="may", contact="telephone", euribor3m=4.96,
        calls_this_campaign=10, earlier_contacts=0),
    "D - good prospect, over-called": dict(age=29, job="student", marital="single", education="high.school",
        default="no", poutcome="failure", month="apr", contact="cellular", euribor3m=1.4,
        calls_this_campaign=12, earlier_contacts=0),
    "E - retiree, first call": dict(age=62, job="retired", marital="married", education="basic.4y",
        default="no", poutcome="nonexistent", month="mar", contact="cellular", euribor3m=0.9,
        calls_this_campaign=1, earlier_contacts=0),
    "F - same retiree after 15 calls": dict(age=62, job="retired", marital="married", education="basic.4y",
        default="no", poutcome="nonexistent", month="mar", contact="cellular", euribor3m=0.9,
        calls_this_campaign=15, earlier_contacts=0),
}
