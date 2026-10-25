"""
Train both regression models, print the results and save charts + model_params.json.
Run:  python train.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

import cis_core as cis

os.makedirs("outputs", exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})
TEAL, ORANGE, GREY, RED = "#0F766E", "#EA580C", "#94A3B8", "#B91C1C"

# ---------------------------------------------------------------- 1. data
df = cis.load_data()
train, test = cis.split(df)
print(f"Rows: {len(df):,}  (train {len(train):,} / test {len(test):,})")
print(f"Brush-off rate: {df.brush_off.mean():.1%}   Subscription rate: {df.subscribed.mean():.1%}")

# ---------------------------------------------------------------- 2. fatigue model (main)
fat = cis.train_fatigue_model(train)
test = test.copy()
test["fatigue"] = cis.fatigue_score(fat, test)
test["band"] = test["fatigue"].apply(cis.fatigue_band)
fat_auc = roc_auc_score(test.brush_off, test.fatigue)
fat_brier = brier_score_loss(test.brush_off, test.fatigue / 100)

print("\n=== FATIGUE MODEL (main) ===")
print(f"z = {fat.intercept_[0]:+.3f}", end="")
for f, c in zip(cis.FATIGUE_FEATURES, fat.coef_[0]):
    print(f"  {c:+.3f} x {f}", end="")
print()
for f, c in zip(cis.FATIGUE_FEATURES, fat.coef_[0]):
    print(f"  {f:22s} weight {c:+.3f}   odds x {np.exp(c):.2f}")
print(f"AUC {fat_auc:.3f}   Brier {fat_brier:.3f}")

band_table = (test.groupby("band")
              .agg(customers=("brush_off", "size"), predicted=("fatigue", "mean"),
                   actual_brush_off=("brush_off", "mean"))
              .reindex(["Low", "Medium", "High"]))
band_table["share"] = band_table.customers / len(test)
print(band_table.round(3))

# how many customers get flagged at other cut-offs (for the viva question "why so few?")
flag_table = {cut: float((test.fatigue > cut).mean()) for cut in (15, 20, 25, 30)}
print("Share flagged as tired at cut-off:", {k: f"{v:.1%}" for k, v in flag_table.items()})

# ---------------------------------------------------------------- 3. propensity model (support)
prop = cis.train_propensity_model(train)
test["propensity"] = cis.propensity_score(prop, test)
train_prop = cis.propensity_score(prop, train)
high_cut = float(np.quantile(train_prop, 0.70))       # top 30% = "high propensity"
prop_auc = roc_auc_score(test.subscribed, test.propensity)
test["decile"] = pd.qcut(test.propensity.rank(method="first", ascending=False), 10, labels=range(1, 11))
lift = test.groupby("decile", observed=True).subscribed.mean()
base = test.subscribed.mean()

print("\n=== PROPENSITY MODEL (supporting evidence) ===")
print(f"AUC {prop_auc:.3f}   base rate {base:.1%}   top-10% rate {lift.iloc[0]:.1%}  "
      f"(lift {lift.iloc[0] / base:.1f}x)   high-propensity cut {high_cut:.3f}")

# ---------------------------------------------------------------- 4. decision matrix
test["action"] = [cis.decide(p, f, high_cut) for p, f in zip(test.propensity, test.fatigue)]
order = ["CONTACT NOW", "WAIT, THEN RETRY", "LOW PRIORITY", "SUPPRESS"]
decision = (test.groupby("action")
            .agg(customers=("subscribed", "size"), subscribe_rate=("subscribed", "mean"),
                 brush_off_rate=("brush_off", "mean"))
            .reindex(order))
print("\n=== DECISION MATRIX on held-out customers ===")
print(decision.round(3))

# ---------------------------------------------------------------- 5. charts
# 5a. brush-off rate vs number of calls: data vs model
g = test.assign(calls=test.calls_this_campaign.clip(upper=15)).groupby("calls")
actual = g.brush_off.mean()
grid = pd.DataFrame({"calls_this_campaign": range(1, 16), "earlier_contacts": 0, "landline": 0})
grid_l = grid.assign(landline=1)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(actual.index, actual.values * 100, color=GREY, label="Actual brush-off rate (test data)")
ax.plot(grid.calls_this_campaign, cis.fatigue_score(fat, grid), color=TEAL, lw=2.5, marker="o",
        label="Fatigue score - mobile")
ax.plot(grid.calls_this_campaign, cis.fatigue_score(fat, grid_l), color=ORANGE, lw=2.5, marker="o",
        label="Fatigue score - landline")
ax.axhline(30, color=RED, ls="--", lw=1)
ax.text(1, 31.5, "Medium band starts (30)", color=RED, fontsize=9)
ax.set_xlabel("Calls made to the customer in this campaign (15 = 15 or more)")
ax.set_ylabel("Brush-off rate / fatigue score (%)")
ax.set_title("Fatigue rises with every extra call")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig("outputs/fatigue_curve.png", dpi=200); plt.close(fig)

# 5b. ROC curves
fig, ax = plt.subplots(figsize=(5, 4.5))
for y, s, name, col, auc in [(test.brush_off, test.fatigue, "Fatigue model", TEAL, fat_auc),
                             (test.subscribed, test.propensity, "Propensity model", ORANGE, prop_auc)]:
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, color=col, lw=2.5, label=f"{name} (AUC {auc:.2f})")
ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="Random guess (AUC 0.50)")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves on held-out customers")
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig("outputs/roc_curves.png", dpi=200); plt.close(fig)

# 5c. lift by propensity decile
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(lift.index.astype(int), lift.values * 100, color=[ORANGE] + [GREY] * 9)
ax.axhline(base * 100, color=TEAL, ls="--", lw=1.5)
ax.text(10.4, base * 100 + 1, f"Average {base:.1%}", color=TEAL, ha="right", fontsize=9)
for b, v in zip(bars, lift.values):
    ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.8, f"{v:.0%}", ha="center", fontsize=9)
ax.set_xticks(range(1, 11)); ax.set_xlabel("Propensity decile (1 = top 10% of scores)")
ax.set_ylabel("Actual subscription rate (%)")
ax.set_title("Propensity model ranks customers well")
fig.tight_layout(); fig.savefig("outputs/propensity_lift.png", dpi=200); plt.close(fig)

# 5d. outcomes by decision
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(order)); w = 0.38
ax.bar(x - w / 2, decision.subscribe_rate * 100, w, color=TEAL, label="Subscribed")
ax.bar(x + w / 2, decision.brush_off_rate * 100, w, color=ORANGE, label="Brushed off")
for i, (s, b) in enumerate(zip(decision.subscribe_rate, decision.brush_off_rate)):
    ax.text(i - w / 2, s * 100 + 1, f"{s:.0%}", ha="center", fontsize=9)
    ax.text(i + w / 2, b * 100 + 1, f"{b:.0%}", ha="center", fontsize=9)
ax.set_xticks(x, [f"{a}\n(n={n:,})" for a, n in zip(order, decision.customers)], fontsize=9)
ax.set_ylabel("Rate on held-out customers (%)")
ax.set_title("What happened to customers in each decision group")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("outputs/decision_outcomes.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 6. save everything
metrics = {
    "rows": len(df), "train_rows": len(train), "test_rows": len(test),
    "brush_off_rate": float(df.brush_off.mean()), "subscribe_rate": float(df.subscribed.mean()),
    "brush_off_by_calls": {int(k): float(v) for k, v in
                           df.groupby(df.campaign.clip(upper=15)).brush_off.mean().items()},
    "fatigue_auc": float(fat_auc), "fatigue_brier": float(fat_brier),
    "odds_ratios": {f: float(np.exp(c)) for f, c in zip(cis.FATIGUE_FEATURES, fat.coef_[0])},
    "bands": {b: {"share": float(r.share), "predicted": float(r.predicted),
                  "actual": float(r.actual_brush_off), "customers": int(r.customers)}
              for b, r in band_table.iterrows()},
    "flagged_at_cutoff": flag_table,
    "propensity_auc": float(prop_auc), "base_rate": float(base),
    "top_decile_rate": float(lift.iloc[0]), "top_decile_lift": float(lift.iloc[0] / base),
    "decision": {a: {"customers": int(r.customers), "subscribe_rate": float(r.subscribe_rate),
                     "brush_off_rate": float(r.brush_off_rate)} for a, r in decision.iterrows()},
}
cis.export_params(fat, prop, high_cut, metrics)
print("\nSaved outputs/model_params.json and 4 charts in outputs/")
