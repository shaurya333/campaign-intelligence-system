"""Builds CIS_Colab.ipynb from the project files so the notebook never drifts from the code."""
import json

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}

core = open("cis_core.py", encoding="utf-8").read()
train = open("train.py", encoding="utf-8").read()
demo = open("score_demo.py", encoding="utf-8").read()

cells = [
    md("""
# Campaign Intelligence System - working prototype
**DSE3170 Project Based Learning 3**

Two logistic regression models (no tree models):
1. **Fatigue model (main)** - probability that the next call is a brush-off, shown as a score from 0 to 100.
2. **Propensity model (supporting)** - probability that the customer subscribes.

A decision matrix turns the two scores into one action: *contact now, wait then retry, low priority* or *suppress*.

**How to run:** Runtime -> Run all. Colab already has every library this notebook needs.
"""),
    md("## 1. Get the real data (UCI Bank Marketing, 41,188 calls)"),
    code("""
import os, zipfile, urllib.request, io
os.makedirs("data", exist_ok=True); os.makedirs("outputs", exist_ok=True)
URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
try:
    outer = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip")))
    open("data/bank_marketing.csv", "wb").write(inner.read("bank-additional/bank-additional-full.csv"))
    print("Downloaded data/bank_marketing.csv")
except Exception as e:
    print("Download failed:", e)
    print("Upload bank_marketing.csv from the project zip's data/ folder instead:")
    from google.colab import files
    up = files.upload()
    open("data/bank_marketing.csv", "wb").write(next(iter(up.values())))
"""),
    md("## 2. Model code (`cis_core.py`)\nData loading, the brush-off label, both regressions, scoring and the decision matrix."),
    code("%%writefile cis_core.py\n" + core),
    md("## 3. Train both models and print the results (`train.py`)"),
    code("%%writefile train.py\n" + train),
    code("!python train.py"),
    md("## 4. Charts"),
    code("""
from IPython.display import Image, display
for f in ["fatigue_curve", "roc_curves", "propensity_lift", "decision_outcomes"]:
    display(Image(f"outputs/{f}.png", width=700))
"""),
    md("## 5. Six worked examples (`score_demo.py`)"),
    code("%%writefile score_demo.py\n" + demo),
    code("!python score_demo.py"),
    md("## 6. Score your own customer\nChange any value below and run the cell again."),
    code("""
import json, cis_core as cis
params = json.load(open("outputs/model_params.json"))
customer = dict(
    calls_this_campaign=4,      # calls made in this campaign
    earlier_contacts=0,         # contacts in earlier campaigns
    contact="cellular",         # "cellular" or "telephone"
    age=40, job="technician", marital="married", education="university.degree",
    default="no", poutcome="nonexistent", month="may", euribor3m=4.86,
    opted_out=False,
)
r = cis.score_from_params(params, customer)
print(f"Fatigue score : {r['fatigue']:.1f} / 100  ({r['band']})")
print(f"Propensity    : {r['propensity']:.1%}")
print(f"Decision      : {r['action']}")
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python"}, "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("CIS_Colab.ipynb", "w", encoding="utf-8"), indent=1)
print("Wrote CIS_Colab.ipynb")
