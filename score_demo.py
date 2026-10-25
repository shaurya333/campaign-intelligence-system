"""
Score the six worked examples with the saved model (outputs/model_params.json).
Run after train.py:  python score_demo.py
"""
import json

import cis_core as cis

with open("outputs/model_params.json") as f:
    params = json.load(f)

print(f"{'Customer':34s} {'Calls':>5s} {'z':>7s} {'Fatigue':>8s} {'Band':>7s} {'Propensity':>11s}  Action")
for name, c in cis.EXAMPLES.items():
    r = cis.score_from_params(params, c)
    print(f"{name:34s} {c['calls_this_campaign']:5d} {r['fatigue_z']:7.3f} {r['fatigue']:8.1f} "
          f"{r['band']:>7s} {r['propensity']:11.1%}  {r['action']}")
print(f"\nHigh propensity means >= {params['propensity']['high_cut']:.1%} (top 30% of customers).")
print("Tired means fatigue score > 30 (Medium or High band).")
