# Campaign Intelligence System

**DSE3170 Project Based Learning 3 · Manipal University Jaipur**
Adit Tanted (2430010585) · Shaurya Bhardwaj (2430010592) · Guide: Dr. Ram Kumar Yadav

The system decides **whom to contact and when to hold back**:

1. **Fatigue model (main):** a logistic regression that turns a customer's call history into a score from 0 to 100 (Low 0–30, Medium 31–60, High 61–100).
2. **Propensity model (supporting):** a logistic regression that gives the chance the customer subscribes.
3. **Decision matrix:** combines the two scores into CONTACT NOW, WAIT THEN RETRY, LOW PRIORITY or SUPPRESS. An opt-out always overrides both.

Both models are plain regressions, with no tree models, so every weight can be explained.

## Run the web app
```
pip install -r requirements.txt
streamlit run app.py
```
The app reads the trained weights from `outputs/model_params.json`, so it starts instantly.

## Retrain the models
```
python train.py        # trains both models, prints results, saves charts + outputs/model_params.json
python score_demo.py   # scores the six worked examples A–F
```

**Google Colab:** open `CIS_Colab.ipynb` in Colab, then click Runtime → Run all.

## Data
UCI Bank Marketing (`data/bank_marketing.csv`): 41,188 real phone calls from a Portuguese bank.
S. Moro, P. Cortez and P. Rita, "A data-driven approach to predict the success of bank telemarketing," *Decision Support Systems* 62 (2014) 22–31.

Fatigue label: a **brush-off** is a call shorter than 60 seconds that did not end in a subscription.

## Results (8,238 held-out customers)
- Each extra call multiplies the odds of a brush-off by about 1.18. The fatigue model's AUC is 0.65.
- Propensity model: AUC 0.80. The top 10% of customers subscribe at 48.8%, against an average of 11.3%.
- CONTACT NOW customers subscribed at 27.1%. SUPPRESS customers subscribed at 3.7% and brushed off 46% of calls.

## Files
| File | What it does |
|---|---|
| `app.py` | Streamlit web app (live demo, how it works, results) |
| `cis_core.py` | Loads data, builds the label, trains both models, scores, decides |
| `train.py` | Trains, prints results, saves charts and weights |
| `score_demo.py` | Scores six worked examples |
| `CIS_Colab.ipynb` | Whole project as one notebook (built by `make_notebook.py`) |


