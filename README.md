# Phishing URL Detection: PhiUSIIL Update

This mirrors the original CSC 572 project (`572_code.zip`), retrained and
re-deployed on the newer, larger PhiUSIIL Phishing URL Dataset (235,795
URLs, 54 features) instead of the original UCI dataset (11,055 URLs, 30
features).

## Files
| File | Purpose |
|---|---|
| `PhiUSIIL_Phishing_URL_Dataset.csv` | The new dataset |
| `Phishing_URL_Detection_PhiUSIIL.ipynb` | Training notebook: EDA, training, evaluation, feature importance, model export |
| `feature_extraction.py` | Converts a live URL into the 50-feature PhiUSIIL schema |
| `app.py` | Streamlit demo app |
| `reference_data/` | Bundled lookup tables `feature_extraction.py` needs (see below) |

## Setup
```
pip install -r requirements.txt
```
Run the notebook top to bottom first (it saves `phishing_model_phiusiil.pkl`
and `feature_columns_phiusiil.pkl` into this folder), then:
```
streamlit run app.py
```

## What changed vs. the original project

**Dataset & label convention.** PhiUSIIL uses `label = 1` for legitimate
(opposite of the old UCI dataset) — remapped throughout to the same
`1 = phishing, 0 = legitimate` convention used in your thesis.

**Feature scaling added** to the notebook for Logistic Regression and SVM.
The old dataset was already on a uniform -1/0/1 scale; PhiUSIIL's features
are on very different scales, and without scaling, SVM performed near
chance-level and Logistic Regression failed to converge (verified during
development).

**No more WHOIS/DNS dependency.** PhiUSIIL's feature set doesn't use
domain age or DNS records, so `feature_extraction.py` only needs `requests`,
`beautifulsoup4`, and `tldextract` — a smaller/more reliable dependency
surface than the old extractor. (Also: the original `requirements.txt`
never actually listed `tldextract`, even though the old `feature_extraction.py`
imported it — fixed here.)

**Results are near-perfect across all five models** (99.9%+ F1). This is a
documented property of PhiUSIIL's feature set, not a bug — worth naming
explicitly in your write-up.

### `URLSimilarityIndex`: an important, genuine limitation

This is worth reading even if you skip everything else. `URLSimilarityIndex`
is the single strongest predictor in the trained model. While digging into
how to reproduce it live, I found that in the training data, **every one of
the 134,850 legitimate rows has this value at exactly 100, with zero
variance**, and all 134,850 legitimate domains are unique. That strongly
suggests the dataset authors assigned 100 once a URL was already known to
be legitimate, rather than computing a genuine similarity score against
some independent reference — i.e., for the legitimate class, this feature
behaves like an oracle rather than a measurement.

That can't be honestly reproduced for a URL nobody has vetted yet. This
script's best available stand-in: an exact match against a bundled list of
~20,000 known-legitimate domains (sampled from the training data) scores
100; anything else falls back to fuzzy string matching, scaled to 0-100.
In testing, this systematically under-scores real, legitimate, less-common
sites — e.g. a realistic synthetic Wikipedia-style page still came back
~83% phishing confidence purely because of this one feature. **This is a
property of the PhiUSIIL dataset, not a bug in this code** — and it's a
substantive point for your thesis: near-100% benchmark accuracy on
held-out PhiUSIIL data doesn't necessarily mean the trained model will
perform that well against arbitrary new URLs in the real world, since its
single strongest feature can't be computed the same way outside the
dataset's closed construction process.

**Two smaller approximations**, also documented inline in
`feature_extraction.py`:
- `TLDLegitimateProb` — verified to match a per-TLD frequency table
  (P(TLD | legitimate)) computed from the training data at 0.995
  correlation, so this one's a faithful reconstruction.
- `URLCharProb` — approximated via a character-frequency model built from
  the training data; the paper's exact method isn't public.

### Testing note
This was developed and unit-tested in a sandbox with no internet access,
so every piece that touches the network (fetching the live page) could
only be tested with synthetic HTML rather than a real request — the
lexical/HTML-parsing logic was verified this way, but please sanity-check
a handful of real URLs yourself once you run it locally.

### Not carried over
Section 10 from the original notebook ("Try it on a new URL" as an inline
cell) is replaced by this standalone `app.py`/`feature_extraction.py` pair
instead of a notebook cell, since that's the natural equivalent for a
different, larger feature schema.
