# Customer Churn Prediction

Predicting whether a telecom customer will churn, using 11 features
(age, tenure, support calls, payment delay, spend, contract type, …) and
comparing five classifiers.

**Data:** `customer_churn_dataset-training-master.csv` — 440,833 rows, included
in this repo.

## Results

20% held-out test set (88,167 rows), `random_state=42` throughout, nominal
columns one-hot encoded:

| Model | Test accuracy | Train accuracy |
|---|---|---|
| Naive Bayes | 0.8656 | 0.8650 |
| K-Neighbours | 0.8917 | 0.9243 |
| Logistic Regression | 0.8964 | 0.8951 |
| **Random Forest** | 0.9991 | 1.0000 |
| **Decision Tree** | **0.9999** | 1.0000 |

The notebook selects the Decision Tree (99.99%). **But that number should not be
read as churn being 99.99% predictable — see the next section.** The notebook
itself contains this audit as its final section, so the analysis and the
conclusion agree.

## Important: the 99.99% is a property of this dataset

A churn model in the real world tops out around 75–85%. Before quoting 99.99%
anywhere, note what is actually happening:

**1. The label is close to a deterministic rule.** A decision tree needs only
**102 leaves** to classify 440,832 rows at 99.99% — and it is already at 95.7%
by depth 4. Real churn needs thousands of leaves and still does not get near this.

**2. Some rules are exactly perfect, not just accurate:**

```python
Support Calls > 5.5                   -> churn rate 1.000000  (118,513 rows)
Support Calls <= 4 and Spend < 499.93 -> churn rate 1.000000  (52,699 rows)
```

Not one exception in 118,513 rows. Human behaviour does not behave like that.

**3. Every numeric feature is uniform.** Age 18–65, tenure 1–60, usage 1–30,
spend 100–1000 — all flat, skew ≈ 0, mean ≈ median. Real customer data is
clustered and skewed, never uniform. This is the signature of a synthetic
generator.

So the high accuracy is a property of how the data was *generated*, not evidence
that the model learned something generalisable. It is not overfitting either —
train and test accuracy match — which is exactly what makes it misleading.

**What this means in practice:** Logistic Regression (0.8964) is the most
believable figure in the table, and this dataset cannot tell you how any of these
models would perform on real customers. If you present this project, present the
diagnosis, not the 99.99% — the diagnosis is the more interesting result.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook "Churn Pred.ipynb"
```

The notebook reads the CSV by relative path, so it runs from a fresh clone. Each
model is trained once, and a full run takes a few minutes.

## Layout

```
Churn Pred.ipynb                          the analysis, executed with outputs saved
customer_churn_dataset-training-master.csv
requirements.txt
```

## Method

1. Drop `CustomerID` — an identifier, not a feature.
2. Drop the single row containing nulls.
3. Confirm no duplicate rows.
4. One-hot encode the nominal columns (`Gender`, `Subscription Type`,
   `Contract Length`).
5. 80/20 train-test split, `random_state=42`.
6. Train and compare Logistic Regression, Decision Tree, Random Forest,
   K-Neighbours and Naive Bayes once each, printing a confusion matrix and
   per-class report for every one, then table the results.

## Corrections made

An earlier revision had four rough edges; all four are fixed:

- **The null row was dropped by hardcoded position** (`df.drop(199295)`), which
  deletes the wrong row — or none — if the CSV changes. Now `df.dropna()`.
- **The models were trained twice** (once for the confusion matrices, once for
  the results table). Now a single loop does both.
- **`SVC` was imported and never used.** Removed.
- **`LabelEncoder` was applied to nominal columns**, which imposes a false order
  (`Basic=1 < Premium=2 < Standard=3`). Now one-hot encoded.

Two consequences of the encoding change are worth knowing:

- Logistic Regression improved from 0.8485 to 0.8964 — a linear model benefits
  from not being told that Premium sits "between" Basic and Standard.
- **Naive Bayes got worse, 0.9094 to 0.8656.** `GaussianNB` assumes each feature
  is normally distributed, and 0/1 dummy columns violate that assumption more
  than a single ordinal integer did. A one-hot change is not neutral across model
  families.

`random_state` is also now fixed on the split and on every model, so these
numbers reproduce; previously the tree-based models gave a slightly different
result on each run.

One thing deliberately left alone: features are not scaled, which K-Neighbours is
sensitive to. It makes no difference here because the label is near-deterministic,
but on a real dataset KNN should be run on standardised features.
