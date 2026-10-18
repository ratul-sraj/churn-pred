# Customer Churn Prediction

Predicting whether a telecom customer will churn, using 11 features
(age, tenure, support calls, payment delay, spend, contract type, …) and
comparing five classifiers.

**Data:** `customer_churn_dataset-training-master.csv` — 440,833 rows, included
in this repo.

## Results

20% held-out test set, `random_state=42`:

| Model | Test accuracy | Train accuracy |
|---|---|---|
| Logistic Regression | 0.8485 | 0.8502 |
| K-Neighbours | 0.8912 | 0.9240 |
| Naive Bayes | 0.9094 | 0.9094 |
| **Random Forest** | 0.9997 | 1.0000 |
| **Decision Tree** | **0.9998** | 1.0000 |

The notebook selects the Decision Tree (99.98%). **But that number should not be
read as churn being 99.98% predictable — see the next section.** The notebook
itself now contains this audit as its final section, so the analysis and the
conclusion agree.

## Important: the 99.98% is a property of this dataset

A churn model in the real world tops out around 75–85%. Before quoting 99.98%
anywhere, note what is actually happening:

**1. The label is close to a deterministic rule.** A decision tree needs only
**176 leaves** to classify 440,832 rows at 99.99%. Real churn needs thousands of
leaves and still does not get near this.

**2. Some rules are exactly perfect, not just accurate:**

```python
Support Calls > 5.5                  -> churn rate 1.000000  (118,513 rows)
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

**What this means in practice:** Logistic Regression (0.8485) is the most
believable figure in the table, and this dataset cannot tell you how any of these
models would perform on real customers. If you present this project, present the
diagnosis, not the 99.98% — the diagnosis is the more interesting result.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook "Churn Pred.ipynb"
```

The notebook reads the CSV by relative path, so it runs from a fresh clone. It
trains five models twice (once to display confusion matrices, once to build the
results table) — it takes a few minutes, and K-Neighbours is by far the slowest
step because it compares every test row against every training row.

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
4. Label-encode the categorical columns (`Gender`, `Subscription Type`,
   `Contract Length`).
5. 80/20 train-test split.
6. Train and compare Logistic Regression, Decision Tree, Random Forest,
   K-Neighbours and Naive Bayes on accuracy, with a confusion matrix and
   per-class report for each.

## Known rough edges

- The null row is dropped by hardcoded position (`df.drop(199295)`) rather than
  `df.dropna()`, so it breaks if the CSV changes.
- `SVC` is imported but never used.
- The models are trained twice; the first loop could be folded into the second.
- `LabelEncoder` on nominal categories (`Subscription Type`, `Contract Length`)
  imposes an arbitrary order. One-hot encoding would be more correct, though it
  makes no difference here given the label is essentially deterministic.
