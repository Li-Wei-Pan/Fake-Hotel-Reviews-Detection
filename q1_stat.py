#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2



# In[2]:


ALPHA = 0.01
def load_predictions(path):
    return pd.read_csv(path).iloc[:, 0].to_numpy()

def mcnemar_test(y_true, pred_a, pred_b, name_a="Model A", name_b="Model B",
                 alpha=ALPHA, exact_threshold=25):
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, "
            f"{name_a}={len(pred_a)}, {name_b}={len(pred_b)}"
        )

    a_ok = pred_a == y_true
    b_ok = pred_b == y_true

    n11 = int(np.sum(a_ok & b_ok))    # both correct
    n10 = int(np.sum(a_ok & ~b_ok))   # only A correct
    n01 = int(np.sum(~a_ok & b_ok))   # only B correct
    n00 = int(np.sum(~a_ok & ~b_ok))  # both wrong

    table = pd.DataFrame(
        [[n11, n10], [n01, n00]],
        index=[f"{name_a} correct", f"{name_a} wrong"],
        columns=[f"{name_b} correct", f"{name_b} wrong"],
    )

    n_disagree = n01 + n10
    if n_disagree == 0:
        statistic, p_value, method = 0.0, 1.0, "no disagreements"
    elif n_disagree < exact_threshold:
        statistic = min(n01, n10)
        p_value = binomtest(n10, n_disagree, 0.5).pvalue
        method = "exact binomial"
    else:
        statistic = (abs(n01 - n10) - 1) ** 2 / n_disagree
        p_value = chi2.sf(statistic, df=1)
        method = "chi-square with continuity correction"

    significant = p_value < alpha

    print(f"\n=== McNemar's test: {name_a} vs {name_b} ===")
    print(table.to_string())
    print(f"Accuracy {name_a}: {a_ok.mean():.3f} | Accuracy {name_b}: {b_ok.mean():.3f}")
    print(f"Disagreements: only {name_a} correct = {n10}, only {name_b} correct = {n01}")
    print(f"Method: {method}")
    print(f"Statistic = {statistic:.4f}, p-value = {p_value:.4f}")
    verdict = "significant" if significant else "not significant"
    print(f"Difference in accuracy is {verdict} at alpha = {alpha}")

    return {
        "table": table,
        "n11": n11, "n10": n10, "n01": n01, "n00": n00,
        "statistic": statistic,
        "p_value": p_value,
        "method": method,
        "significant": significant,
    }


def table_to_latex(result, name_a, name_b):
    """Return the 2x2 table as a LaTeX tabular for the report."""
    return (
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        f" & {name_b} correct & {name_b} wrong \\\\\n"
        "\\midrule\n"
        f"{name_a} correct & {result['n11']} & {result['n10']} \\\\\n"
        f"{name_a} wrong & {result['n01']} & {result['n00']} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )


if __name__ == "__main__":
    # True labels come from the saved test split; predictions were made in the same row order.
    y_test = pd.read_csv("test.csv")["source"].to_numpy()

    y_pred_lr = load_predictions("data/y_pred_lg.csv")
    y_pred_mnb = load_predictions("data/y_pred_mnb.csv")

    result = mcnemar_test(y_test, y_pred_mnb, y_pred_lr, name_a="MNB", name_b="LR")
    print("\nLaTeX table:\n")
    print(table_to_latex(result, "MNB", "LR"))


# In[ ]:




