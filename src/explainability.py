from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance


def global_permutation_importance(model, X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        scoring="f1_macro",
        n_repeats=15,
        random_state=random_state,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def local_sensitivity(
    model, row: pd.DataFrame, baselines: dict[str, float], dropout_label: str = "Dropout"
) -> pd.DataFrame:
    classes = list(model.classes_)
    if dropout_label not in classes:
        return pd.DataFrame(columns=["feature", "risk_change"])
    idx = classes.index(dropout_label)
    base_prob = float(model.predict_proba(row)[0][idx])
    rows = []
    for feature, baseline in baselines.items():
        if feature not in row.columns:
            continue
        altered = row.copy()
        altered.loc[altered.index[0], feature] = baseline
        p = float(model.predict_proba(altered)[0][idx])
        rows.append({"feature": feature, "risk_change": base_prob - p})
    return pd.DataFrame(rows).sort_values("risk_change", ascending=False)
