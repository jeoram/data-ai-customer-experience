from __future__ import annotations

import numpy as np
import pandas as pd


def generate_customer_data(path: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_rows = 1500

    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(1, n_rows + 1)],
        "age": rng.integers(18, 70, size=n_rows),
        "tenure_months": rng.integers(1, 72, size=n_rows),
        "monthly_spend": np.clip(rng.normal(90, 35, size=n_rows), 20, 300).round(2),
        "support_tickets": rng.poisson(2.2, size=n_rows),
        "satisfaction_score": rng.integers(1, 6, size=n_rows),
        "channel": rng.choice(["web", "store", "call", "app"], size=n_rows, p=[0.4, 0.2, 0.2, 0.2]),
        "issue_type": rng.choice(["billing", "technical", "product", "account"], size=n_rows, p=[0.3, 0.25, 0.25, 0.2]),
        "resolved": rng.binomial(1, 0.8, size=n_rows),
        "churn_risk": rng.uniform(0, 1, size=n_rows).round(3),
    })

    df["priority"] = np.where(df["support_tickets"] >= 3, "high", "medium")
    df["priority"] = np.where((df["satisfaction_score"] <= 2) | (df["churn_risk"] > 0.8), "critical", df["priority"])

    df.to_csv(path, index=False)
    return df
