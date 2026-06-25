from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def run_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = ["age", "tenure_months", "monthly_spend", "support_tickets", "satisfaction_score", "churn_risk"]
    X = df[feature_cols]
    X_scaled = StandardScaler().fit_transform(X)

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["segment"] = model.fit_predict(X_scaled)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="segment", palette="viridis")
    plt.title("Répartition des segments clients")
    plt.tight_layout()
    plt.savefig(output_dir / "segments.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="segment", y="churn_risk", palette="magma")
    plt.title("Score de risque par segment")
    plt.tight_layout()
    plt.savefig(output_dir / "risk_by_segment.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df.groupby("issue_type").size().reset_index(name="count"), x="issue_type", y="count", palette="Set2")
    plt.title("Répartition des types de demande")
    plt.tight_layout()
    plt.savefig(output_dir / "issue_types.png", dpi=180)
    plt.close()

    summary = df.groupby("segment").agg(
        clients=("customer_id", "count"),
        avg_risk=("churn_risk", "mean"),
        avg_spend=("monthly_spend", "mean"),
        avg_tickets=("support_tickets", "mean"),
        avg_satisfaction=("satisfaction_score", "mean"),
    ).reset_index()
    summary.to_csv(output_dir / "segment_summary.csv", index=False)

    df.to_csv(output_dir / "customer_analysis.csv", index=False)
