"""
AI-Driven Financial Governance - Base Project
---------------------------------------------
Pipeline:
  1. Load (or generate) transactions
  2. Rule-based policy / compliance checks
  3. ML anomaly detection (Isolation Forest)
  4. Combined risk score + governance report

Run:   python financial_governance.py
       python financial_governance.py --csv my_transactions.csv

CSV columns required: txn_id, date, department, vendor, category,
                      amount, requested_by, approved_by
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------------------
# Governance policy configuration (edit to match your organisation)
# ----------------------------------------------------------------------
APPROVAL_LIMITS = {          # max amount a single approver may sign off
    "Finance": 50_000,
    "HR": 20_000,
    "IT": 40_000,
    "Operations": 30_000,
    "Marketing": 15_000,
}
DUPLICATE_WINDOW_DAYS = 3
SPLIT_THRESHOLD_RATIO = 0.95     # amounts just under the limit look suspicious
SPLIT_WINDOW_DAYS = 2
RULE_WEIGHT, ML_WEIGHT = 0.6, 0.4
HIGH_RISK, MEDIUM_RISK = 40, 20


# ----------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------
def generate_transactions(n=1000, seed=42):
    """Create synthetic transactions with a few injected violations."""
    rng = np.random.default_rng(seed)
    depts = list(APPROVAL_LIMITS)
    people = [f"EMP{i:03d}" for i in range(1, 21)]
    vendors = [f"Vendor_{c}" for c in "ABCDEFGHIJ"]
    cats = ["Software", "Travel", "Supplies", "Consulting", "Equipment"]

    df = pd.DataFrame({
        "txn_id": [f"T{i:05d}" for i in range(n)],
        "date": pd.to_datetime("2025-01-01")
                + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        "department": rng.choice(depts, n),
        "vendor": rng.choice(vendors, n),
        "category": rng.choice(cats, n),
        "amount": np.round(rng.lognormal(8, 1, n), 2),
        "requested_by": rng.choice(people, n),
    })
    df["approved_by"] = [rng.choice([p for p in people if p != r])
                         for r in df["requested_by"]]

    # Inject violations so the system has something to find
    for i in rng.choice(n, 10, replace=False):          # self-approval
        df.loc[i, "approved_by"] = df.loc[i, "requested_by"]
    for i in rng.choice(n, 8, replace=False):           # huge amounts
        df.loc[i, "amount"] = rng.uniform(80_000, 200_000)
    for i in rng.choice(n - 1, 6, replace=False):       # duplicates
        dup = df.loc[i].copy()
        dup["txn_id"] = f"D{i:05d}"
        dup["date"] += pd.Timedelta(days=1)
        df = pd.concat([df, dup.to_frame().T], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    return df.sort_values("date").reset_index(drop=True)


def load_data(path=None):
    if path:
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        df = generate_transactions()
    return df


# ----------------------------------------------------------------------
# 2. Rule-based compliance checks
# ----------------------------------------------------------------------
def apply_rules(df):
    df = df.copy()
    df["limit"] = df["department"].map(APPROVAL_LIMITS)

    df["rule_over_limit"] = df["amount"] > df["limit"]
    df["rule_self_approval"] = df["requested_by"] == df["approved_by"]
    df["rule_weekend"] = (df["date"].dt.dayofweek >= 5) & (df["amount"] > 10_000)

    # Duplicate payments: same vendor + amount within a short window
    df = df.sort_values(["vendor", "amount", "date"])
    gap = df.groupby(["vendor", "amount"])["date"].diff().dt.days
    next_gap = df.groupby(["vendor", "amount"])["date"].diff(-1).dt.days.abs()
    df["rule_duplicate"] = (gap <= DUPLICATE_WINDOW_DAYS) | (next_gap <= DUPLICATE_WINDOW_DAYS)

    # Split transactions: several just-under-limit payments to one vendor
    near = df["amount"].between(df["limit"] * SPLIT_THRESHOLD_RATIO, df["limit"])
    df["rule_split"] = near

    df = df.sort_values("date").reset_index(drop=True)

    rule_cols = [c for c in df.columns if c.startswith("rule_")]
    weights = {"rule_over_limit": 35, "rule_self_approval": 40,
               "rule_weekend": 10, "rule_duplicate": 30, "rule_split": 20}
    df["rule_score"] = sum(df[c].astype(int) * w for c, w in weights.items())
    df["rule_score"] = df["rule_score"].clip(upper=100)
    df["violations"] = df[rule_cols].apply(
        lambda r: ", ".join(c.replace("rule_", "") for c in rule_cols if r[c]), axis=1)
    return df


# ----------------------------------------------------------------------
# 3. ML anomaly detection
# ----------------------------------------------------------------------
def detect_anomalies(df, contamination=0.03):
    feats = pd.DataFrame({
        "log_amount": np.log1p(df["amount"]),
        "dow": df["date"].dt.dayofweek,
        "dept": LabelEncoder().fit_transform(df["department"]),
        "category": LabelEncoder().fit_transform(df["category"]),
        "vendor": LabelEncoder().fit_transform(df["vendor"]),
    })
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=42)
    model.fit(feats)
    raw = -model.score_samples(feats)                  # higher = more anomalous
    df = df.copy()
    df["ml_score"] = (100 * (raw - raw.min()) / (raw.max() - raw.min())).round(1)
    df["ml_anomaly"] = model.predict(feats) == -1
    return df


# ----------------------------------------------------------------------
# 4. Risk scoring & reporting
# ----------------------------------------------------------------------
def score_risk(df):
    df = df.copy()
    df["risk_score"] = (RULE_WEIGHT * df["rule_score"]
                        + ML_WEIGHT * df["ml_score"]).round(1)
    df["risk_level"] = pd.cut(df["risk_score"],
                              bins=[-1, MEDIUM_RISK, HIGH_RISK, 101],
                              labels=["Low", "Medium", "High"])
    return df


def report(df, out_csv="flagged_transactions.csv"):
    print("=" * 60)
    print(" AI-DRIVEN FINANCIAL GOVERNANCE REPORT")
    print("=" * 60)
    print(f"Transactions analysed : {len(df):,}")
    print(f"Total value           : {df['amount'].sum():,.2f}")
    print("\nRisk distribution:")
    print(df["risk_level"].value_counts().reindex(["High", "Medium", "Low"]).to_string())

    print("\nPolicy violations found:")
    for c in [c for c in df.columns if c.startswith("rule_") and c != "rule_score"]:
        print(f"  {c.replace('rule_', ''):<15} {int(df[c].sum())}")
    print(f"  {'ml_anomaly':<15} {int(df['ml_anomaly'].sum())}")

    print("\nRisk by department (avg score):")
    print(df.groupby("department")["risk_score"].mean().round(1)
            .sort_values(ascending=False).to_string())

    flagged = df[df["risk_level"] == "High"].sort_values("risk_score", ascending=False)
    cols = ["txn_id", "date", "department", "vendor", "amount",
            "requested_by", "approved_by", "violations", "risk_score"]
    print("\nTop 10 high-risk transactions:")
    print(flagged[cols].head(10).to_string(index=False))

    flagged[cols + ["ml_score", "risk_level"]].to_csv(out_csv, index=False)
    print(f"\nAll {len(flagged)} high-risk transactions saved to {out_csv}")


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AI-Driven Financial Governance")
    parser.add_argument("--csv", help="Path to transactions CSV (optional)")
    args = parser.parse_args()

    df = load_data(args.csv)
    df = apply_rules(df)
    df = detect_anomalies(df)
    df = score_risk(df)
    report(df)


if __name__ == "__main__":
    main()
