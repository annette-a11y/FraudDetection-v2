"""
Fraud Detection Model
=====================

Two layers:
  1. Rule engine  - turns raw transaction features into risk flags, scores them
                    with RISK_WEIGHTS, assigns a severity and an action.
  2. ML model     - a RandomForest trained on the same features, used to check
                    the rule engine and to give a fraud probability per row.

Usage:
  python fraud_model.py                     # synthetic demo data (1000 rows)
  python fraud_model.py --rows 2000
  python fraud_model.py --input my_data.csv # score your own CSV (same columns)

Outputs:
  transactions_scored.csv   every transaction with flags, score, severity, action
  model_report.json         rule-engine + ML metrics
  dashboard.html            one-page dashboard (open in any browser)
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

RISK_WEIGHTS = {
    "flag_repeated_purchase": 10,
    "flag_card_invalid":       9,  # Priority Escalation Weight
    "flag_high_frequency":     8,
    "flag_duplicate_account":  7,
    "flag_same_device":        6,
    "flag_location_changed":   5,
    "flag_same_product_high":  4,
    "flag_payment_retry":      3,
    "flag_failed_login":       2,
    "flag_id_not_verified":    1,
}
MAX_SCORE = sum(RISK_WEIGHTS.values())  # 55

FLAG_LABELS = {
    "flag_repeated_purchase": "Repeated product purchase (>10)",
    "flag_card_invalid":      "Invalid card",
    "flag_high_frequency":    "High frequency (>3/hr)",
    "flag_duplicate_account": "Duplicate account",
    "flag_same_device":       "Same device, multiple accounts",
    "flag_location_changed":  "Login location changed",
    "flag_same_product_high": "High item quantity",
    "flag_payment_retry":     "Payment retries",
    "flag_failed_login":      "Failed logins",
    "flag_id_not_verified":   "ID not verified",
}

# Thresholds for the count-type features
THRESHOLDS = {
    "repeated_product_purchase": 10,   # > 10
    "transaction_frequency":      3,   # > 3 per hour
    "num_items":                 15,   # >= 15 items in one basket
    "payment_retry_count":        3,   # >= 3 retries
    "failed_login_attempts":      3,   # >= 3 failures
}

# Severity bands on the 0-100 normalised score
SEVERITY_BANDS = [  # (min_score, severity, action)
    (65, "Critical", "Blocked & account frozen"),
    (40, "High",     "Held for manual review"),
    (20, "Medium",   "Step-up verification (OTP)"),
    (0,  "Low",      "Approved"),
]
FRAUD_THRESHOLD = 40  # score >= 40 (High or Critical) counts as fraud

FEATURE_COLUMNS = [
    "Initial_Subsidy", "transaction_amount", "hour_of_day", "num_items",
    "repeated_product_purchase", "prev_transactions", "is_first_transaction",
    "Duplicate_account_detection", "transaction_frequency", "valid_card",
    "failed_login_attempts", "payment_retry_count",
    "same_device_multiple_accounts", "login_location_changed",
]

# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------

FIRST = ["James", "Mary", "John", "Linda", "Mike", "Sarah", "David", "Emily", "Chris",
         "Jessica", "Matt", "Ashley", "Josh", "Amanda", "Ryan", "Megan", "Tom", "Laura",
         "Kevin", "Rachel", "Brian", "Nicole", "Jake", "Amy", "Tyler", "Katie", "Eric",
         "Hannah", "Mark", "Lisa", "Steve", "Anna", "Dan", "Kelly", "Jason", "Emma"]
LAST = ["Smith", "Johnson", "Brown", "Jones", "Miller", "Davis", "Wilson", "Moore",
        "Taylor", "Clark", "Lewis", "Walker", "Hall", "Allen", "Young", "King", "Wright",
        "Scott", "Green", "Baker", "Adams", "Nelson", "Hill", "Carter", "Parker", "Evans"]


def generate_dataset(n=500, fraud_rate=0.18, seed=42):
    """Build a realistic-looking transaction table with a hidden ground truth."""
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < fraud_rate

    def pick(p_fraud, p_legit):
        return np.where(is_fraud, rng.random(n) < p_fraud, rng.random(n) < p_legit).astype(int)

    prev = np.where(is_fraud, rng.poisson(3, n), rng.poisson(8, n))
    first = (prev == 0).astype(int)
    subsidy = rng.choice([5000, 10000, 15000, 20000, 25000], n)
    amount = np.where(is_fraud,
                      subsidy * rng.uniform(0.3, 1.5, n),
                      subsidy * rng.uniform(0.05, 0.9, n)).round(2)
    # fraud skews to late night but overlaps normal hours; some legit is late too
    hour = np.where(is_fraud & (rng.random(n) < 0.55),
                    rng.choice([0, 1, 2, 3, 4, 23], n),
                    rng.integers(6, 24, n))

    df = pd.DataFrame({
        "customer_id": [f"CUST-{10000 + i}" for i in range(n)],
        "Name": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)],
        "Initial_Subsidy": subsidy,
        "transaction_amount": amount,
        "hour_of_day": hour,
        "num_items": np.where(is_fraud, rng.integers(2, 35, n), rng.integers(1, 18, n)),
        "repeated_product_purchase": np.where(is_fraud, rng.integers(2, 22, n), rng.integers(0, 13, n)),
        "prev_transactions": prev,
        "is_first_transaction": first,
        "Duplicate_account_detection": pick(0.45, 0.03),
        "transaction_frequency": np.where(is_fraud, rng.integers(1, 9, n), rng.integers(0, 5, n)),
        "valid_card": 1 - pick(0.40, 0.02),
        "failed_login_attempts": np.where(is_fraud, rng.integers(0, 7, n), rng.choice([0, 0, 0, 1, 2, 3], n)),
        "payment_retry_count": np.where(is_fraud, rng.integers(0, 6, n), rng.choice([0, 0, 0, 1, 1, 3], n)),
        "same_device_multiple_accounts": pick(0.50, 0.04),
        "login_location_changed": pick(0.55, 0.08),
        # ~6% label noise: stealth fraud that looks normal, and false chargebacks
        "is_fraud_actual": (is_fraud ^ (rng.random(n) < 0.06)).astype(int),
    })

    # timestamps across the last 7 days, matching hour_of_day
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    days = rng.integers(0, 7, n)
    minutes = rng.integers(0, 60, n)
    df["timestamp"] = [
        (now - timedelta(days=int(d))).replace(hour=int(h), minute=int(m))
        for d, h, m in zip(days, hour, minutes)
    ]
    return df.sort_values("timestamp", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Rule engine
# --------------------------------------------------------------------------

def compute_flags(df):
    t = THRESHOLDS
    f = pd.DataFrame(index=df.index)
    f["flag_repeated_purchase"] = df["repeated_product_purchase"] > t["repeated_product_purchase"]
    f["flag_card_invalid"]      = df["valid_card"] == 0
    f["flag_high_frequency"]    = df["transaction_frequency"] > t["transaction_frequency"]
    f["flag_duplicate_account"] = df["Duplicate_account_detection"] == 1
    f["flag_same_device"]       = df["same_device_multiple_accounts"] == 1
    f["flag_location_changed"]  = df["login_location_changed"] == 1
    f["flag_same_product_high"] = df["num_items"] >= t["num_items"]
    f["flag_payment_retry"]     = df["payment_retry_count"] >= t["payment_retry_count"]
    f["flag_failed_login"]      = df["failed_login_attempts"] >= t["failed_login_attempts"]
    # First-time buyer with no history to verify identity against
    f["flag_id_not_verified"]   = (df["is_first_transaction"] == 1) & (df["prev_transactions"] == 0)
    return f.astype(int)


def classify(score, card_invalid):
    for min_score, severity, action in SEVERITY_BANDS:
        if score >= min_score:
            break
    # Priority escalation: an invalid card never sits below High
    if card_invalid and severity in ("Low", "Medium"):
        severity, action = "High", "Held for manual review (invalid card)"
    return severity, action


def score_transactions(df):
    flags = compute_flags(df)
    raw = sum(flags[k] * w for k, w in RISK_WEIGHTS.items())
    out = pd.concat([df, flags], axis=1)
    out["risk_score"] = (raw / MAX_SCORE * 100).round(1)
    sev_act = [classify(s, c) for s, c in zip(out["risk_score"], out["flag_card_invalid"])]
    out["severity"] = [s for s, _ in sev_act]
    out["action_taken"] = [a for _, a in sev_act]
    out["is_fraud_predicted"] = out["severity"].isin(["High", "Critical"]).astype(int)
    out["indicators_triggered"] = flags.apply(
        lambda r: [FLAG_LABELS[k] for k in RISK_WEIGHTS if r[k]], axis=1)
    return out


# --------------------------------------------------------------------------
# ML model
# --------------------------------------------------------------------------

def metrics(y_true, y_pred):
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 3),
    }


def train_ml(scored):
    """Train a RandomForest when ground truth labels exist."""
    X = scored[FEATURE_COLUMNS + list(RISK_WEIGHTS)]
    y = scored["is_fraud_actual"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=7,
                                   class_weight="balanced")
    model.fit(X_tr, y_tr)
    scored["ml_fraud_probability"] = model.predict_proba(X)[:, 1].round(3)
    importance = sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1])
    return {
        "ml_test_metrics": metrics(y_te, model.predict(X_te)),
        "rule_engine_test_metrics": metrics(y_te, scored.loc[X_te.index, "is_fraud_predicted"]),
        "top_features": [{"feature": f, "importance": round(i, 3)} for f, i in importance[:8]],
    }


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------

def build_dashboard(scored, report, path):
    rows = [{
        "id": r.customer_id,
        "name": r.Name,
        "indicators": r.indicators_triggered,
        "score": float(r.risk_score),
        "severity": r.severity,
        "time": pd.Timestamp(r.timestamp).strftime("%Y-%m-%d %H:%M"),
        "action": r.action_taken,
        "amount": float(r.transaction_amount),
    } for r in scored.itertuples()]
    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "fraudThreshold": FRAUD_THRESHOLD,
        "rows": rows,
        "report": report,
    }
    template = (HERE / "dashboard_template.html").read_text()
    html = template.replace("/*__DATA__*/null", json.dumps(payload))
    path.write_text(html)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="CSV with the feature columns (optional)")
    ap.add_argument("--rows", type=int, default=1000, help="synthetic rows if no --input")
    args = ap.parse_args()

    if args.input:
        df = pd.read_csv(args.input)
        if "Name" not in df:
            df["Name"] = df["customer_id"]
        if "customer_id" not in df:
            df["customer_id"] = df["Name"]
        df["timestamp"] = pd.to_datetime(df["timestamp"]) if "timestamp" in df else datetime.now()
    else:
        df = generate_dataset(args.rows)

    scored = score_transactions(df)
    report = {
        "total": len(scored),
        "fraud_detected": int(scored["is_fraud_predicted"].sum()),
        "avg_risk_score": round(float(scored["risk_score"].mean()), 1),
        "severity_counts": scored["severity"].value_counts().to_dict(),
    }
    if "is_fraud_actual" in scored:
        report.update(train_ml(scored))

    out_csv = scored.copy()
    out_csv["indicators_triggered"] = out_csv["indicators_triggered"].str.join("; ")
    out_csv.to_csv(HERE / "transactions_scored.csv", index=False)
    (HERE / "model_report.json").write_text(json.dumps(report, indent=2))
    build_dashboard(scored, report, HERE / "dashboard.html")

    print(json.dumps(report, indent=2))
    print("\nWrote transactions_scored.csv, model_report.json, dashboard.html")


if __name__ == "__main__":
    main()
