# Fraud Detection Model

Scores transactions for fraud risk using weighted risk flags, assigns a severity
and an action, and builds a one-page dashboard (`dashboard.html`).

## Just want to see the dashboard?

Double-click `dashboard.html`. It opens in any browser and needs no installs.

## Run the model yourself

1. Install Python 3.10 or newer from https://www.python.org/downloads/
2. Open a terminal in this folder and install the libraries:

   ```
   pip3 install -r requirements.txt
   ```

3. Score the sample data (or your own CSV):

   ```
   python3 fraud_model.py --input transactions.csv
   ```

4. Open the updated dashboard:

   ```
   open dashboard.html        # macOS
   start dashboard.html       # Windows
   ```

To generate fresh made-up data instead: `python3 fraud_model.py --rows 1000`

## Input columns

Your CSV needs these columns (names must match):

| Column | Meaning |
|---|---|
| customer_id, Name | Customer identity (at least one is required) |
| Initial_Subsidy | Starting subsidy / credit amount |
| transaction_amount | Amount of this transaction |
| hour_of_day | 0-23 |
| num_items | Items in the basket |
| repeated_product_purchase | Times the same product was bought |
| prev_transactions | Customer's earlier transactions |
| is_first_transaction | 1 = yes, 0 = no |
| Duplicate_account_detection | 1 = duplicate account found |
| transaction_frequency | Transactions in the last hour |
| valid_card | 1 = valid, 0 = invalid |
| failed_login_attempts | Count |
| payment_retry_count | Count |
| same_device_multiple_accounts | 1 = yes |
| login_location_changed | 1 = yes |
| timestamp | Optional, e.g. 2026-10-23 14:05 |
| is_fraud_actual | Optional, 1 = confirmed fraud. Enables the accuracy check |

## How scoring works

Each flag has a weight from 1 (lowest risk) to 10 (highest risk). Each triggered flag adds its
weight; the total (max 55) is scaled to 0-100.

| Flag | Rule | Weight |
|---|---|---|
| Repeated product purchase | repeated_product_purchase > 10 | 10 |
| Invalid card | valid_card = 0 | 9 |
| High frequency | transaction_frequency > 3 | 8 |
| Duplicate account | Duplicate_account_detection = 1 | 7 |
| Same device | same_device_multiple_accounts = 1 | 6 |
| Location changed | login_location_changed = 1 | 5 |
| High item quantity | num_items >= 15 | 4 |
| Payment retries | payment_retry_count >= 3 | 3 |
| Failed logins | failed_login_attempts >= 3 | 2 |
| ID not verified | first transaction, no history | 1 |

| Score | Severity | Action |
|---|---|---|
| 65+ | Critical | Blocked & account frozen |
| 40-64 | High | Held for manual review |
| 20-39 | Medium | Step-up verification (OTP) |
| < 20 | Low | Approved |

An invalid card always raises severity to at least High. High and Critical count as fraud.
Weights and thresholds are at the top of `fraud_model.py`.

## Outputs

- `dashboard.html` - the dashboard
- `transactions_scored.csv` - every transaction with flags, score, severity and action
- `model_report.json` - summary counts and accuracy metrics

Note: the included `transactions.csv` is made-up sample data, not real customers.

## Run with Docker

```
docker build -t fraud-dashboard-v2 .
docker run --rm -p 8080:8080 fraud-dashboard-v2
```

Then open http://localhost:8080/dashboard.html. On start the container re-scores
`transactions.csv` and rebuilds the dashboard.

Or pull the published image (built automatically by GitHub Actions on every push to `main`):

```
docker run --rm -p 8080:8080 ghcr.io/annette-a11y/fraud-dashboard-v2:latest
```
