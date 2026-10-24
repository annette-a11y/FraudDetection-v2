# Problem Statement: Fraud Detection Dashboard (Version 2)

## Background

Online retailers that give customers an initial subsidy (store credit) are a target
for fraud. Fraudsters create duplicate accounts, share devices across accounts,
use invalid cards, retry failed payments and make many purchases of the same
product in a short time to drain subsidies and resell goods. Every fraudulent
transaction that gets through costs the business the goods, the subsidy and
chargeback fees.

Fraud analysts cannot manually check every transaction. Today they have no
consistent way to tell which transactions are suspicious, why they are
suspicious, or what to do about them, so reviews are slow and decisions vary
between analysts.

## Problem

**How can we automatically identify suspicious transactions, explain why each one
was flagged, and apply a consistent action, so that analysts can focus on the
transactions most likely to be fraud?**

## Objective

Build a fraud detection system that:

1. Checks every transaction against 10 known fraud warning signs.
2. Combines them into a single **risk score from 0 to 100** using weights from 1
   (least serious) to 10 (most serious).
3. Assigns a **severity** (Low, Medium, High, Critical) and an **action** (approve,
   OTP verification, manual review, block and freeze).
4. Shows the results on a **one-page dashboard** that analysts can filter and search.
5. Measures how accurate the rules are against known outcomes, using a
   machine-learning model as a benchmark.

## Intended users

| User | Need |
|---|---|
| Fraud analysts | A prioritised list of suspicious transactions with the reasons for each flag |
| Fraud / risk managers | An overview of fraud levels and which warning signs are most common |
| Product / engineering teams | A transparent, adjustable rule set they can tune |

## Inputs

One record per transaction: customer name and ID, initial subsidy, transaction
amount, hour of day, number of items, repeated product purchases, previous
transactions, first-transaction flag, duplicate account detection, transaction
frequency per hour, card validity, failed login attempts, payment retry count,
same device used by multiple accounts, and login location change.

## Outputs

| Output | Description |
|---|---|
| Risk score | 0–100, the weighted total of triggered warning signs |
| Severity | Low (<20), Medium (20–39), High (40–64), Critical (65+). An invalid card is always at least High |
| Action taken | Approved / Step-up verification (OTP) / Held for manual review / Blocked & account frozen |
| Fraud decision | Fraud if High or Critical, otherwise legitimate |
| Indicators triggered | Which warning signs fired for each transaction |
| Dashboard | Total transactions, fraud detected, legitimate transactions, average risk score, risk threat distribution, and a searchable transaction table |

## Approach

- **Rule engine (makes the decisions):** 10 weighted risk flags. Rules are used
  because every decision can be explained to the analyst and the customer.
- **Machine-learning benchmark (checks the rules):** a Random Forest classifier is
  trained on 70% of transactions with known outcomes and compared with the rules
  on the remaining 30%. If the model clearly outperforms the rules, the weights
  or thresholds should be reviewed.

## Success criteria

- Every transaction receives a score, severity and action.
- Precision and recall of the rule engine are reported on held-out transactions.
- The rules perform within a reasonable margin of the machine-learning benchmark.
  Current result on the sample data: rules 77% precision / 73% recall, against
  86% / 81% for the benchmark.
- The dashboard loads as a single page and runs anywhere through Docker.

## Scope and limitations

- **Sample data is synthetic.** The 1,000 transactions are simulated, so the results
  demonstrate the method, not real-world accuracy.
- **Weights are set by hand**, not learned. They reflect the agreed priority of
  each warning sign and must be reviewed as fraud patterns change.
- **Batch, not real time.** The dashboard is a snapshot produced each time the
  script runs. It does not score transactions live.
- **Fraud yes/no only.** It does not estimate how much money each transaction
  could lose.
- **Human review remains essential.** Signals such as "location changed" or "first
  transaction" can also describe genuine travellers or new customers, so blocking
  decisions should be checked by an analyst.
