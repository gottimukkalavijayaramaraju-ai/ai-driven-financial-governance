# AI-Driven Financial Governance

Rule-based compliance checks + Isolation Forest anomaly detection + combined risk scoring for financial transactions.

## Setup
    pip install -r requirements.txt

## Run
    python financial_governance.py                  # uses generated sample data
    python financial_governance.py --csv data.csv   # uses your own data

CSV columns: txn_id, date, department, vendor, category, amount, requested_by, approved_by

Output: console report + flagged_transactions.csv
