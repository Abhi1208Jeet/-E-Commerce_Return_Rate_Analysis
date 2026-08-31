# E-Commerce Return Rate Reduction Analysis

An end-to-end data analytics project analyzing 5,000 e-commerce orders to identify why customers return products, build a return-risk prediction model, and design a 4-page Power BI dashboard for stakeholders.

## Objective

Identify why customers return products and how return rates vary by category, geography, and discount level. Build a model that flags high-risk orders/products, and present findings in an interactive dashboard with drill-through analysis.

## Tools

Python (pandas, scikit-learn, seaborn, matplotlib) · Power BI (DAX, drill-through) · Jupyter Notebook

## Key Findings

- **Overall return rate: 29%** across 5,000 orders (2022–2025)
- **Clothing has the highest return rate (37.4%)**, well above other categories (24–26%)
- **Discount depth strongly predicts returns**: 24.1% return rate at 0–10% discount vs. 33–35% at 31–50% discount
- **Geographic variability**: return rate ranges from ~14% to ~50% across customer locations
- **Return reasons split evenly** across Defective, Changed Mind, Wrong Item, and Size Issue — no single fix solves this
- Shipping method and payment method have **minimal impact** on return rate
- Every return generates CO2 emissions and packaging waste with **no offsetting environmental savings**

## Predictive Model

A logistic regression model predicts return probability using only pre-purchase features (category, price, discount, shipping, payment, demographics), explicitly excluding any post-purchase fields (return reason, return cost, etc.) to avoid data leakage.

| Metric | Value |
|---|---|
| ROC AUC | 0.60 |
| Top drivers | Product_Category (Clothing), Discount_Applied |

**Note:** AUC of 0.60 reflects weak-to-moderate predictive power given the available features — this is a directional risk-ranking tool for prioritizing manual review, not a high-precision classifier. Its top coefficients independently confirm the same category/discount patterns found in the descriptive analysis.

## Repository Contents

| File | Description |
|---|---|
| `E-Commerce_Return_Rate_Analysis_Completed.ipynb` | Full analysis notebook: cleaning, EDA, modeling, exports |
| `return_rate_analysis.py` | Standalone Python script version of the pipeline |
| `powerbi_dataset_scored.csv` | Order-level dataset with return risk scores (Power BI source) |
| `high_risk_products.csv` | Products flagged as high-risk based on average risk score |
| `all_products_risk_scored.csv` | All products with aggregated risk scores |
| `E-Commerce_Return_Rate_Analysis_Report.docx` | Written summary report |

## Power BI Dashboard

A 4-page interactive dashboard built on `powerbi_dataset_scored.csv`:

1. **Executive Overview** — KPI cards, monthly return-rate trend, returned vs. kept split
2. **Root Cause Analysis** — return rate by category, reason, geography, discount band, shipping, payment
3. **Risk & Prediction** — risk score distribution, price-vs-risk scatter, high-risk product watchlist with drill-through
4. **Financial & Sustainability Impact** — return cost, profit/loss comparison, CO2/waste by return status

## Recommendations

- Prioritize Clothing category for return reduction (size guides, size-recommendation tools, clearer photography)
- Reassess deep-discount promotions (31%+) — consider stricter return windows for heavily discounted items
- Investigate high-return-rate locations with logistics/fulfillment teams
- Do not prioritize shipping/payment method changes — return-rate impact is minimal
- Use the risk model as a prioritization tool for manual review, not a standalone automated decision system

## How to Run

1. Clone this repo
2. Place `returns_sustainability_dataset.csv` in the same folder as the notebook (or update `DATA_PATH` in the first cell)
3. Open `E-Commerce_Return_Rate_Analysis_Completed.ipynb` and run all cells

---

*Part of a Data Analyst portfolio project.*
