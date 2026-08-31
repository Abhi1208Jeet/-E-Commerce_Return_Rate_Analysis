"""
E-Commerce Return Rate Reduction Analysis
Python codebase: cleaning, EDA, logistic regression risk model, exports.
"""

# # E-Commerce Return Rate Reduction Analysis
# 
# **Objective:** Identify why customers return products and how return rates vary by category, geography, and customer segment; build a return-risk prediction model and export a Power BI-ready dataset with drill-through fields.
# 
# **Tools:** Python (pandas, seaborn, scikit-learn), SQL-style aggregation logic, Power BI
# 
# **Note on scope:** The dataset does not include a `Supplier` or `Marketing_Channel` field. Category (`Product_Category`) is used as the closest proxy for supplier-level analysis, and `User_Location` (100 city codes) is used as the geography dimension.
# 
# **Deliverables produced by this notebook:**
# - Cleaned dataset with engineered features
# - Return-rate breakdowns by category, geography, shipping, payment, age, discount, and time
# - Logistic regression model predicting return probability
# - `high_risk_products.csv` — high-risk product list
# - `powerbi_dataset_scored.csv` — full dataset with risk scores for the Power BI dashboard
# 

# ## Step 1: Load the Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

DATA_PATH = "returns_sustainability_dataset.csv"  # place this file next to the notebook, or update the path
df = pd.read_csv(DATA_PATH)
df.head()

# ## Step 2: Understand the Data

df.shape

df.info()

df.describe()

for col in ['Return_Status','Return_Reason','Product_Category','Shipping_Method','Payment_Method']:
    print(f"--- {col} ---")
    print(df[col].value_counts(), "\n")

# ## Step 3: Data Quality Check

print("Nulls per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate Order_IDs:", df['Order_ID'].duplicated().sum())

# **Result:** No missing values, no duplicate rows, no duplicate order IDs — the dataset is already clean at the row level.

# ## Step 4: Convert Dates & Engineer Features

df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Return_Flag'] = np.where(df['Return_Status'] == 'Returned', 1, 0)
df['Year'] = df['Order_Date'].dt.year
df['Month'] = df['Order_Date'].dt.month
df['Month_Name'] = df['Order_Date'].dt.month_name()
df['Year_Month'] = df['Order_Date'].dt.to_period('M').astype(str)

bins_age = [0, 18, 25, 35, 45, 55, 65, 100]
labels_age = ["Under 18","18-25","26-35","36-45","46-55","56-65","65+"]
df['Age_Group'] = pd.cut(df['User_Age'], bins=bins_age, labels=labels_age)

df['Discount_Group'] = pd.cut(df['Discount_Applied'], bins=[-1,10,20,30,40,50],
    labels=["0-10%","11-20%","21-30%","31-40%","41-50%"])

df.head()

# ## Step 5: Overall Return Rate

total_orders = df['Order_ID'].nunique()
returned_orders = df.loc[df['Return_Status']=='Returned','Order_ID'].nunique()
return_rate = returned_orders/total_orders*100

print("Total Orders:", total_orders)
print("Returned Orders:", returned_orders)
print("Return Rate:", round(return_rate,2), "%")

# **~29% of all orders are returned** — a high rate that justifies a dedicated reduction initiative.

# ## Step 6: Return Reasons

reasons = df[df['Return_Status']=='Returned']['Return_Reason'].value_counts()
print(reasons)

plt.figure(figsize=(8,5))
sns.countplot(data=df[df['Return_Status']=='Returned'], y='Return_Reason', order=reasons.index, color='#4C72B0')
plt.title('Return Reasons'); plt.xlabel('Count'); plt.ylabel('')
plt.tight_layout(); plt.show()

# **Defective items (26%)** and **changed mind (26%)** are the top two reasons, followed closely by wrong item and size issues — a near-even split suggests both quality-control and product-description/sizing issues need attention.

# ## Step 7: Return Rate by Product Category

category_analysis = df.groupby('Product_Category').agg(
    Total_Orders=('Order_ID','nunique'),
    Returned_Orders=('Return_Flag','sum'),
    Revenue=('Order_Value','sum'),
    Return_Cost=('Return_Cost','sum')
).reset_index()
category_analysis['Return_Rate'] = category_analysis['Returned_Orders']/category_analysis['Total_Orders']*100
category_analysis = category_analysis.sort_values('Return_Rate', ascending=False)
category_analysis

plt.figure(figsize=(8,5))
sns.barplot(data=category_analysis, x='Return_Rate', y='Product_Category', color='#DD8452')
plt.title('Return Rate by Product Category'); plt.xlabel('Return Rate (%)'); plt.ylabel('')
plt.tight_layout(); plt.show()

# **Clothing has the highest return rate at 37.4%** — well above the 25% overall average of the other four categories — driven likely by sizing/fit issues. This is the single highest-leverage category to target first.

# ## Step 8: Return Rate by Geography (User Location)

geo = df.groupby('User_Location').agg(Total_Orders=('Order_ID','nunique'), Returned=('Return_Flag','sum')).reset_index()
geo = geo[geo['Total_Orders'] >= 20]   # filter for statistical reliability
geo['Return_Rate'] = geo['Returned']/geo['Total_Orders']*100

print("Highest return-rate locations:")
print(geo.sort_values('Return_Rate', ascending=False).head(10))
print("\nLowest return-rate locations:")
print(geo.sort_values('Return_Rate', ascending=True).head(10))

# Return rate varies from **~14% to ~50%** across locations with sufficient order volume — a 3.5x spread. This points to regional factors (courier quality, delivery damage, local product-fit expectations) worth investigating with the logistics/regional teams.

# ## Step 9: Return Rate by Shipping Method

shipping_analysis = df.groupby('Shipping_Method').agg(Total_Orders=('Order_ID','nunique'), Returned=('Return_Flag','sum')).reset_index()
shipping_analysis['Return_Rate'] = shipping_analysis['Returned']/shipping_analysis['Total_Orders']*100
shipping_analysis

plt.figure(figsize=(6,4))
sns.barplot(data=shipping_analysis, x='Shipping_Method', y='Return_Rate', color='#55A868')
plt.title('Return Rate by Shipping Method'); plt.ylabel('Return Rate (%)')
plt.tight_layout(); plt.show()

# Shipping method shows **almost no difference** (28.5%–29.4%) — this is not a meaningful lever for return reduction.

# ## Step 10: Return Rate by Payment Method

payment_analysis = df.groupby('Payment_Method').agg(Total_Orders=('Order_ID','nunique'), Returned=('Return_Flag','sum')).reset_index()
payment_analysis['Return_Rate'] = payment_analysis['Returned']/payment_analysis['Total_Orders']*100
payment_analysis

# Credit Card (31.0%) and COD (30.5%) run slightly higher than Debit Card/Wallet (~27%) — a modest but consistent gap, possibly tied to lower purchase-commitment for non-prepaid or 'buy now, decide later' payment types.

# ## Step 11: Customer Age Analysis

age_analysis = df.groupby('Age_Group', observed=True).agg(Total_Orders=('Order_ID','nunique'), Returned=('Return_Flag','sum')).reset_index()
age_analysis['Return_Rate'] = age_analysis['Returned']/age_analysis['Total_Orders']*100
age_analysis

# Return rate is fairly flat across adult age bands (27–30%); Under-18 is lower (22.6%) but represents a small sample (106 orders).

# ## Step 12: Discount vs Return Rate

discount_analysis = df.groupby('Discount_Group', observed=True).agg(Total_Orders=('Order_ID','nunique'), Returned=('Return_Flag','sum')).reset_index()
discount_analysis['Return_Rate'] = discount_analysis['Returned']/discount_analysis['Total_Orders']*100
discount_analysis

plt.figure(figsize=(7,4))
sns.barplot(data=discount_analysis, x='Discount_Group', y='Return_Rate', color='#C44E52')
plt.title('Return Rate by Discount Band'); plt.ylabel('Return Rate (%)')
plt.tight_layout(); plt.show()

# **Clear upward trend:** return rate climbs from 24% at 0–10% discount to 33–35% at 31–50% discount. Deep discounting appears to attract lower-commitment purchases — a strong, actionable signal for pricing/promo strategy.

# ## Step 13: Monthly Return Rate Trend

trend = df.groupby('Year_Month').agg(Total_Orders=('Order_ID','nunique'), Returned=('Return_Flag','sum')).reset_index()
trend['Return_Rate'] = trend['Returned']/trend['Total_Orders']*100

plt.figure(figsize=(12,5))
plt.plot(trend['Year_Month'], trend['Return_Rate'], marker='o', color='#4C72B0')
plt.xticks(rotation=90, fontsize=7)
plt.title('Monthly Return Rate Trend (2022-2025)'); plt.ylabel('Return Rate (%)')
plt.tight_layout(); plt.show()

# ## Step 14: Return Cost & Profit/Loss Impact

returned_df = df[df['Return_Status']=='Returned']
total_return_cost = returned_df['Return_Cost'].sum()
average_return_cost = returned_df['Return_Cost'].mean()
print("Total Return Cost:", total_return_cost)
print("Average Return Cost per Return:", round(average_return_cost,2))

pl = df.groupby('Return_Status')['Profit_Loss'].agg(['sum','mean'])
pl

plt.figure(figsize=(6,5))
sns.boxplot(data=df, x='Return_Status', y='Profit_Loss', hue='Return_Status', palette=['#55A868','#C44E52'], legend=False)
plt.title('Profit/Loss by Return Status')
plt.tight_layout(); plt.show()

# Returns cost the business **₹2.9 lakh (₹290,000) total** at an average of ₹200/return, and returned orders average ~11% lower profit per order than kept orders.

# ## Step 15: Sustainability Impact

sustainability = df.groupby('Return_Status').agg(
    CO2_Emissions=('CO2_Emissions','sum'),
    Packaging_Waste=('Packaging_Waste','sum'),
    CO2_Saved=('CO2_Saved','sum'),
    Waste_Avoided=('Waste_Avoided','sum')
).reset_index()
sustainability

# Returned orders generate additional CO2 emissions and packaging waste from reverse logistics with **zero offsetting savings** — every return is a pure environmental cost on top of its financial cost, reinforcing the case for return prevention over return handling.

# ## Step 16: Logistic Regression — Predicting Return Probability
# 
# We build a classification model to estimate each order's probability of being returned, using only features known **at the time of purchase** (excluding return reason, days-to-return, return cost, profit/loss, and sustainability fields, which are only known *after* a return happens and would leak the outcome).

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report)

cat_features = ['Product_Category','Shipping_Method','Payment_Method','User_Gender','Age_Group']
num_features = ['Product_Price','Order_Quantity','Discount_Applied','User_Age','Order_Value']

X = pd.get_dummies(df[cat_features + num_features], columns=cat_features, drop_first=True)
y = df['Return_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:,1]

print("Accuracy:", round(accuracy_score(y_test,y_pred),3))
print("Precision:", round(precision_score(y_test,y_pred),3))
print("Recall:", round(recall_score(y_test,y_pred),3))
print("F1 Score:", round(f1_score(y_test,y_pred),3))
print("ROC AUC:", round(roc_auc_score(y_test,y_proba),3))
print("\n", classification_report(y_test,y_pred))

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test,y_proba):.3f}', color='#4C72B0')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Return Prediction Model')
plt.legend(); plt.tight_layout(); plt.show()

# **Model performance note:** ROC AUC of ~0.60 indicates the pre-purchase fields in this dataset (category, price, discount, shipping, payment, demographics) carry only a **weak-to-moderate** signal for predicting returns — real-world return prediction usually needs richer features (return history per customer, product review/rating data, size-chart deviation, image-vs-received mismatches). Treat this model as a *first-pass risk-ranking tool*, not a high-precision classifier — it's still useful for prioritizing which products/orders to review, just don't oversell its accuracy in the portfolio writeup.

coefs = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
coefs['Abs_Coef'] = coefs['Coefficient'].abs()
coefs = coefs.sort_values('Abs_Coef', ascending=False)

plt.figure(figsize=(8,6))
top15 = coefs.head(15).sort_values('Coefficient')
colors = ['#C44E52' if c<0 else '#55A868' for c in top15['Coefficient']]
plt.barh(top15['Feature'], top15['Coefficient'], color=colors)
plt.title('Logistic Regression Coefficients (Top 15 Drivers)')
plt.xlabel('Coefficient (standardized)')
plt.tight_layout(); plt.show()

coefs.head(10)[['Feature','Coefficient']]

# The strongest positive drivers of return probability are **Clothing category** and **Discount_Applied** — consistent with the EDA findings above. This cross-validation between the descriptive stats and the model coefficients is a good sign the signal is real, even if modest.

# ## Step 17: Score All Orders & Products for Risk

X_full_s = scaler.transform(X)
df['Return_Risk_Score'] = model.predict_proba(X_full_s)[:,1]

def risk_band(p):
    if p >= 0.7: return 'High'
    elif p >= 0.4: return 'Medium'
    else: return 'Low'

df['Risk_Band'] = df['Return_Risk_Score'].apply(risk_band)
df['Risk_Band'].value_counts()

product_risk = df.groupby('Product_ID').agg(
    Product_Category=('Product_Category','first'),
    Orders=('Order_ID','nunique'),
    Actual_Returns=('Return_Flag','sum'),
    Avg_Risk_Score=('Return_Risk_Score','mean')
).reset_index()
product_risk['Actual_Return_Rate_%'] = (product_risk['Actual_Returns']/product_risk['Orders']*100).round(2)
product_risk['Avg_Risk_Score'] = product_risk['Avg_Risk_Score'].round(3)
product_risk = product_risk.sort_values('Avg_Risk_Score', ascending=False)

high_risk_products = product_risk[product_risk['Avg_Risk_Score'] >= 0.5]
print(f"High-risk products (avg score >= 0.5): {len(high_risk_products)} of {len(product_risk)}")
high_risk_products.head(15)

# ## Step 18: Export Deliverables

product_risk.to_csv('all_products_risk_scored.csv', index=False)
high_risk_products.to_csv('high_risk_products.csv', index=False)

powerbi_cols = ['Order_ID','Product_ID','User_ID','Order_Date','Product_Category','Product_Price',
    'Order_Quantity','Discount_Applied','Shipping_Method','Payment_Method','User_Age','User_Gender',
    'User_Location','Age_Group','Discount_Group','Return_Status','Return_Reason','Days_to_Return',
    'Order_Value','Return_Cost','Profit_Loss','CO2_Emissions','Packaging_Waste','CO2_Saved',
    'Waste_Avoided','Year','Month','Month_Name','Year_Month','Return_Flag','Return_Risk_Score','Risk_Band']
df[powerbi_cols].to_csv('powerbi_dataset_scored.csv', index=False)

print("Saved: all_products_risk_scored.csv, high_risk_products.csv, powerbi_dataset_scored.csv")

# ## Conclusion & Key Insights
# 
# 1. **Overall return rate is 29%** — high enough to represent a material cost and sustainability drag.
# 2. **Clothing drives the most returns (37.4%)**, well above other categories — start reduction efforts here (better size guides, size-recommendation tools, clearer product photos).
# 3. **Discounting is strongly linked to returns** — orders with 31–50% discount return at 33–35% vs. 24% at 0–10% discount. Reassess deep-discount promotions or add stricter return windows for heavily discounted items.
# 4. **Geography matters a lot** (14%–50% return rate range across cities) — worth a regional deep-dive with logistics/fulfillment teams for the worst-performing locations.
# 5. **Return reasons split almost evenly** between Defective, Changed Mind, Wrong Item, and Size Issue — meaning both quality control *and* customer-expectation-setting (photos, sizing, descriptions) need attention; no single fix will move the needle alone.
# 6. **Shipping method and payment method are weak levers** — small return-rate differences, not worth prioritizing.
# 7. **Every return is a pure sustainability cost** — no CO2/waste is offset when an order is returned, strengthening the business case for prevention over handling.
# 8. **The logistic regression model (AUC ≈ 0.60) is a directional risk-ranking tool**, not a precise classifier — useful to flag ~40% of products as elevated-risk for manual review, but its main value here is confirming the category/discount findings independently.
# 
# ### Recommended next steps for the Power BI dashboard
# - **Page 1 – Executive Overview:** overall return rate, cost, trend line, KPI cards
# - **Page 2 – Root Cause Analysis:** return reasons, category and geography breakdowns
# - **Page 3 – Risk & Prediction:** risk score distribution, high-risk product table with drill-through to order level
# - **Page 4 – Financial & Sustainability Impact:** return cost, profit/loss, CO2/waste comparison
# 
