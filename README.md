# Analysis of Olist and Model for Predicting Seller Churn

Contributors: Dyah Almira | Lukman Fathoni | Axel Alexander

<img width="897" height="595" alt="image" src="https://github.com/user-attachments/assets/56f2424c-5076-4acc-bc95-83b7e10da0a7" />

Tableau link : 

<img width="1903" height="881" alt="image" src="https://github.com/user-attachments/assets/eca7c9f7-28e2-4ef0-970b-515d04f92251" />

Streamlit link : [Seller Churn Prediction Application](https://finpro-app-launch-olist-mdlfyncwpwevkdzlfe4rgu.streamlit.app/)

# 1. Business Problem

# 2. Analysis

## 2.1 Buyer Analysis
### 📋 Overview

This component analyzes **seller churn risk from the buyer perspective** using transaction and behavioral data from the Olist marketplace.

### Core Hypothesis

Since direct seller churn labels are unavailable, buyer experience serves as an **early warning proxy** to evaluate whether sellers are losing competitiveness on the platform:

```
Buyer Dissatisfaction → Declining Demand → Weakening Seller Performance → Potential Seller Churn
```

---
### 🎯 Objectives

- Identify patterns in buyer behavior that may signal seller performance degradation
- Analyze buyer satisfaction metrics as indirect indicators of seller health
- Examine delivery performance and its impact on buyer retention
- Assess repeat purchase behavior as a proxy for seller competitiveness
- Evaluate transaction value patterns across different seller segments

---

### 🧠 Analytical Framework

Four buyer-based signals were investigated:

1. **Buyer Satisfaction Signal**  
   - Impact of late delivery on `review_score`  
   - Rating distribution between late vs on-time orders

2. **Delivery Performance Signal**  
   - Effect of `delivery_time` duration  
   - Proportion of late deliveries

3. **Repeat Buyer Signal**  
   - Purchase frequency per `customer_unique_id`  
   - Dependence on one-time buyers

4. **Transaction Value Signal**  
   - `total_value` as economic attractiveness  
   - Relationship between value, lateness, and satisfaction

---

### 📊 Key Insights

| Insight Category | Key Finding | Business Impact |
|---|---|---|
| ⭐ Satisfaction Collapse | Late deliveries reduce average rating from ~4.1 → ~2.5 | Seller reputation drops sharply |
| 🔁 Loyalty Risk | Majority of buyers are one-time customers | Revenue base is unstable |
| 💸 Value Erosion | Late orders show lower transaction value | Profitable buyers leave first |
| ⏱ Delivery Sensitivity | Longer delivery = lower rating | Buyer experience mirrors seller ops |

**Behavioral Flow Identified**

Late delivery  
→ lower review score  
→ fewer repeat buyers  
→ declining order value  
→ weakened seller performance  
→ potential seller churn

### Analytical Approach

- **Exploratory Data Analysis (EDA):** Understanding data distributions and patterns
- **Correlation Analysis:** Identifying relationships between buyer behavior and seller performance
- **Segmentation Analysis:** Examining patterns across product categories and customer segments
- **Trend Analysis:** Tracking changes in buyer behavior over time

### 🛠️ Methodology

#### Data Preprocessing

**1. Missing Value Treatment**
   - Product category: Flagged and retained for analysis
   - Delivery status: Created binary flag for delivered orders
   - Review data: Preserved NaN values with review flag indicator
   - Payment information: Handled missing sequential data
   - Geolocation: Managed missing city/state information

**2. Feature Engineering**
   - Delivery delay calculation (actual vs. estimated)
   - Review flag (reviewed vs. not reviewed)
   - Repeat buyer identification
   - Category-based segmentation

**3. Data Quality Checks**
   - Duplicate detection and removal
   - Outlier identification
   - Data consistency validation

## 2.2 Seller Analysis
### Data Cleaning & Engineering Actions

To ensure the analysis reflects the true operational reality, I performed advanced data engineering rather than simple row deletion.

| Data Issue | Action Taken | Rationale & Methodology |
| :--- | :--- | :--- |
| **Missing Logistics Dates** | **Seller-Specific Median Imputation** | Global averages hide bad performers. I calculated the *Median Processing Time* for each specific seller to fill missing `approval` and `carrier_handover` dates. This restored **100% of the timeline** for active orders. |
| **Price Outliers** | **Strategic Retention** | Detected high-ticket items (Price > IQR). I decided to **keep them** because they contribute **35.61% of Total Revenue**. Removing them would distort the financial value of the platform. |
| **Complex IDs** | **ID Standardization** | Transformed long hash UUIDs (e.g., `3442f8...`) into human-readable formats (e.g., `S0001`, `S0798`) to make visualizations and reporting easier to understand. |
| **Language Barrier** | **Translation Mapping** | The raw data contained Portuguese category names. I mapped them to English and manually fixed missing translations. |
| **"Ghost" Shipments** | **Logic Correction** | Identify orders marked as `delivered` but missing arrival dates. Imputed these specific gaps using the seller's historical *Transit Time* median to prevent data loss in logistics analysis. |

### Seller Perspective: The Operational "Black Box"

To understand the health of the Olist ecosystem, we must look at the platform through the eyes of the Seller. Our initial audit suggests that while Olist solves the problem of "Market Access," it creates a new challenge: **Logistics Complexity**.

We have identified two primary friction points that we aim to investigate in this notebook:

**1. The "First Mile" Variance**
*   **The Problem:** Once an order is approved, the control shifts entirely to the seller. Currently, Olist has limited visibility into how long a seller takes to pack and hand over an item.
*   **The Investigation:** We hypothesize that high-volume sellers might be struggling with processing backlogs. Does selling *more* inevitably mean delivering *slower*? We aim to find out if there is a specific "tipping point" where operational quality breaks down.

**2. The Geography Mismatch**
*   **The Problem:** Brazil is massive. If a seller is based in the South but most buyers are in the North, the "Time-to-Delivery" is physically capped by distance.
*   **The Investigation:** We suspect there are "Supply Deserts"—regions with high buyer demand but zero local sellers. Identifying these gaps is crucial because serving these buyers from far away destroys profit margins.

**3. Business Impact:**
If these frictions are left unchecked, they lead to two silent killers:
*   **Margin Erosion:** High freight costs due to inefficient shipping routes.
*   **Silent Churn:** Sellers don't just "quit" suddenly; they likely experience operational failure (delays) first. If we can't detect these delays, we can't prevent them from leaving.

---

### Strategic Recommendations Framework

Based on the problems identified above, this project aims to formulate a three-pillared strategy. *Note: Specific targets and city names will be revealed in the final conclusion.*

**Strategy 1: The "Local-to-Local" Initiative**
*   **Objective:** Reduce freight costs and unlock "Same-Day Delivery."
*   **Approach:** Instead of recruiting sellers randomly, we will use **Geospatial Analysis** to pinpoint specific cities with the highest "Buyer-to-Seller Ratio." We recommend focusing acquisition efforts strictly on these high-demand zones to localize the supply chain.

**Strategy 2: Proactive "Health Scorecards"**
*   **Objective:** Shift from reactive churn management to proactive support.
*   **Approach:** Currently, we only know a seller is in trouble when they cancel orders. We propose building an **Early Warning System** based on "Processing Time" benchmarks. If a seller starts slowing down relative to their peers, the system should trigger an intervention *before* the customer complains.

**Strategy 3: Targeted Infrastructure Support**
*   **Objective:** Support high-volume product clusters.
*   **Approach:** Not all products are equal. We aim to identify specific product categories (e.g., heavy goods or fragile items) that consistently cause delays. The recommendation will involve establishing specialized **Logistics Collection Points** in key hubs to assist sellers in these difficult categories.
## 3. Data Overview





# 4. Seller Churn Prediction Model

## 📋 Project Overview

This project develops a machine learning model to predict seller churn on an e-commerce platform (Olist). The goal is to identify sellers at risk of leaving the platform, enabling proactive retention interventions that maximize business value.

**Key Achievement:** The tuned model delivers a **net benefit of $285,945** compared to -$897,500 for no intervention, representing a **69% reduction in missed churn cases**.

---

## 🎯 Business Context

### Problem Statement
Seller churn directly impacts marketplace revenue and ecosystem health. Identifying at-risk sellers early allows for targeted retention efforts, but false alarms waste resources. This project balances these trade-offs through cost-aware machine learning.

### Success Metrics
- **Primary:** Maximize recall (catch as many churners as possible)
- **Secondary:** Optimize business impact (total benefit after accounting for intervention costs)
- **Constraint:** Minimize false positive costs while accepting some risk

### Business Assumptions
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Average Quarterly Revenue per Seller** | $500 | Baseline seller contribution |
| **Seller Lifetime Value (LTV)** | $2,500 | 5 quarters average |
| **Retention Program Cost** | $265/seller | Intervention expense (FP cost) |
| **Churn Prevention Success Rate** | 70% | Intervention effectiveness |
| **True Positive Benefit** | $1,750 | Value of saving a seller (70% of LTV) |
| **False Negative Cost** | $2,500 | Lost revenue from missed churner |

---

## 📊 Dataset

### Source
- **Primary Dataset:** `seller_data.csv`
- **Period Covered:** Multiple quarters of transaction data
- **Granularity:** Seller-level quarterly observations

### Key Features
- **Seller Information:** seller_id, location (city, state, zip code)
- **Order Details:** order_id, product_id, category, pricing, freight
- **Temporal Data:** order timestamps, delivery dates, quarterly periods
- **Transaction Metrics:** price, freight_value, order status
- **Customer Data:** customer_id, geolocation

### Feature Engineering
The model uses engineered features capturing:
- **Quarterly Transaction Patterns:** order frequency, revenue trends
- **Seller Engagement:** active months, product diversity, customer reach
- **Performance Indicators:** average order value, fulfillment rates
- **Temporal Trends:** tenure, quarter-over-quarter growth, activity gaps

---

## 🔄 Project Workflow

### 1. Data Cleaning
- **Missing Value Treatment:**
  - `product_category_name_english`: Imputed based on context
  - `order_approved_at`: Handled missing timestamps
  - `order_delivered_carrier_date` & `order_delivered_customer_date`: Delivery data cleaning
  - `geolocation_city` & `geolocation_state`: Location standardization
  
- **Data Type Conversion:** Ensured correct datetime and categorical formats
- **Duplicate Detection:** Identified and resolved duplicate records

### 2. Churn Definition & Labeling
**Churn Definition:** A seller is considered "churned" if they have **no activity for an entire quarter** after previously being active.

**Process:**
1. Filter to approved orders only
2. Create quarterly time periods
3. Track seller activity per quarter
4. Label churned vs. active sellers
5. Generate churn features (last active quarter, inactivity duration, etc.)

### 3. Feature Engineering
Created comprehensive feature set including:
- Quarterly aggregations (revenue, orders, products)
- Behavioral trends (growth rates, volatility)
- Engagement metrics (customer count, category diversity)
- Temporal indicators (tenure, seasonality)

### 4. Model Development

#### Preprocessing Pipeline
```python
ColumnTransformer([
    ('numeric_features', StandardScaler(), numeric_cols),
    ('categorical_features', OneHotEncoder(), categorical_cols)
])
```

#### Models Evaluated
- **Decision Tree Classifier** (baseline and tuned)
- **Logistic Regression**
- **K-Nearest Neighbors**
- **Ensemble Methods** (potential future work)

#### Cross-Validation Strategy
- **Method:** `GroupTimeSeriesSplit` (time-aware, preserves temporal order)
- **Splits:** 3-fold forward chaining
- **Scoring:** Recall (primary), with business impact evaluation

#### Hyperparameter Tuning
- **Approach:** RandomizedSearchCV
- **Optimization Target:** Recall score
- **Threshold Optimization:** Custom threshold tuning for cost-benefit maximization

### 5. Model Evaluation

#### Performance Comparison
| Model | Total Benefit | FP Cost | FN Cost | TP Benefit |
|-------|--------------|---------|---------|------------|
| **No Model** | -$897,500 | $0 | -$897,500 | $0 |
| **Base Decision Tree** | -$291,265 | -$31,845 | -$577,500 | $318,080 |
| **Tuned Model** | **$285,945** | -$47,850 | -$280,000 | $613,795 |

**Key Improvements:**
- ✅ **69% reduction** in false negative costs (missed churners)
- ✅ **93% increase** in true positive benefits (successful interventions)
- ✅ **Net positive ROI** despite higher false positive costs

### 6. Model Interpretability

#### SHAP (SHapley Additive exPlanations)
- **Purpose:** Global feature importance and individual prediction explanations
- **Method:** TreeExplainer for decision tree model
- **Insights:**
  - Quarterly transaction volume is the strongest predictor
  - Seller engagement metrics (active months, product diversity) are critical
  - Temporal trends (declining revenue, increasing inactivity gaps) amplify churn risk
  - Feature interactions reveal compounding effects of multiple negative signals

#### LIME (Local Interpretable Model-agnostic Explanations)
- **Purpose:** Instance-level explanations for specific predictions
- **Method:** LimeTabularExplainer
- **Use Cases:**
  - Help account managers understand why specific sellers are flagged
  - Validate model logic and identify potential biases
  - Debug misclassifications and edge cases
  - Build stakeholder trust through transparency

---

## 🚀 Installation & Usage

### Requirements
```bash
# Core libraries
numpy
pandas
scikit-learn
xgboost
imbalanced-learn

# Visualization
matplotlib
seaborn
missingno

# Interpretability
shap
lime

# Cross-validation
mlxtend

# Other utilities
scipy
joblib
```

### Installation
```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn missingno shap lime mlxtend scipy joblib
```

### Running the Notebook
```python
# 1. Update the base directory path
base_dir = "path/to/your/dataset/"

# 2. Run cells sequentially
# - Section 1: Data Cleaning
# - Section 2: Churn Processing & Feature Engineering
# - Section 3: Modeling, Evaluation, and Interpretability

# 3. Key outputs:
# - Trained model: saved via joblib
# - Performance metrics: confusion matrix, cost-benefit analysis
# - SHAP plots: feature importance visualizations
# - LIME explanations: individual prediction breakdowns
```

---

## 📈 Results Summary

### Model Strengths
✅ **Financial Impact:** $285,945 net benefit per evaluation period  
✅ **High Recall:** Successfully identifies 69% more churners than baseline  
✅ **Interpretability:** Clear explanations via SHAP and LIME for business stakeholders  
✅ **Actionable:** Enables targeted, cost-effective retention interventions  

### Model Weaknesses
⚠️ **False Positives:** Higher intervention costs for non-churning sellers  
⚠️ **Single Algorithm:** Decision tree may miss complex patterns captured by ensembles  
⚠️ **Class Imbalance:** Churn is minority class requiring careful handling  
⚠️ **Feature Limitations:** Current features may not capture all behavioral nuances  

---

## 💡 Key Insights from SHAP Analysis

### Top Predictive Features
1. **Quarterly Order Volume:** Sharp declines strongly indicate churn risk
2. **Revenue Trends:** Quarter-over-quarter revenue drops amplify churn probability
3. **Engagement Metrics:** Fewer active months and reduced product diversity signal disengagement
4. **Customer Reach:** Declining unique customer counts correlate with churn
5. **Temporal Patterns:** Longer inactivity gaps and newer sellers show higher risk

### Feature Interactions
- Sellers with **simultaneous declines** across revenue, orders, and categories have compounding negative effects
- **High revenue but low order frequency** creates uncertainty (potential high-value bulk sellers vs. one-time transactions)
- **Geographic and category factors** show moderate importance, suggesting segment-specific strategies

---

## 🎯 Business Recommendations

### 1. **Proactive Retention Program**
- Deploy model weekly to score all active sellers
- Flag top 10-20% highest-risk sellers for immediate outreach
- Implement tiered interventions based on seller value and churn probability

### 2. **Targeted Interventions**
Use SHAP/LIME explanations to personalize retention strategies:
- **Low Order Volume → Marketing Credits:** Boost visibility and sales
- **Category Stagnation → Expansion Incentives:** Encourage product diversification
- **Fulfillment Issues → Logistics Support:** Address operational pain points
- **Revenue Decline → Fee Discounts:** Reduce cost burden during downturns

### 3. **Platform Improvements**
- **Seller Dashboards:** Real-time performance metrics and benchmarks
- **Onboarding Optimization:** Strengthen first 90-day engagement
- **Community Building:** Forums, best practices, peer support
- **Payment Acceleration:** Faster payouts to improve cash flow

### 4. **Success Tracking**
- Monitor retention rate lift from model-driven interventions
- Calculate ROI: (prevented churn value - intervention costs) / intervention costs
- A/B test different retention strategies on flagged sellers
- Quarterly model retraining with updated data

---

## 🔬 Future Work

### Model Improvements
- [ ] **Ensemble Methods:** Random Forest, XGBoost, Gradient Boosting for better accuracy
- [ ] **Advanced Feature Engineering:** Rolling window statistics, trend indicators, seasonal adjustments
- [ ] **Deep Learning:** LSTM/GRU for temporal sequence modeling
- [ ] **Cost-Sensitive Learning:** Directly incorporate asymmetric costs into training objective

### Data Enhancements
- [ ] **Customer Feedback:** Review scores, NPS, complaint data
- [ ] **Competitive Intelligence:** Market share, competitor activity
- [ ] **External Factors:** Economic indicators, seasonality, events
- [ ] **Seller Engagement:** Login frequency, platform usage, support tickets

### Operational Integration
- [ ] **Production Deployment:** Real-time scoring API
- [ ] **Automated Alerting:** Notify account managers of high-risk sellers
- [ ] **Intervention Tracking:** CRM integration to log actions and outcomes
- [ ] **Continuous Monitoring:** Model drift detection, performance dashboards

---

## 🤝 Acknowledgments

- **Dataset:** Olist Brazilian E-Commerce dataset
- **Interpretability Tools:** SHAP and LIME libraries
- **Cross-Validation:** mlxtend GroupTimeSeriesSplit for time-series-aware validation
- **Business Framework:** Cost-benefit analysis adapted from customer churn literature

---

## 🔑 Key Takeaways

1. **Business Value First:** Model optimization focused on financial impact, not just accuracy
2. **Interpretability Matters:** SHAP and LIME make ML actionable for non-technical stakeholders
3. **Cost-Aware Decisions:** Explicit handling of asymmetric costs (FN >> FP) drives better outcomes
4. **Continuous Improvement:** Model is a starting point; ongoing iteration and domain expertise essential
5. **Proactive > Reactive:** Early intervention based on predictions prevents revenue loss

---

**🎉 Result:** Transforming seller churn from a revenue leak into a manageable, data-driven retention opportunity.
