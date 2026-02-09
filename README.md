# Analysis of Olist and Model for Predicting Seller Churn

Contributors: Dyah Almira | Lukman Fathoni | Axel Alexander

<img width="897" height="595" alt="image" src="https://github.com/user-attachments/assets/56f2424c-5076-4acc-bc95-83b7e10da0a7" />

Tableau link : [Olist Dashboard](https://public.tableau.com/app/profile/dyah.almira/viz/Alphagroup_Olist/Homepage)

<img width="1119" height="824" alt="image" src="https://github.com/user-attachments/assets/26474f06-e66e-4c39-8c1e-7e5db593cf2e" />


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

###  Overview

This component conducts a **Strategic Supply Chain Audit** to evaluate merchant efficiency, geographical coverage, and revenue stability within the Olist ecosystem.

### Core Hypothesis

We hypothesize that platform inefficiencies are not just random, but structural. A seller's location and operational speed are the primary predictors of their ability to scale and remain profitable:

Operational Friction (Delays) + Supply Scarcity (Distance) → High Logistics Costs → Margin Erosion

---
###  Objectives

-   **Quantify "First Mile" Latency:** Measure the time from Order Approval to Carrier Handover to identify specific category bottlenecks.
-   **Map Expansion Opportunities:** Calculate the **Buyer-to-Seller Ratio** to locate high-demand cities with zero local inventory.
-   **Assess Revenue Concentration:** Evaluate the financial dependence on high-ticket items (Price Outliers).
-   **Benchmark Merchant Performance:** Contrast high-volume "Super Sellers" against operationally struggling merchants.

---

###  Analytical Framework

Four seller-based signals were investigated:

1.  **Processing Speed Signal**
    -   Analysis of `seller_processing_days` (Approval $\to$ Carrier).
    -   Impact of complex categories (e.g., Furniture, Music) on fulfillment time.

2.  **Geospatial Tension Signal**
    -   Supply vs. Demand density across municipalities.
    -   Identification of "Vacuum Markets" (High Demand / Low Supply).

3.  **Economic Impact Signal**
    -   Revenue contribution of Price Outliers ($>$ IQR).
    -   Stability of "Veteran" sellers vs. "New Entrants."

4.  **Logistics Consistency Signal**
    -   Variance between Mean and Median delivery times.
    -   Detection of "Ghost Shipments" (missing timestamps).

---

### Key Insights

| Insight Category | Key Finding | Business Impact |
|---|---|---|
| ⏳ Bottleneck Detection | **Music & Food** categories average **>17 days** to process | These categories drive disproportionate customer complaints. |
| 🗺️ Market Opportunity | **Guariba** has **1,132 buyers** per **1 seller** | Major missed revenue due to lack of local fulfillment. |
| 💎 Value Distribution | **35.61% of Revenue** comes from Price Outliers | High-value merchants are critical to platform GMV. |
| 🚀 Scale vs. Speed | Top Seller **S0798** processes orders in **~1.5 days** | Proves high volume does not necessitate slow service. |

**Operational Flow Identified**

Inefficient Category (e.g., Music)
→ Extended Processing Time
→ Delayed Handover to Carrier
→ Missed Delivery Promise
→ Increased Logistics Cost & Customer Friction

### Analytical Approach

-   **Logistics Forensics:** Reconstructing broken timelines using seller-specific historical data.
-   **Gap Analysis:** Side-by-side comparison of Seller Count vs. Order Volume per city.
-   **Statistical Outlier Assessment:** Using IQR to determine the financial weight of premium products.
-   **Performance Benchmarking:** Profiling top sellers to establish "Gold Standard" operational metrics.

### 🛠️ Methodology

#### Data Preprocessing

**1. Advanced Data Imputation (Seller-Specific)**
   - **Timestamps:** Applied **Median Imputation** for missing `approval` and `carrier_handover` dates based on each seller's unique history.
   - *Rationale:* A global average would mask the inefficiencies of slow sellers. Individual benchmarks restored **100%** of active timelines.
   - **Ghost Shipments:** Corrected logic for orders marked 'delivered' but lacking arrival dates.

**2. Feature Engineering**
   - **ID Standardization:** Converted hash UUIDs (`3442f8...`) to readable formats (`S0001`) for clarity.
   - **Processing Time Calculation:** Derived precise daily values for the "First Mile" duration.
   - **Translation:** Mapped Portuguese categories to English (e.g., `esporte_lazer` → `sports_leisure`) for accessibility.

**3. Data Quality Checks**
   - **Duplicate Detection:** Rigorous validation of composite keys; **Zero duplicates found.**
   - **Outlier Strategy:** Detected high-price outliers but **Retained** them as they drive over 35% of total revenue.
   - **Logic Validation:** Removed negative processing durations caused by input errors.
###  Strategic Recommendations

1.  **Expansion Strategy (Local-to-Local Fulfillment):**
    *   **Action:** Prioritize merchant acquisition in **Guariba** and **Ilicinea**.
    *   **Goal:** To slash logistics costs and unlock *Same-Day Delivery* capabilities in these under-served zones [Source 95].

2.  **Logistics Infrastructure:**
    *   **Action:** Establish a dedicated Collection Point (Hub) in **Ibitinga**.
    *   **Goal:** To streamline the "First Mile" for the high volume of textile orders originating from this specific cluster [Source 101].

3.  **Churn Prevention System:**
    *   **Action:** Implement a "Logistics Health Scorecard."
    *   **Goal:** Trigger automated alerts if a seller's average processing time exceeds **3 days** (before reaching the critical 7-day failure point), shifting from reactive to proactive support [Source 99].
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
