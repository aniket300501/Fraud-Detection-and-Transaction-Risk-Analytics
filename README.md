# Fraud-Detection-and-Transaction-Risk-Analytics

This project develops an **end-to-end fraud detection framework** for credit card transactions, with a strong focus on **business-driven decision making rather than pure model accuracy**. The objective is to detect fraudulent transactions effectively while balancing **financial loss, customer friction, and operational cost**, reflecting how fraud systems operate in real payment and fintech environments.

The project combines **supervised machine learning**, **anomaly detection**, and **business-cost–optimised thresholding** to address the core challenges of transaction fraud: extreme class imbalance, evolving fraud patterns, and the trade-off between false positives and false negatives.

---

## Data Source

The raw dataset used in this project was obtained from **Kaggle**. It consists of **anonymised credit card transactions** with a highly imbalanced fraud label, closely resembling real-world payment data where fraudulent events are rare but financially significant.

Key characteristics:

* Extreme class imbalance (fraud < 1%)
* Transaction-level data
* No personally identifiable information (PII)

---

## Project Objectives

* Detect fraudulent transactions in highly imbalanced data
* Compare multiple supervised machine learning models
* Capture anomalous transaction behaviour using unsupervised methods
* Optimise fraud decision thresholds based on **business cost**, not accuracy
* Balance fraud detection performance with customer experience

---

## Methodology

### Data Preparation

* Scaled numerical features (Time, Amount)
* Stratified train–test split to preserve fraud distribution
* Maintained transaction-level granularity for realistic evaluation

---

### Supervised Fraud Detection Models

The following supervised models were implemented and evaluated:

* **Logistic Regression** (baseline)
* **Cost-Sensitive Logistic Regression** (class-weighted)
* **Random Forest**
* **XGBoost**

These models were assessed using metrics appropriate for imbalanced classification, including:

* ROC–AUC
* Precision–Recall curves
* Confusion matrices across multiple thresholds

---

### Anomaly Detection (Unsupervised)

To capture unusual transaction behaviour not easily detected by supervised models, **Local Outlier Factor (LOF)** was applied:

* Identifies transactions that deviate significantly from normal behaviour
* Helps surface potential emerging or previously unseen fraud patterns
* Complements supervised fraud probability scores

---

### Business-Optimal Threshold Selection

Instead of using an arbitrary probability threshold (e.g. 0.5), this project introduces a **business-cost–based threshold optimisation framework**.

The total business cost is defined as:

* **False Positives** → Fixed operational / customer friction cost
* **False Negatives** → Full transaction amount lost

The optimal fraud decision threshold is chosen to **minimise total expected business cost**, explicitly balancing:

* Fraud losses
* Customer experience
* Operational overhead

This approach reflects how real-world payment platforms tune fraud systems for deployment.

---

### Hybrid Fraud Decision Framework

A hybrid fraud decision rule was implemented by combining:

* Supervised model fraud probabilities (XGBoost)
* LOF anomaly scores

Transactions are flagged as fraud if either:

* The fraud probability exceeds the business-optimal threshold, or
* The anomaly score exceeds a high-risk percentile

This improves recall for high-risk edge cases while maintaining control over false positives.

---

## Key Results & Insights

* Cost-sensitive learning significantly improves fraud recall
* Tree-based models outperform linear models on non-linear fraud patterns
* LOF helps identify anomalous transactions missed by supervised models
* Threshold optimisation has a larger business impact than marginal gains in model accuracy
* Fraud detection is fundamentally a **cost optimisation problem**, not a pure classification task

---

## Disclaimer

This project is for **educational and demonstration purposes only**. It does not constitute financial or fraud prevention advice.
