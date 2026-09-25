# Northgate AI — Stock Predictor & Portfolio Analytics

An end-to-end AI-powered stock market analytics project combining machine learning, statistical validation, sentiment analysis, portfolio analytics, risk analysis, and interactive dashboards.

The project focuses on predicting short-term AAPL price direction and evaluating whether selected feature sets provide useful predictive and trading information under a locked out-of-sample evaluation framework.

---

## 📌 Project Overview

**Northgate AI** is a comprehensive stock-market analysis system built around:

- Machine Learning
- Feature Engineering
- Time-Series Validation
- Out-of-Sample Testing
- Statistical Significance Testing
- Trading Backtesting
- Sentiment Analysis
- Portfolio Optimization
- Risk Analytics
- Interactive Dashboards

The project includes both exploratory research notebooks and reproducible Python scripts for model development, validation, statistical testing, and trading analysis.

---

## 🎯 Main Objective

The primary prediction task is:

> Predict whether AAPL will move upward over the next 5 trading days.

The project evaluates different feature configurations, including:

- TOP-5 selected features
- 24-feature model
- Baseline models
- Market-related features
- Technical indicators
- Macro/market variables
- Sentiment-related components

The final evaluation uses a previously untouched **FINAL OOS test period**.

---

# 🏗️ Project Architecture

```text
Northgate_AI_Stock_Predictor/
│
├── app.py
│
├── dashboard/
│   ├── Model_Comparison.py
│   ├── Portfolio_Analytics.py
│   ├── Price_Prediction.py
│   ├── Recommendations.py
│   ├── Risk_Dashboard.py
│   ├── Sentiment.py
│   └── screenshots/
│
├── notebooks/
│   ├── 01_EDA_Features.ipynb
│   ├── 02_Sequence_Preparation.ipynb
│   ├── 03_Classical_ML.ipynb
│   ├── 04_Deep_Learning.ipynb
│   ├── 05_Sentiment_FinBERT.ipynb
│   ├── 06_Portfolio_Optimization_MPT.ipynb
│   └── 07_Recommendation_Rebalancing.ipynb
│
├── scr/
│   ├── data preparation scripts
│   ├── feature engineering scripts
│   ├── model training scripts
│   ├── walk-forward validation
│   ├── statistical tests
│   ├── trading backtests
│   └── final OOS audit/report scripts
│
├── data/
│   └── processed/
│
├── clean_data.py
├── download_data.py
├── feature_engineering.py
├── inspect_data.py
│
├── .gitignore
└── project_files.txt
```
---

👩‍💻 Author

**Aparna Patel**
