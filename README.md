# 🚀 AI Startup Success Predictor

An end-to-end Machine Learning project that analyzes startup data and predicts the potential unicorn status of a startup.

The project combines startup funding data, Indian AI startup data, and global unicorn data to perform data cleaning, dataset integration, exploratory data analysis, feature engineering, machine learning, model evaluation, and interactive dashboard development.

---

## 📌 Project Overview

The startup ecosystem generates large amounts of information about funding, valuation, employees, industries, startup stages, and company growth.

However, identifying startups with strong potential to become unicorns is difficult because many different business and financial factors influence startup success.

This project uses Machine Learning to analyze startup characteristics and estimate the probability of a startup being classified as a unicorn.

The final system provides an interactive Streamlit dashboard where users can:

- Explore startup data
- Analyze funding trends
- Analyze unicorn companies
- Explore sectors and countries
- Enter startup information
- Predict potential unicorn status
- View the model's estimated unicorn probability
- View model evaluation metrics

---

## 🎯 Objectives

The main objectives of this project are:

1. Collect and combine startup-related datasets.
2. Clean and validate raw startup data.
3. Integrate multiple startup datasets.
4. Perform exploratory data analysis.
5. Engineer useful Machine Learning features.
6. Train classification models.
7. Evaluate model performance.
8. Save the trained model.
9. Build an interactive Streamlit dashboard.
10. Provide startup success probability predictions.

---

## 🗂️ Datasets

The project uses three major datasets.

### 1. Startup Funding Dataset

Contains startup funding and investment information.

Important features include:

- Company
- Deal Date
- Round Type
- Funding Amount
- Valuation
- Investors
- Sector
- Subsector
- Headquarters
- Founded Year
- Employees
- Revenue
- Profitability
- IPO Status

### 2. India AI Startup Dataset

Contains information about AI startups operating in India.

Important features include:

- Company
- Website
- Founded Year
- City
- State
- Startup Stage
- AI Type
- Sector
- Subsector
- Total Funding
- Latest Funding Round
- Employees
- Unicorn Status
- Valuation
- Government Backing
- IIT Founded
- YC Backed
- Description

### 3. Global Unicorn Dataset

Contains global unicorn startup information.

Important features include:

- Company
- Valuation
- Date Joined
- Country
- City
- Industry

---

## 📊 Dataset Statistics

The current integrated project contains:

| Dataset | Records |
|---|---:|
| Funding Dataset | 38 |
| India AI Dataset | 112 |
| Unicorn Dataset | 1,233 |
| Total Integrated Records | 1,383 |

The current labeled modeling dataset contains:

- 112 startup records
- 101 non-unicorn records
- 11 unicorn records

Because the labeled dataset is relatively small, the Machine Learning component should be considered an academic/research prototype rather than a production prediction system.

---

## 🔄 Project Workflow

```text
Raw Startup Datasets
        ↓
Data Cleaning
        ↓
Data Validation
        ↓
Dataset Integration
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Trained Model
        ↓
Streamlit Dashboard
        ↓
Startup Success Prediction
