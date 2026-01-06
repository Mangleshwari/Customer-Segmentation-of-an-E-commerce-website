# Customer-Segmentation-of-an-E-commerce-website
This repository contains an end‑to‑end implementation of customer segmentation using the Online Retail dataset from Kaggle. The project applies unsupervised learning techniques to group customers based on their purchasing behavior, enabling actionable business insights for targeted marketing and retention strategies.

## Table of Contents
- [1. Project Overview](#project-overview)
- [2. Dataset](#dataset)
- [3. Methods](#methods)
  - [3.1 Data Cleaning & Preprocessing](#1-data-cleaning--preprocessing)
  - [3.2 Feature Engineering](#2-feature-engineering)
  - [3.3 Customer-Level Aggregation](#3-customer-level-aggregation)
  - [3.4 Clustering & Evaluation](#4-clustering--evaluation)
  - [3.5 Visualization & Interpretation](#5-visualization--interpretation)
- [4. Repository Structure](#repository-structure)
- [5. How to Run](#how-to-run)
- [6. Results](#results)

# Customer Segmentation for an E‑Commerce Retailer

This project performs end‑to‑end customer segmentation on the UCI Online Retail transactional dataset, using classic RFM analysis, feature engineering, and clustering to identify actionable customer groups for marketing.

## Project overview

The goal is to group customers of an online retail store based on their purchasing behaviour so that marketing teams can target VIP, loyal, at‑risk, and lost customers with appropriate strategies.

Key steps:
- Clean and preprocess raw transaction data (~0.5M rows).  
- Engineer customer‑level features (RFM, monetary value, time and product features).  
- Cluster customers and interpret the resulting segments.

## Dataset

- **Source**: Online Retail dataset (Kaggle).
- **Granularity**: Invoice‑level transaction data (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country).
- **Objective**: Aggregate and transform these records into customer‑level behavioural profiles for segmentation.

> Note: The dataset itself is **not** included in this repository. Please download `data.csv` from the original source and place it in the `data/` folder (or update the path in the notebook).

## Methods

1. **Data cleaning & preprocessing**
   - Remove rows with missing `CustomerID`.  
   - Drop exact duplicates.  
   - Handle cancellations and discounts using invoice patterns and a `QuantityCanceled` field.  
   - Remove non‑product / operational codes (e.g., `POST`, `BANK CHARGES`).  
   - Inspect and treat outliers in quantity and price.

2. **Feature engineering**
   - Compute **TotalPrice** per line and per invoice.  
   - Compute **RFM** features (Recency, Frequency, Monetary value) at customer level.  
   - Create time‑based features (month, weekday, day, hour of purchase).  
   - Build product category features from descriptions using TF‑IDF, TruncatedSVD and K‑Means to cluster product types.

3. **Customer‑level aggregation**
   - Aggregate behaviour per invoice and then per customer (quantities, basket prices, total monetary value, country, product‑category mix).  
   - Normalize / scale features for clustering (StandardScaler).

4. **Clustering & evaluation**
   - Use K‑Means with silhouette scores to select the number of customer clusters (8 clusters used).  
   - Visualize clusters and analyze distributions of RFM and other features per cluster.  
   - Interpret segments as VIP, loyal, almost‑lost, lost, foreign customers, bulk buyers, etc.

5. **Visualization & interpretation**
   - Histograms and bar plots for country revenue, invoices per country.  
   - Time‑based plots (month, weekday, hour).  
   - 2D projections of clusters to show separation between customer groups.  
   - Narrative descriptions of each cluster’s behaviour and business importance.

## Repository structure

Suggested layout:

- `customer-segmentation-2.ipynb` – main Jupyter notebook with full pipeline.  
- `data/` – folder where you place `data.csv` (not tracked).  
- `outputs/` – optional folder for saving cleaned data and cluster assignments (e.g., `finaldatasetV2.csv`, pickled objects).  

You can adjust paths inside the notebook if you use a different structure.

## How to run

1. Create and activate a Python environment (e.g., conda or venv).  
2. Install dependencies (approximate list):

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud
   ```

3. Download the Online Retail dataset and save it as `data/data.csv`.
4. Open the notebook.

   ```bash
   jupyter notebook customer-segmentation-2.ipynb
   ```

5. Run all cells from top to bottom to reproduce cleaning, feature engineering, clustering, and visualizations.

## Results

- Constructed a cleaned, feature‑rich customer dataset from raw online retail transactions.
- Identified 8 distinct customer segments (e.g., VIP, loyal, almost‑lost, foreign, bulk buyers, lost) based on RFM and purchasing patterns.
- Produced interpretable visualizations and summaries to support targeted marketing strategies (retention of VIPs, win‑back of almost‑lost, differentiated treatment for bulk buyers and low‑value customers).
