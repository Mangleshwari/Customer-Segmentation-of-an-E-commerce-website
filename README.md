# Customer-Segmentation-of-an-E-commerce-website
This repository contains an end‑to‑end implementation of customer segmentation using the Online Retail dataset from Kaggle. The project applies unsupervised learning techniques to group customers based on their purchasing behavior, enabling actionable business insights for targeted marketing and retention strategies.

## Table of Contents
- [1. Data Preparation](#1-data-preparation)
- [2. Exploring the content of variables](#2-exploring-the-content-of-variables)
  - [2.1 Countries](#21-countries)
  - [2.2 Customers and products](#22-customers-and-products)
    - [2.2.1 Cancelling orders](#221-cancelling-orders)
    - [2.2.2 StockCode](#222-stockcode)
    - [2.2.3 Basket price](#223-basket-price)
- [3. Insight on product categories](#3-insight-on-product-categories)
  - [3.1 Product description](#31-product-description)
  - [3.2 Defining product categories](#32-defining-product-categories)
    - [3.2.1 Data encoding](#321-data-encoding)
    - [3.2.2 Clusters of products](#322-clusters-of-products)
    - [3.2.3 Characterizing the content of clusters](#323-characterizing-the-content-of-clusters)
- [4. Customer categories](#4-customer-categories)
  - [4.1 Formating data](#41-formating-data)
    - [4.1.1 Grouping products](#411-grouping-products)
    - [4.1.2 Time spliting of the dataset](#412-time-spliting-of-the-dataset)
    - [4.1.3 Grouping orders](#413-grouping-orders)
  - [4.2 Creating customer categories](#42-creating-customer-categories)
    - [4.2.1 Data enconding](#421-data-enconding)
    - [4.2.2 Creating categories](#422-creating-categories)
- [5. Classifying customers](#5-classifying-customers)
  - [5.1 Support Vector Machine Classifier (SVC)](#51-support-vector-machine-classifier-svc)
    - [5.1.1 Confusion matrix](#511-confusion-matrix)
    - [5.1.2 Leraning curves](#512-leraning-curves)
  - [5.2 Logistic regression](#52-logistic-regression)
  - [5.3 k-Nearest Neighbors](#53-k-nearest-neighbors)
  - [5.4 Decision Tree](#54-decision-tree)
  - [5.5 Random Forest](#55-random-forest)
  - [5.6 AdaBoost](#56-adaboost)
  - [5.7 Gradient Boosting Classifier](#57-gradient-boosting-classifier)
  - [5.8 Let's vote !](#58-lets-vote-)
- [6. Testing the predict](#6-testing-the-predict)
- [7. Conclusion}(#7-conclusion)

Here’s a concise, GitHub‑ready README tailored to your notebook.

***

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
