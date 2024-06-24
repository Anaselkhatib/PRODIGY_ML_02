Prodigy ML 02: Customer Segmentation Using K-Means Clustering

This project implements a K-Means clustering model to segment customers based on their annual income and spending score.
Table of Contents

    Overview
    Datasets
    Requirements
    Installation
    Usage
    Clustering Analysis
    Results
    Contributing

Overview

The goal of this project is to segment customers into distinct groups based on their annual income and spending score. We use K-Means clustering, a popular and straightforward machine learning algorithm, to achieve this.
Datasets

The datasets used in this project include customer information such as:

    CustomerID
    Gender
    Age
    Annual Income (k$)
    Spending Score (1-100)

The datasets are preprocessed to handle missing values, normalize the data, and prepare it for clustering analysis.
Requirements

    Python 3.x
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    plotly (optional, for interactive visualization)

You can install the required packages using:

bash

pip install pandas numpy scikit-learn matplotlib seaborn plotly

Installation

    Clone the repository:

    bash

git clone https://github.com/your-username/prodigy-ml-02.git
cd prodigy-ml-02

Install the required packages:

bash

    pip install -r requirements.txt

Usage

    Ensure your dataset is in the correct format (CSV recommended) and place it in the project directory.
    Open the Jupyter Notebook:

    bash

    jupyter notebook PRODIGY_ML_02.ipynb

    Run the cells sequentially to perform data analysis and clustering.

Clustering Analysis

The clustering analysis includes the following steps:

    Loading and preprocessing the dataset: Handling missing values and normalizing the data.
    Exploratory Data Analysis (EDA): Visualizing the dataset to understand its structure and key characteristics.
    Applying K-Means Clustering: Using the KMeans class from the scikit-learn library to segment the customers.
    Evaluating the clusters: Using metrics like silhouette score to assess the quality of the clusters.

Example code snippet for applying K-Means clustering:

python

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Assuming X is the preprocessed feature matrix
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# Predicting the clusters
clusters = kmeans.predict(X)

# Evaluating the clusters
silhouette_avg = silhouette_score(X, clusters)
print(f"Silhouette Score: {silhouette_avg:.2f}")

Results

The performance of the clustering model is evaluated based on:

    Silhouette Score
    Visual inspection of the clusters

Example output:

yaml

Silhouette Score: 0.55

Visualizations of the clusters can be generated using matplotlib and plotly for better interpretation of the results.
Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.