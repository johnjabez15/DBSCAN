# DBSCAN Clustering – Customer Segmentation

## Overview

This project implements a **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)** algorithm to perform customer segmentation.

The model is trained on a synthetic dataset and deployed via a **Flask** web application, which allows users to input new customer data and predict which segment they belong to. The application also provides a real-time visualization of the new customer's position relative to the existing clusters.

---

## Project Structure

```
DataScience/
│
├── DBSCAN/
│   ├── model/
│   │   ├── dbscan_model.pkl
│   │   ├── scaler.pkl
│   │   └── training_data.pkl
│   ├── static/
│   │   ├── clustering_plot.png
│   │   └── style.css
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   ├── train_model.py
│   ├── app.py
│   └── requirements.txt
```

---

## Installation & Setup

1.  **Clone the repository**

    ```
    git clone <your-repo-url>
    cd "DataScience/DBSCAN"
    ```

2.  **Create a virtual environment (recommended)**

    ```
    python -m venv venv
    source venv/bin/activate    # For Linux/Mac
    venv\Scripts\activate       # For Windows
    ```

3.  **Install dependencies**

    ```
    pip install -r requirements.txt
    ```

---

## Dataset

The synthetic dataset contains customer behavior data with the following features:

* **Annual_Spending** (numeric)
* **Visit_Frequency** (numeric)
* **Time_Spent_Online** (numeric)

This is an unsupervised learning problem, so there is no pre-defined target variable. The model generates cluster labels based on the density of data points.

---

## Problem Statement

Understanding customer behavior is crucial for effective marketing and business strategy. By segmenting customers into distinct groups, we can identify high-value clients, at-risk users, and other key demographics. This project uses DBSCAN to automatically discover these patterns in customer data, including the identification of unusual (outlier) customers.

---

## Why DBSCAN?

* **No Predefined Number of Clusters:** Unlike K-Means, DBSCAN does not require you to specify the number of clusters in advance, as it discovers them automatically.
* **Handles Irregular Shapes:** It can find clusters of arbitrary shapes, whereas K-Means assumes spherical clusters.
* **Robust to Outliers:** The algorithm can effectively identify and label outliers (noise) as a separate category, which is perfect for identifying unusual customer behavior.

---

## How to Run

1.  **Train the Model**

    ```
    python train_model.py
    ```

    This will create the following files in the `model/` directory:

    * `dbscan_model.pkl` (the trained DBSCAN model)
    * `scaler.pkl` (the fitted data scaler)
    * `training_data.pkl` (the pre-processed training data with cluster labels)

2.  **Run the Flask App**

    ```
    python app.py
    ```

    Visit `http://127.0.0.1:5000/` in your browser.

---

## Frontend Input Example

Example customer behavior input:

```
Annual Spending ($): 400
Visit Frequency (times/year): 30
Time Spent Online (hours/month): 200
```

---

## Prediction Goal

The application predicts the customer segment, for example: `High Value & Highly Engaged Customers`.

---

## Tech Stack

* **Python** – Core programming language
* **Pandas & NumPy** – Data manipulation
* **Scikit-learn** – Machine learning model training
* **Flask** – Web framework for deployment
* **Matplotlib** – Data visualization
* **HTML/CSS** – Frontend UI design

---

## Future Scope

* Add a user interface for tuning the DBSCAN parameters (`eps` and `min_samples`) to see how they affect the clustering results.
* Deploy the application on a cloud platform like Heroku or Render for wider public access.
* Integrate a more interactive visualization library like Plotly to allow users to hover over data points for more information.

---

## Screen Shots

**Sample Input 1:**

<img width="1920" height="1080" alt="Screenshot (51)" src="https://github.com/user-attachments/assets/b2cc4bd2-ff12-4344-be6b-540f5ff1649b" />
<img width="1920" height="1080" alt="Screenshot (52)" src="https://github.com/user-attachments/assets/0c703736-5ce8-493c-ba23-3bd33dd994dc" />



**Sample Input 2:**

<img width="1920" height="1080" alt="Screenshot (53)" src="https://github.com/user-attachments/assets/18dc14a6-5761-4ae2-bc2d-c5661b82d839" />
<img width="1920" height="1080" alt="Screenshot (54)" src="https://github.com/user-attachments/assets/e0a181ec-95fc-42a5-992d-8025c723911a" />


