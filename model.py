import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# --- Paths and Setup ---
DATA_PATH = os.path.join("dataset", "customer_data.csv")
MODEL_DIR = "model"
DBSCAN_MODEL_PATH = os.path.join(MODEL_DIR, "dbscan_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
TRAINING_DATA_PATH = os.path.join(MODEL_DIR, "training_data.pkl")

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
print("Loading data...")
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATA_PATH}.")
    exit()

# Separate features
features = ['Annual_Spending', 'Visit_Frequency', 'Time_Spent_Online']
X = df[features]

# --- Data Preprocessing ---
print("Training StandardScaler...")
# Standardize the data before clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the trained scaler
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
print(f"StandardScaler saved to {SCALER_PATH}")

# --- DBSCAN Clustering ---
print("Training DBSCAN model...")
# Using default parameters for this example. These can be tuned for better results.
dbscan = DBSCAN(eps=0.5, min_samples=5)
# fit_predict directly returns cluster labels for the training data
df['Cluster'] = dbscan.fit_predict(X_scaled)

# Save the trained DBSCAN model and the training data
with open(DBSCAN_MODEL_PATH, "wb") as f:
    pickle.dump(dbscan, f)
print(f"DBSCAN model saved to {DBSCAN_MODEL_PATH}")

with open(TRAINING_DATA_PATH, "wb") as f:
    pickle.dump(df, f)
print(f"Training data with clusters saved to {TRAINING_DATA_PATH}")

