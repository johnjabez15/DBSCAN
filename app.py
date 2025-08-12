from flask import Flask, render_template, request, url_for, send_from_directory
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# --- Paths ---
MODEL_PATH = os.path.join("model", "dbscan_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")
TRAINING_DATA_PATH = os.path.join("model", "training_data.pkl")
PLOT_PATH = os.path.join("static", "clustering_plot.png")

# --- Load Model, Scaler, and Training Data ---
try:
    with open(MODEL_PATH, "rb") as f:
        dbscan_model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(TRAINING_DATA_PATH, "rb") as f:
        training_df = pickle.load(f)
except FileNotFoundError:
    print("Error: Model, scaler, or training data files not found. Please run 'train_model.py' first.")
    exit()

# --- Meaningful Cluster Names ---
# This dictionary maps the numerical labels to descriptive names.
# -1 is always DBSCAN's label for outliers (noise).
cluster_names = {
    0: "Moderate Spenders & Engaged Customers",
    1: "High Value & Highly Engaged Customers",
    2: "Low Spending & Infrequent Customers"
}

def get_cluster_name(label):
    """
    Returns the descriptive name for a given cluster label.
    """
    if label == -1:
        return "Outlier (Noise)"
    return cluster_names.get(label, "Unclassified Cluster")

def generate_plot(new_point, new_cluster_label):
    """
    Generates a scatter plot of the clusters and highlights the new data point.
    """
    plt.figure(figsize=(10, 8))

    # Get unique cluster labels from the training data
    unique_clusters = np.unique(training_df['Cluster'])
    
    # Use a colormap for the clusters
    cluster_colors = plt.cm.get_cmap('viridis', len(unique_clusters[unique_clusters != -1]))

    # Plot each existing cluster
    color_index = 0
    for i, cluster_label in enumerate(unique_clusters):
        if cluster_label == -1:
            # Noise points
            plt.scatter(training_df[training_df['Cluster'] == cluster_label]['Annual_Spending'],
                        training_df[training_df['Cluster'] == cluster_label]['Visit_Frequency'],
                        c='gray', marker='x', s=50, label='Outlier (Noise)')
        else:
            # Core clusters
            plt.scatter(training_df[training_df['Cluster'] == cluster_label]['Annual_Spending'],
                        training_df[training_df['Cluster'] == cluster_label]['Visit_Frequency'],
                        c=[cluster_colors(color_index)], s=100, alpha=0.8, label=get_cluster_name(cluster_label))
            color_index += 1

    # Highlight the new data point
    new_point_name = get_cluster_name(new_cluster_label)
    if new_cluster_label == -1:
        new_point_color = 'red'
    else:
        new_point_color = 'blue'

    plt.scatter(new_point[0], new_point[1], c=new_point_color, marker='*', s=250, edgecolor='white', linewidth=1.5, label=f'New Customer ({new_point_name})', zorder=5)

    plt.title('DBSCAN Customer Segmentation', fontsize=16)
    plt.xlabel('Annual Spending ($)', fontsize=12)
    plt.ylabel('Visit Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

@app.route("/")
def index():
    """
    Renders the input form page.
    """
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Processes the form data, predicts the cluster, and generates the visualization.
    """
    try:
        # Get form inputs
        annual_spending = float(request.form["annual_spending"])
        visit_frequency = float(request.form["visit_frequency"])
        time_spent_online = float(request.form["time_spent_online"])

        # Prepare input data for the model
        input_data = np.array([annual_spending, visit_frequency, time_spent_online]).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)

        # Predict the cluster for the new point.
        # This is a simplified approach, as DBSCAN doesn't have a native predict method.
        # We assign the cluster of the nearest data point.
        distances = np.linalg.norm(dbscan_model.components_ - input_data_scaled, axis=1)
        nearest_core_point_idx = np.argmin(distances)
        new_cluster_label = dbscan_model.labels_[dbscan_model.core_sample_indices_[nearest_core_point_idx]]

        if distances[nearest_core_point_idx] > dbscan_model.eps:
            new_cluster_label = -1

        # Get the meaningful name for the cluster
        cluster_name = get_cluster_name(new_cluster_label)
        
        # Generate and save the visualization
        generate_plot(input_data[0], new_cluster_label)

        # Prepare the text result
        if new_cluster_label == -1:
            result = f"This customer is an **{cluster_name}**."
        else:
            result = f"This customer belongs to the **{cluster_name}** group."

        return render_template("result.html", prediction=result, plot_url=url_for('static', filename='clustering_plot.png'))

    except (ValueError, KeyError) as e:
        return f"Error: Invalid input data. Please ensure all fields are filled correctly. Details: {str(e)}", 400
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
