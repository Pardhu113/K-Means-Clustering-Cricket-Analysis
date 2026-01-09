"""
K-Means Clustering Analysis on ODI Cricket Player Statistics

This script performs K-Means clustering on ODI (One Day International) 
cricket player performance data. It includes:
- Data loading and cleaning
- Data preprocessing and standardization
- K-Means clustering with optimal k selection
- 3D visualization of clusters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
print("Loading cricket statistics data...")
df = pd.read_csv('ODI_data.csv', encoding='latin1')

# Data cleaning
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df = df.loc[:, ~df.columns.str.contains('unnamed')]

# Handle missing values
df = df.replace('-', np.nan)
df = df.dropna()

print(f"Data shape after cleaning: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Select numeric columns for clustering
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
X = df[numeric_cols]

# Standardization
print("\nStandardizing data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using Elbow method
print("\nFinding optimal number of clusters...")
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)

# Plot Elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True, alpha=0.3)
plt.savefig('elbow_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Fit K-Means with optimal k (using 3 clusters)
print("\nFitting K-Means with k=3...")
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\nCluster distribution:")
print(df['Cluster'].value_counts().sort_index())

# 3D Visualization
print("\nCreating 3D visualization...")
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)
X_scaled_df['Cluster'] = df['Cluster'].values

fig = px.scatter_3d(
    X_scaled_df,
    x=numeric_cols[0],
    y=numeric_cols[1],
    z=numeric_cols[2],
    color='Cluster',
    title='3D K-Means Clustering of Cricket Players',
    labels={numeric_cols[0]: numeric_cols[0], 
            numeric_cols[1]: numeric_cols[1],
            numeric_cols[2]: numeric_cols[2]}
)
fig.write_html('k_means_3d_visualization.html')
fig.show()

print("\nAnalysis complete!")
print(f"Visualization saved to 'k_means_3d_visualization.html'")
