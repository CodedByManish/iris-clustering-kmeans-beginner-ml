import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Load and Prepare Data
# ==========================================
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Check for duplicates (Iris dataset usually has 1 duplicate row)
duplicate_count = df.duplicated().sum()
print(f"Duplicate rows found: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicates removed.")

# ==========================================
# 2. Exploratory Data Analysis (EDA)
# ==========================================
print("\n--- Dataset Info ---")
df.info()

print("\n--- Statistical Summary ---")
print(df.describe())

# Visualize relationships between all features
sns.pairplot(df)
plt.suptitle("Feature Relationships", y=1.02)
plt.show()

# ==========================================
# 3. Feature Selection & Scaling
# ==========================================
# We select Petal Length and Petal Width as they provide the best separation
X = df[['petal length (cm)', 'petal width (cm)']]

# Best Practice: Scale the data
# K-Means is distance-based; scaling ensures features contribute equally
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 4. Finding Optimal K (Elbow Method)
# ==========================================
error = []
for i in range(1, 11):
    # n_init='auto' is recommended in newer sklearn versions
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    error.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), error, marker='o', color='purple')
plt.title('Elbow Method (Finding Optimal K)')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Error)')
plt.show()

# ==========================================
# 5. Apply K-Means with Optimal K (K=3)
# ==========================================
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
y_predict = kmeans.fit_predict(X_scaled)

# ==========================================
# 6. Final Visualization & Evaluation
# ==========================================
# Apply KMeans with K=3
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
y_predict = kmeans.fit_predict(X_scaled)

# Visualization
plt.figure(figsize=(10, 6))

# Plotting Clusters
plt.scatter(X.iloc[y_predict == 0, 0], X.iloc[y_predict == 0, 1], c='red', label='Cluster 1', alpha=0.6)
plt.scatter(X.iloc[y_predict == 1, 0], X.iloc[y_predict == 1, 1], c='green', label='Cluster 2', alpha=0.6)
plt.scatter(X.iloc[y_predict == 2, 0], X.iloc[y_predict == 2, 1], c='blue', label='Cluster 3', alpha=0.6)

# Transform centroids back to original scale for plotting
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=250, c='yellow', marker='*', label='Centroids', edgecolors='black')

plt.title('Final Iris Clusters (K=3)')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
plt.show()

# Final Score
score = silhouette_score(X_scaled, y_predict)
print(f"Silhouette Score: {score:.4f}")








# ==========================================
# PYNB-EDITABLE-END
# ==========================================
# ==========================================Cell 1: Import Libraries
# ==========================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Set plot style
sns.set_theme(style="whitegrid")
# ==========================================Cell 2: Load and Clean Data
# ==========================================
# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Check for and remove duplicates
print(f"Duplicates before: {df.duplicated().sum()}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicates after: {df.duplicated().sum()}")

df.head()
# ==========================================Cell 3: Data Exploration (EDA)
# ==========================================Python
print("--- Target Names ---")
print(iris.target_names)

print("\n--- Statistical Summary ---")
(df.describe())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())
# ==========================================Cell 4: Visualization
# ==========================================Python
# Pairplot to see feature relationships
sns.pairplot(df)
plt.show()
# ==========================================Cell 5: Feature Selection and Scaling
# ==========================================Python
# Selecting Petal Length and Petal Width
X = df[['petal length (cm)', 'petal width (cm)']]

# Scaling data for distance-based algorithm
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X.head()
# ==========================================Cell 6: Elbow Method (Finding K)
# ==========================================Python
error = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    error.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), error, marker='o', color='blue', ls='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()
# ==========================================Cell 7: Final Clustering and Evaluation
# ==========================================Python
# Apply KMeans with K=3
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
y_predict = kmeans.fit_predict(X_scaled)

# Visualization
plt.figure(figsize=(10, 6))

# Plotting Clusters
plt.scatter(X.iloc[y_predict == 0, 0], X.iloc[y_predict == 0, 1], c='red', label='Cluster 1', alpha=0.6)
plt.scatter(X.iloc[y_predict == 1, 0], X.iloc[y_predict == 1, 1], c='green', label='Cluster 2', alpha=0.6)
plt.scatter(X.iloc[y_predict == 2, 0], X.iloc[y_predict == 2, 1], c='blue', label='Cluster 3', alpha=0.6)

# Transform centroids back to original scale for plotting
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=250, c='yellow', marker='*', label='Centroids', edgecolors='black')

plt.title('Final Iris Clusters (K=3)')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
plt.show()

# Final Score
score = silhouette_score(X_scaled, y_predict)
print(f"Silhouette Score: {score:.4f}")
