# 🌸 Iris Species Clustering: A Step-by-Step Guide

> This guide walks through the process of unsupervised learning using the **K-Means Clustering** algorithm on the famous Iris dataset.
> You will learn how to explore data, visualize patterns, find optimal clusters, and evaluate model performance

---

## 📑 Table of Contents
1. [Import Libraries](#step-1-import-libraries)
2. [Load Dataset](#step-2-load-dataset)
3. [Explore Data](#step-3-explore-data)
4. [Data Visualization](#step-4-data-visualization)
5. [feature-scaling](#step-5-feature-scaling)
6. [Elbow Method](#-step-6-elbow-method)
6. [Final Clustering & Evaluation](#step-7-final-clustering-&-evaluation)

---

## 📦 `Step 1: Import Libraries`

```md
We begin by importing the necessary tools for data manipulation, visualization, and machine learning.
```

```python
# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

### Explaination
- pandas → data manipulation
- seaborn & matplotlib → visualization
- load_iris → dataset loader
- KMeans → clustering algorithm
- silhouette_score → evaluation metric
- StandardScaler → feature scaling

---

## 🟢 `Step 2: Load Dataset`


```md
We load and prepare the dataset.
```

### *Code :*

```python

# Load dataset and convert to DataFrame
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Remove duplicates (if exist)
df = df.drop_duplicates().reset_index(drop=True)

# Preview dataset
df.head()
```
---

## 🟢 `Step 3: Explore Data`

```md
We inspect and understand the dataset.
```

### *Code :*

```python
# Inspect dataset

print(iris.target_names)   # target names
df.info()                 # Dataset info
df.describe()             # statistical summary
df.isnull().sum()         # missing values check
```

---

# 🟢 `Step 4: Data Visualization`

```md
We visualize relationships between features.
```

### *Code :*

```python
sns.pairplot(df)
plt.show()
```
### 🖼️ Output

![Feature Relationships](images/Feature_relationships.png)
---

# 🟢 `Step 5: Feature Scaling`

```md
We select important features and scale them for better clustering.
```
### *Code :*

```python
X = df[['petal length (cm)', 'petal width (cm)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

# 🟢 `Step 6: Elbow Method`

```md
We apply K-Means and evaluate performance.
```
### *Code :*

```python
error = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    error.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), error, marker='o', linestyle='--')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()
```

### 🖼️ Output

![Elbow Method](images/k-means_Elbow.png)
---

# 🟢 `Step 7: Final Clustering & Evaluation`

```md
We apply K-Means, visualize clusters, and evaluate performance.
```
### *Code :*

```python
# Train model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X_scaled)

# Plot clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, cmap='viridis')

# Plot centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=200)

plt.title("K-Means Clusters")
plt.show()

# Evaluation
print("Silhouette Score:", silhouette_score(X_scaled, y_pred))
```
## 🖼️ Output :
<div style="display: flex; gap: 10px; align-items: center;">
  <img src="images/K-means_Clusters.png" width="56%" >
  <img src="images/Cluster_Centroids.png" width="44%" >
</div>