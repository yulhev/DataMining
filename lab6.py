import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, BisectingKMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids, CLARA
from pyclustering.cluster.clarans import clarans
from minisom import MiniSom

df = pd.read_csv(r'D:\Homeworks\Data Mining\spotify.csv' )

# print("First 5 rows:\n", df.head())
# print("Shape:\n", df.shape)
# print("Columns:\n", df.columns)
# print("Data types:\n",df.dtypes)

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()
df = df.replace('?', np.nan)
# print("Missing values after replacement:\n", df.isnull().sum())

for col in df.select_dtypes(include='number').columns:
    print(col, (df[col] == 0).sum())

cat_cols = df.select_dtypes(include='object').columns
str_imp = SimpleImputer(strategy='most_frequent')
df[cat_cols] = str_imp.fit_transform(df[cat_cols])

df['tempo'] = df['tempo'].replace(0, np.nan)
num_cols = df.select_dtypes(include='number').columns
imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])
# print("Missing values after replacement:\n", df.isnull().sum())

X = df.select_dtypes(include='number')

zscore = StandardScaler()
X_scaled = zscore.fit_transform(X)

print("Full data shape:", X_scaled.shape)

sample_size = 5000
rng = np.random.RandomState(42)
sample_idx = rng.choice(X_scaled.shape[0], sample_size, replace=False)
X_sample = X_scaled[sample_idx]

print("Sample shape:", X_sample.shape)

inertia = []
k_range = range(2, 11)
#
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X_scaled)
#     inertia.append(kmeans.inertia_)
#
# plt.plot(k_range, inertia)
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Inertia")
# plt.title("Elbow Method")
# plt.show()
#
# sil_scores = []
#
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(X_scaled)
#     score = silhouette_score(
#         X_scaled,
#         labels,
#         sample_size=8500,
#         random_state=42
#     )
#     sil_scores.append(score)
#     print(f"k={k}, silhouette={score:.4f}")
#
# plt.plot(k_range, sil_scores)
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Silhouette Score")
# plt.title("Silhouette Method")
# plt.show()

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)
print("K-Means:")
print("Inertia:", kmeans.inertia_)
print("Silhouette:",
      silhouette_score(X_scaled, df['cluster'], sample_size=5000, random_state=42))
print("Calinski-Harabasz:",
      calinski_harabasz_score(X_scaled, df['cluster']))
print("Davies-Bouldin:",
      davies_bouldin_score(X_scaled, df['cluster']))

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis')
# plt.title("KMeans Clusters (k=4)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.show()

pam = KMedoids(n_clusters=4, random_state=42)
pam_labels = pam.fit_predict(X_sample)
print("PAM:")
print("Silhouette:",
      silhouette_score(X_sample, pam_labels, sample_size=5000, random_state=42))
print("Calinski-Harabasz:",
      calinski_harabasz_score(X_sample, pam_labels))
print("Davies-Bouldin:",
      davies_bouldin_score(X_sample, pam_labels))

# pam_scores = []
# for k in k_range:
#     pam = KMedoids(n_clusters=k, random_state=42)
#     labels = pam.fit_predict(X_sample)
#     score = silhouette_score(X_sample, labels, sample_size=2000, random_state=42)
#     pam_scores.append(score)
#     print(f"k={k}, silhouette={score:.4f}")
#
# plt.plot(k_range, pam_scores, marker='o')
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Silhouette Score")
# plt.title("Optimal k for PAM")
# plt.show()

clara = CLARA(n_clusters=4, random_state=42)
clara_labels = clara.fit_predict(X_sample)
print("CLARA:")
print("Silhouette:",
      silhouette_score(X_sample, clara_labels, sample_size=2000, random_state=42))
print("Calinski-Harabasz:",
      calinski_harabasz_score(X_sample, clara_labels))
print("Davies-Bouldin:",
      davies_bouldin_score(X_sample, clara_labels))

clara_scores = []

for k in k_range:
    clara = CLARA(n_clusters=k, random_state=42)
    labels = clara.fit_predict(X_sample)
    score = silhouette_score(X_sample, labels, sample_size=2000, random_state=42)
    clara_scores.append(score)
    print(f"k={k}, silhouette={score:.4f}")

plt.plot(k_range, clara_scores, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Optimal k for CLARA")
plt.show()