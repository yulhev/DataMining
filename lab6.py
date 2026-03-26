import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

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

inertia = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(k_range, inertia)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

sil_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(
        X_scaled,
        labels,
        sample_size=8500,
        random_state=42
    )
    sil_scores.append(score)
    print(f"k={k}, silhouette={score:.4f}")

plt.plot(k_range, sil_scores)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.show()


