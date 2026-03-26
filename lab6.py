import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, BisectingKMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn_extra.cluster import KMedoids, CLARA
from pyclustering.cluster.clarans import clarans
from minisom import MiniSom

def print_metrics(name, X, labels, sample_size=5000, inertia=None):
    print(f"\n{name}:")
    print("Number of clusters:", len(set(labels)) - (1 if -1 in labels else 0))

    if inertia is not None:
        print("Inertia:", inertia)

    if -1 in labels:
        mask = labels != -1
        X_eval = X[mask]
        labels_eval = labels[mask]
        print("Noise points:", np.sum(labels == -1))
    else:
        X_eval = X
        labels_eval = labels

    if len(set(labels_eval)) > 1:
        print("Silhouette:",
              silhouette_score(X_eval, labels_eval, sample_size=min(sample_size, len(X_eval)), random_state=42))
        print("Calinski-Harabasz:",
              calinski_harabasz_score(X_eval, labels_eval))
        print("Davies-Bouldin:",
              davies_bouldin_score(X_eval, labels_eval))
    else:
        print("Not enough clusters for evaluation")


def plot_k(model_factory, X, k_values, title, sample_size=5000):
    scores = []

    for k in k_values:
        model = model_factory(k)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=min(sample_size, len(X)), random_state=42)
        scores.append(score)
        print(f"k={k}, silhouette={score:.4f}")

    plt.plot(list(k_values), scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title(title)
    plt.show()

    return scores


def som_labels_from_model(som, X, size):
    labels = []
    for row in X:
        winner = som.winner(row)
        label = winner[0] * size + winner[1]
        labels.append(label)
    return np.array(labels)


def clarans_labels_from_clusters(clusters, n_points):
    labels = np.full(n_points, -1)
    for cluster_id, indices in enumerate(clusters):
        for idx in indices:
            labels[idx] = cluster_id
    return labels


def plot_dbscan(X, eps_values, min_samples=10, sample_size=5000):
    db_scores = []
    cluster_counts = []
    noise_counts = []

    print("\nDBSCAN parameter tuning:")
    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        cluster_counts.append(n_clusters)
        noise_counts.append(n_noise)

        mask = labels != -1
        labels_no_noise = labels[mask]

        if len(set(labels_no_noise)) > 1:
            score = silhouette_score(
                X[mask],
                labels_no_noise,
                sample_size=min(sample_size, np.sum(mask)),
                random_state=42
            )
        else:
            score = np.nan

        db_scores.append(score)
        print(f"eps={eps}, clusters={n_clusters}, noise={n_noise}, silhouette={score}")

    return db_scores, cluster_counts, noise_counts


df = pd.read_csv(r'D:\Homeworks\Data Mining\spotify.csv')

print("First 5 rows:\n", df.head())
print("Shape:\n", df.shape)
print("Columns:\n", df.columns)
print("Data types:\n",df.dtypes)

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

df = df.replace('?', np.nan)

print("Missing values after replacement:\n", df.isnull().sum())

for col in df.select_dtypes(include='number').columns:
    print(col, (df[col] == 0).sum())

cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])

df['tempo'] = df['tempo'].replace(0, np.nan)

num_cols = df.select_dtypes(include='number').columns
df[num_cols] = KNNImputer(n_neighbors=5).fit_transform(df[num_cols])

co_mtx = df.corr(numeric_only=True)
print(co_mtx)
plt.figure(figsize=(16, 12))
sns.heatmap(co_mtx, cmap="YlGnBu")
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()

X = df.select_dtypes(include='number')
X_scaled = StandardScaler().fit_transform(X)

print("Full data shape:", X_scaled.shape)

sample_size = 5000
rng = np.random.RandomState(42)
sample_idx = rng.choice(X_scaled.shape[0], sample_size, replace=False)
X_sample = X_scaled[sample_idx]

print("Sample shape:", X_sample.shape)

k_range = range(2, 11)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
print_metrics("K-Means", X_scaled, kmeans_labels, sample_size=5000, inertia=kmeans.inertia_)
plot_k(lambda k: KMeans(n_clusters=k, random_state=42, n_init=10), X_scaled, k_range, "Optimal k for K-Means")

pam = KMedoids(n_clusters=4, random_state=42)
pam_labels = pam.fit_predict(X_sample)
print_metrics("PAM", X_sample, pam_labels)
plot_k(lambda k: KMedoids(n_clusters=k, random_state=42),X_sample, k_range, "Optimal k for PAM")

clara = CLARA(n_clusters=4, random_state=42)
clara_labels = clara.fit_predict(X_sample)
print_metrics("CLARA", X_sample, clara_labels)
plot_k(lambda k: CLARA(n_clusters=k, random_state=42), X_sample, k_range, "Optimal k for CLARA")

agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
agg_labels = agg.fit_predict(X_sample)
print_metrics("Agglomerative", X_sample, agg_labels)
plot_k(lambda k: AgglomerativeClustering(n_clusters=k, linkage='ward'),X_sample, k_range, "Optimal k for Agglomerative")

divisive = BisectingKMeans(n_clusters=4, random_state=42, n_init=10)
divisive_labels = divisive.fit_predict(X_sample)
print_metrics("Divisive (BisectingKMeans)", X_sample, divisive_labels)
plot_k(lambda k: BisectingKMeans(n_clusters=k, random_state=42, n_init=10),X_sample, k_range, "Optimal k for Divisive")


dbscan = DBSCAN(eps=2.0, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_sample)
print_metrics("DBSCAN", X_sample, dbscan_labels)

eps_values = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
db_scores, cluster_counts, noise_counts = plot_dbscan(X_sample, eps_values, min_samples=10)

plt.figure(figsize=(8, 5))
plt.plot(eps_values, db_scores, marker='o')
plt.xlabel("eps")
plt.ylabel("Silhouette Score")
plt.title("DBSCAN Parameter Tuning")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(eps_values, cluster_counts, marker='o')
plt.xlabel("eps")
plt.ylabel("Number of clusters")
plt.title("DBSCAN: Clusters vs eps")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(eps_values, noise_counts, marker='o')
plt.xlabel("eps")
plt.ylabel("Number of noise points")
plt.title("DBSCAN: Noise points vs eps")
plt.grid(True)
plt.show()


som = MiniSom(
    x=2, y=2,
    input_len=X_sample.shape[1],
    sigma=1,
    learning_rate=0.3,
    random_seed=42
)
som.random_weights_init(X_sample)
som.train_random(X_sample, 5000)

som_labels = som_labels_from_model(som, X_sample, size=2)
print_metrics("SOM", X_sample, som_labels)

som_sizes = [2, 3, 4, 5]
som_scores = []
som_cluster_counts = []

for size in som_sizes:
    som = MiniSom(
        x=size, y=size,
        input_len=X_sample.shape[1],
        sigma=1,
        learning_rate=0.3,
        random_seed=42
    )
    som.random_weights_init(X_sample)
    som.train_random(X_sample, 5000)

    labels = som_labels_from_model(som, X_sample, size)
    n_clusters = len(set(labels))
    som_cluster_counts.append(n_clusters)

    if n_clusters > 1:
        score = silhouette_score(X_sample, labels, sample_size=5000, random_state=42)
    else:
        score = np.nan

    som_scores.append(score)
    print(f"size={size}x{size}, clusters={n_clusters}, silhouette={score}")

plt.plot(som_cluster_counts, som_scores, marker='o')
plt.xlabel("Number of SOM clusters")
plt.ylabel("Silhouette Score")
plt.title("SOM Map Size Comparison")
plt.show()


clarans_sample_size = 500
clarans_idx = rng.choice(X_scaled.shape[0], clarans_sample_size, replace=False)
X_clarans = X_scaled[clarans_idx]
X_list = X_clarans.tolist()

print("\nStarting CLARANS...")

clarans_model = clarans(
    data=X_list,
    number_clusters=4,
    numlocal=1,
    maxneighbor=2
)
clarans_model.process()
clarans_clusters = clarans_model.get_clusters()
clarans_labels = clarans_labels_from_clusters(clarans_clusters, len(X_list))

print_metrics("CLARANS", X_clarans, clarans_labels, sample_size=500)

clarans_k_values = []
clarans_scores = []

for k in range(2, 6):
    print(f"Running CLARANS for k={k}...")

    clarans_model = clarans(
        data=X_list,
        number_clusters=k,
        numlocal=1,
        maxneighbor=2
    )
    clarans_model.process()

    clusters = clarans_model.get_clusters()
    labels = clarans_labels_from_clusters(clusters, len(X_list))

    score = silhouette_score(
        X_clarans,
        labels,
        sample_size=min(500, len(X_clarans)),
        random_state=42
    )

    clarans_k_values.append(k)
    clarans_scores.append(score)
    print(f"k={k}, silhouette={score:.4f}")

plt.plot(clarans_k_values, clarans_scores, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Optimal k for CLARANS")
plt.show()