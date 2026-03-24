import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


df = pd.read_csv("booksToScrapeF.csv")

df["rating_word"] = df["rating"].str.extract(r"(One|Two|Three|Four|Five)")
rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}
df["rating_num"] = df["rating_word"].map(rating_map)

text = ' '.join(df['title'].astype(str).tolist())
text = re.sub(r'[^A-Za-z\s]', '', text)
text = text.lower()
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(["volume", "vol"])

wordcloud = WordCloud(width=800, height=480,background_color='white', stopwords=custom_stopwords).generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

tfidf = TfidfVectorizer(
    stop_words=list(custom_stopwords),
    max_features=3000,
    ngram_range=(1,2),
    min_df=2
)
X_text = tfidf.fit_transform(df["title"].astype(str))

X_num = df[["price", "rating_num"]].values
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)
X = hstack([X_text, csr_matrix(X_num_scaled)])

k_range = range(2, 12)

wcss = []
sil_scores = []
db_scores = []

svd_db = TruncatedSVD(n_components=50, random_state=42)
X_db = svd_db.fit_transform(X)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))
    db_scores.append(davies_bouldin_score(X_db, labels))

plt.figure(figsize=(6,4))
plt.plot(k_range, wcss, marker='o')
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

plt.plot(k_range, sil_scores, marker='o')
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(k_range, db_scores, marker='o')
plt.xlabel("k")
plt.ylabel("Davies-Bouldin Index")
plt.title("Davies-Bouldin Index")
plt.show()

optimal_k = 4

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df["cluster"] = kmeans.fit_predict(X)
print(df.groupby("cluster")[["price", "rating_num"]].mean())
print(df.groupby("cluster")[["price", "rating_num"]].median())

df["Genre"] = df["cluster"].apply(lambda c: f"Cluster_{c}")
df["Genre_label"] = df["cluster"]
print(df["Genre"].value_counts())
print(df["Genre_label"].value_counts())
df.to_csv("booksToScrapeF_C.csv", index=False)

svd = TruncatedSVD(n_components=50, random_state=42)
X_reduced = svd.fit_transform(X)

tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
X_2d = tsne.fit_transform(X_reduced)

plt.figure(figsize=(8,6))

for label in sorted(df["Genre_label"].unique()):
    idx = (df["Genre_label"].to_numpy() == label)
    genre_name = df.loc[df["Genre_label"] == label, "Genre"].iloc[0]
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=10, label=genre_name)

plt.title("Book Cluster Visualization by Discovered Genres")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.show()

#----------------------------------------------------------------

y = df["Genre_label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = [
    {"hidden_layer_sizes": (50,),     "activation": "relu", "solver": "adam",  "alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 400},
    {"hidden_layer_sizes": (100,),    "activation": "relu", "solver": "adam",  "alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 400},
    {"hidden_layer_sizes": (150,),    "activation": "relu", "solver": "adam",  "alpha": 1e-4, "learning_rate_init": 5e-4, "max_iter": 500},
    {"hidden_layer_sizes": (100,50),  "activation": "relu", "solver": "adam",  "alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 600},

    {"hidden_layer_sizes": (100,),    "activation": "tanh", "solver": "adam",  "alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 600},
    {"hidden_layer_sizes": (100,),    "activation": "relu", "solver": "lbfgs", "alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 400},
    {"hidden_layer_sizes": (100,),    "activation": "relu", "solver": "adam",  "alpha": 1e-3, "learning_rate_init": 1e-3, "max_iter": 500},
]

results = []
best_model = None
best_f1 = -1
best_params = None

for i, params in enumerate(param_grid, start=1):
    mlp = MLPClassifier(
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        **params
    )

    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    row = {
        "run": i,
        "hidden_layer_sizes": str(params["hidden_layer_sizes"]),
        "activation": params["activation"],
        "solver": params["solver"],
        "alpha": params["alpha"],
        "lr_init": params["learning_rate_init"],
        "max_iter": params["max_iter"],
        "acc": acc,
        "prec_w": prec,
        "rec_w": rec,
        "f1_w": f1,
        "n_iter_": mlp.n_iter_
    }
    results.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_model = mlp
        best_params = params

results_df = pd.DataFrame(results).sort_values("f1_w", ascending=False)
print(" MLP hyperparameter comparison ")
print(results_df.to_string(index=False))
print("\nBest params:", best_params)
print("Best weighted F1:", best_f1)

best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Confusion Matrix - Best MLP (by weighted F1)")
plt.show()
