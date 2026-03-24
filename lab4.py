import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import  MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r'D:\Homeworks\Data Mining\adult_.csv' )
df = pd.DataFrame(data)

print("First 5 rows:\n", df.head())
print("Shape:\n", df.shape)
print("Columns:\n", df.columns)
print("Data types:\n",df.dtypes)
print("Shape:\n", df.shape)

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()
df = df.replace('?', np.nan)
print("Missing values after replacement:\n", df.isnull().sum())

duplicates = df.duplicated().sum()
print("Duplicate rows before:", duplicates)
df = df.drop_duplicates()
duplicatesN = df.duplicated().sum()
print("Duplicate rows after:", duplicatesN)

X = df.drop("income", axis=1)
y = df["income"]
print("Features shape:", X.shape)
print("Target shape:", y.shape)

le = LabelEncoder()
yEnc = le.fit_transform(y)
print("\nTarget classes:", le.classes_)
print("First 10 encoded targets:", yEnc[:10])

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns
print("Numeric columns:\n", numeric_cols)
print("\nCategorical columns:\n", categorical_cols)

str_imp = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = str_imp.fit_transform(X[categorical_cols])
print("\nMissing values after categorical imputation:\n", X[categorical_cols].isnull().sum())

X_mean = X.copy()
mean_imp = SimpleImputer(strategy='mean')
X_mean[numeric_cols] = mean_imp.fit_transform(X_mean[numeric_cols])
print("\nMissing values after MEAN imputation:\n", X_mean[numeric_cols].isnull().sum())

X_median = X.copy()
median_imp = SimpleImputer(strategy='median')
X_median[numeric_cols] = median_imp.fit_transform(X_median[numeric_cols])
print("\nMissing values after MEDIAN imputation:\n", X_median[numeric_cols].isnull().sum())

X_encoded = pd.get_dummies(X, columns=categorical_cols)
print("\nShape before encoding:", X.shape)
print("Shape after encoding:", X_encoded.shape)

knn_imp = KNNImputer(n_neighbors=5)
X_knn = pd.DataFrame(knn_imp.fit_transform(X_encoded), columns=X_encoded.columns)
print("\nMissing values after KNN imputation:\n", X_knn.isnull().sum().sum())

def detect_outliers(series, col_name):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((series < lower) | (series > upper)).sum()
    print(col_name, "outliers:", outliers)

print("\nOutlier detection (IQR method):")
for col in numeric_cols:
    detect_outliers(X[col], col)

X_full = X_knn.copy()
scaler = StandardScaler()
X_full_zscore = pd.DataFrame(
    scaler.fit_transform(X_full),
    columns=X_full.columns
)
print("Full scaled data shape:", X_full_zscore.shape)
print(X_full_zscore.head())

scaler_minmax = MinMaxScaler()
X_full_minmax = pd.DataFrame(
    scaler_minmax.fit_transform(X_full),
    columns=X_full.columns
)
print("\nMinMax scaled data shape:", X_full_minmax.shape)
print(X_full_minmax.head())

features_to_plot = ['age', 'fnlwgt', 'education.num', 'hours.per.week']

for feature in features_to_plot:
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.hist(X_full[feature], bins=30, alpha=0.7)
    plt.title(f'Original: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.hist(X_full_minmax[feature], bins=30, alpha=0.7)
    plt.title(f'MinMax: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.hist(X_full_zscore[feature], bins=30, alpha=0.7)
    plt.title(f'Z-score: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

pca = PCA()
X_pca = pca.fit_transform(X_full_zscore)

explained_var = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_var)

n_components = np.argmax(cumulative_variance >= 0.95) + 1
print("\nComponents needed for 95% variance:", n_components)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)
plt.legend()
plt.show()

pca_final = PCA(n_components=n_components)
X_pca_final = pca_final.fit_transform(X_full_zscore)
print("Shape after PCA:", X_pca_final.shape)

X_train, X_test, y_train, y_test = train_test_split(
X_pca_final,
    yEnc,
    test_size=0.2,
    random_state=42,
    stratify=yEnc
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

configs = [
    {"hidden_layer_sizes": (10,), "activation": "relu"},
    {"hidden_layer_sizes": (20,), "activation": "relu"},
    {"hidden_layer_sizes": (50,), "activation": "relu"},

    {"hidden_layer_sizes": (10,), "activation": "tanh"},
    {"hidden_layer_sizes": (20,), "activation": "tanh"},
    {"hidden_layer_sizes": (50,), "activation": "tanh"},

    {"hidden_layer_sizes": (10,), "activation": "logistic"},
    {"hidden_layer_sizes": (20,), "activation": "logistic"},
    {"hidden_layer_sizes": (50,), "activation": "logistic"},

    {"hidden_layer_sizes": (10, 20), "activation": "relu"},
    {"hidden_layer_sizes": (10, 20), "activation": "tanh"},
    {"hidden_layer_sizes": (10, 20), "activation": "logistic"},
]

results = []

for i, cfg in enumerate(configs, start=1):
    mlp = MLPClassifier(
        hidden_layer_sizes=cfg["hidden_layer_sizes"],
        activation=cfg["activation"],
        max_iter=800,
        random_state=42
    )

    mlp.fit(X_train, y_train)

    y_train_pred = mlp.predict(X_train)
    y_test_pred = mlp.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    cm = confusion_matrix(y_test, y_test_pred)

    print("\n" + "=" * 60)
    print(f"Model {i}")
    print("Hidden Layers:", cfg["hidden_layer_sizes"])
    print("Activation:", cfg["activation"])
    print("Training Accuracy:", round(train_acc, 4))
    print("Testing Accuracy:", round(test_acc, 4))
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_test_pred,
        target_names=le.classes_
    ))

    results.append({
        "Model": i,
        "Hidden Layers": cfg["hidden_layer_sizes"],
        "Activation": cfg["activation"],
        "Training Accuracy": train_acc,
        "Testing Accuracy": test_acc,
        "Gap": train_acc - test_acc
    })

    plt.figure(figsize=(8, 5))
    plt.plot(mlp.loss_curve_)
    plt.title(f"Loss Curve - Model {i}: {cfg['hidden_layer_sizes']}, {cfg['activation']}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Testing Accuracy", ascending=False)

print("\nFinal comparison table:")
print(results_df[[
    "Model",
    "Hidden Layers",
    "Activation",
    "Training Accuracy",
    "Testing Accuracy",
    "Gap"
]])
