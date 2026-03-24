import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import  MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, \
    f1_score

df = pd.read_csv(r'D:\Homeworks\Data Mining\cancer.csv' )

print("First 5 rows:\n", df.head())
print("Shape:\n", df.shape)
print("Columns:\n", df.columns)
print("Data types:\n",df.dtypes)
print("Shape:\n", df.shape)

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()
df = df.replace('?', np.nan)

print("Missing values after replacement:\n", df.isnull().sum())
df = df.drop(columns=['Unnamed: 32'], errors='ignore')
print("Missing values after handling:\n", df.isnull().sum())

print("Duplicate rows before:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicate rows after:", df.duplicated().sum())

print(df.describe())

cols = df.columns   # includes 'diagnosis'
mid = len(cols) // 2
cols_1 = cols[:mid]
cols_2 = cols[mid:]

df[cols_1].hist(figsize=(14, 10), bins=20)
plt.suptitle("Histograms (Part 1)", y=1.02)
plt.tight_layout()
plt.show()

df[cols_2].hist(figsize=(14, 10), bins=20)
plt.suptitle("Histograms (Part 2)", y=1.02)
plt.tight_layout()
plt.show()

co_mtx = df.corr(numeric_only=True)
print(co_mtx)
plt.figure(figsize=(16, 12))
sns.heatmap(co_mtx, cmap="YlGnBu")
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()

X = df.drop(['diagnosis', 'id'], axis=1)
y = df["diagnosis"]

le = LabelEncoder()
y = le.fit_transform(y)

print("Missing values in X before imputation:\n", X.isnull().sum())

minmax_scaler = MinMaxScaler()
X_minmax = pd.DataFrame(
    minmax_scaler.fit_transform(X),
    columns=X.columns
)
print("\nMin-Max scaled data (first 5 rows):")
print(X_minmax.head())

zscore = StandardScaler()
X_zscore = pd.DataFrame(
    zscore.fit_transform(X),
    columns=X.columns
)

colS = X.columns
midS = len(colS) // 2

colS_1 = colS[:midS]
colS_2 = colS[midS:]

X_minmax[colS_1].hist(figsize=(14, 10), bins=20)
plt.suptitle("Min-Max Scaled Data (Part 1)", y=1.02)
plt.tight_layout()
plt.show()

X_minmax[colS_2].hist(figsize=(14, 10), bins=20)
plt.suptitle("Min-Max Scaled Data (Part 2)", y=1.02)
plt.tight_layout()
plt.show()

X_zscore[colS_1].hist(figsize=(14, 10), bins=20)
plt.suptitle("Z-score Scaled Data (Part 1)", y=1.02)
plt.tight_layout()
plt.show()

X_zscore[colS_2].hist(figsize=(14, 10), bins=20)
plt.suptitle("Z-score Scaled Data (Part 2)", y=1.02)
plt.tight_layout()
plt.show()

X_temp, X_test, y_temp, y_test = train_test_split(
    X_zscore, y, test_size=0.2, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42
)

print("\nTraining set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)

kFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_model = LogisticRegression(max_iter=5000, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

models = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model
}
results = []

for name, model in models.items():
    print(f"MODEL: {name}\n")

    cv_scores = cross_val_score(model, X_train, y_train, cv=kFold, scoring='accuracy')
    print("CV scores:", cv_scores)
    print("Mean CV accuracy:", cv_scores.mean())

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    print("\nValidation metrics:")
    print("Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Precision:", precision_score(y_val, y_val_pred))
    print("Recall:", recall_score(y_val, y_val_pred))
    print("F1-score:", f1_score(y_val, y_val_pred))

    y_test_pred = model.predict(X_test)
    print("\nTest metrics:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Precision:", precision_score(y_test, y_test_pred))
    print("Recall:", recall_score(y_test, y_test_pred))
    print("F1-score:", f1_score(y_test, y_test_pred))
    print("Classification Report:\n", classification_report(y_test, y_test_pred))

    results.append({
        "Model": name,
        "CV Accuracy": cv_scores.mean(),
        "Validation Accuracy": accuracy_score(y_val, y_val_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred),
        "Recall": recall_score(y_test, y_test_pred),
        "F1-score": f1_score(y_test, y_test_pred)
    })

comparison_table = pd.DataFrame(results)

print("\n Model Comparison ")
print(comparison_table)

for name, model in models.items():
    y_test_pred = model.predict(X_test)

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()