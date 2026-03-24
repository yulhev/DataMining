import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import  MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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