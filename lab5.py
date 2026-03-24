import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import  MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv(r'D:\Homeworks\Data Mining\cancer.csv' )

# print("First 5 rows:\n", df.head())
# print("Shape:\n", df.shape)
# print("Columns:\n", df.columns)
# print("Data types:\n",df.dtypes)
# print("Shape:\n", df.shape)

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()
df = df.replace('?', np.nan)

#print("Missing values after replacement:\n", df.isnull().sum())
df = df.drop(columns=['Unnamed: 32'], errors='ignore')
#print("Missing values after handling:\n", df.isnull().sum())

print("Duplicate rows before:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicate rows after:", df.duplicated().sum())

print(df.describe())

cols = df.columns   # includes 'diagnosis'

mid = len(cols) // 2

cols_1 = cols[:mid]
cols_2 = cols[mid:]

# first plot
df[cols_1].hist(figsize=(14, 10), bins=20)
plt.suptitle("Histograms (Part 1)", y=1.02)
plt.tight_layout()
plt.show()

# second plot
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

#numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# X_mean = df.copy()
# mean_imp = SimpleImputer(strategy='mean')
# X_mean[numeric_cols] = mean_imp.fit_transform(X_mean[numeric_cols])
# print("\nMissing values after MEAN imputation:\n", X_mean[numeric_cols].isnull().sum())
# #
# X_median = df.copy()
# median_imp = SimpleImputer(strategy='median')
# X_median[numeric_cols] = median_imp.fit_transform(X_median[numeric_cols])
# print("\nMissing values after MEDIAN imputation:\n", X_median[numeric_cols].isnull().sum())
#
# scaler = StandardScaler()
# X_zscore = pd.DataFrame(
#     scaler.fit_transform(df),
#     columns=df.columns
# )
# print("Full scaled data shape:", X_zscore.shape)
# print(X_zscore.head())
# #
# scaler_minmax = MinMaxScaler()
# X_minmax = pd.DataFrame(
#     scaler_minmax.fit_transform(df),
#     columns=df.columns
# )
# print("\nMinMax scaled data shape:", X_minmax.shape)
# print(X_minmax.head())