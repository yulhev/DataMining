import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

df = pd.read_csv("WineQT.csv")

plt.figure(figsize=(6,4))
sns.histplot(df['alcohol'], bins=20, kde=True, color='steelblue')
plt.title('Alcohol Distribution')
plt.xlabel('Alcohol')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(7,5))
sns.countplot(x='quality', data=df)
plt.xlabel('Wine quality')
plt.ylabel('Count')
plt.title('Distribution of Wine Quality')
plt.show()

df_parallel = df[['alcohol','pH','density','fixed acidity','quality']]
plt.figure(figsize=(10,6))
parallel_coordinates(df_parallel,'quality',
    color=['orangered','gold','darkolivegreen','deepskyblue','navy','deeppink']
)
plt.title("Parallel Coordinates Plot")
plt.xlabel("Attributes")
plt.ylabel("Values")
plt.show()

sns.jointplot(data=df, x='alcohol', y='quality')
plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.show()

sns.boxplot(y='alcohol', x='quality', data=df)
plt.title("Comparison distributions of alcohol content across quality levels")
plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.show()

sns.violinplot( y='alcohol', x='quality',  data=df )
plt.title("Comparison distributions of alcohol content across quality levels")
plt.xlabel("Alcohol")
plt.ylabel("Quality")
print(df.head())
plt.show()

colors=df['alcohol']
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
sc=ax.scatter(  df['alcohol'], df['pH'], df['density'], c=colors, cmap='viridis', alpha=0.8)
plt.colorbar(sc, ax=ax, label='alcohol,%', pad=0.15)
ax.set_xlabel('Alcohol')
ax.set_ylabel('pH')
ax.set_zlabel('Density')
plt.show()

plt.figure(figsize=(7,5))
sns.stripplot(x='quality',y='alcohol', data=df, jitter=True, alpha=0.6)
plt.title("Alcohol Distribution Across Wine Quality Levels")
plt.xlabel("Wine Quality")
plt.ylabel("Alcohol (%)")
plt.show()