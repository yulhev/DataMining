import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv("booksToScrapeF.csv")

print("\nDataset info:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values per column:")
print(df.isna().sum())

plt.boxplot(df["price"])
plt.title("Boxplot of book prices")
plt.ylabel("Price")
plt.show()

genre_counts = df["category"].value_counts().sort_values()

plt.figure(figsize=(10, 10))
plt.barh(genre_counts.index, genre_counts.values, height=0.5)
plt.title("Number of books per each category")
plt.xlabel("count")
plt.ylabel("category")
plt.tight_layout()
plt.show()

df["rating_word"] = df["rating"].str.extract(r"(One|Two|Three|Four|Five)")
rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}
df["rating_num"] = df["rating_word"].map(rating_map)

plt.figure(figsize=(8,6))
plt.scatter(df["rating_num"], df["price"])
plt.title("Scatter Plot: Rating vs Price")
plt.xlabel("Rating (1–5)")
plt.ylabel("Price")
plt.xticks([1, 2, 3, 4, 5])
plt.grid(True)
plt.show()
