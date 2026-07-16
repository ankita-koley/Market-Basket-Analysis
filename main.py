import pandas as pd
df = pd.read_csv("data/groceries.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Shape
print("\nShape:")
print(df.shape)

# Columns
print("\nColumns:")
print(df.columns)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Dataset Info
print("\nInfo:")
print(df.info())


# Data Cleaning

print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# Check missing values again
print("\nMissing Values:")
print(df.isnull().sum())

# Data types
print("\nData Types:")
print(df.dtypes)


# Create Transaction List

transactions = df.groupby(
    ['Member_number', 'Date']
)['itemDescription'].apply(list).tolist()

print("\nTotal Transactions:", len(transactions))

print("\nFirst 5 Transactions:")
for t in transactions[:5]:
    print(t)


    # Transaction Encoder

from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()

te_array = te.fit(transactions).transform(transactions)

basket = pd.DataFrame(te_array, columns=te.columns_)

print("\nBasket Shape:")
print(basket.shape)

print("\nFirst 5 Rows of Basket:")
print(basket.head())


# Apriori Algorithm

from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(
    basket,
    min_support=0.003,
    use_colnames=True
)

print("Number of Frequent Itemsets:", len(frequent_itemsets))
print(frequent_itemsets.head(20))

# Association Rules

from mlxtend.frequent_patterns import association_rules

rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1
)

print("Total Rules:", len(rules))

print(rules[['antecedents',
             'consequents',
             'support',
             'confidence',
             'lift']].head(20))

import matplotlib.pyplot as plt

# Top 10 Selling Products
top_products = df['itemDescription'].value_counts().head(10)

plt.figure(figsize=(10,6))
top_products.plot(kind='bar', color='skyblue')

plt.title("Top 10 Selling Products")
plt.xlabel("Products")
plt.ylabel("Frequency")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

top_itemsets = frequent_itemsets.sort_values(
    by='support',
    ascending=False
).head(10)

plt.figure(figsize=(10,6))
plt.bar(
    top_itemsets['itemsets'].astype(str),
    top_itemsets['support']
)

plt.xticks(rotation=70)
plt.title("Top Frequent Itemsets")
plt.xlabel("Itemsets")
plt.ylabel("Support")

plt.tight_layout()
plt.show()

frequent_itemsets.to_csv("frequent_itemsets.csv", index=False)
rules.to_csv("association_rules.csv", index=False)