import pandas as pd
import numpy as np
import mlxtend
from mlxtend.frequent_patterns import fpgrowth, association_rules


# -----------------------
# Load and Explore Data
# -----------------------

# Load the CSV file into a DataFrame
csv_path = '/Users/BTCJULIAN/Downloads/mma_mart.csv'
df = pd.read_csv(csv_path)


column_headings = df.columns
print(column_headings)

unique_departments = df['department'].unique()
print(unique_departments)

basic_stats = df.describe()
print(basic_stats)

# -------------------------------
# Frequency Analysis of Products & Cleaning Data
# -------------------------------

# Perform a frequency analysis to count the number of times each product shows up in the dataset
product_frequency = df['product_id'].value_counts().reset_index()
product_frequency.columns = ['product_id', 'frequency']

# Sort the products by their frequency in descending order
product_frequency_sorted = product_frequency.sort_values(by='frequency', ascending=False)

# Add product name, remove any duplicates
merged_df = pd.merge(product_frequency_sorted, df[['product_id', 'product_name', 'department']], on='product_id', how='left')
merged_df_unique = merged_df.drop_duplicates(subset=['product_id'])

# Filter the top 1000 products that have 'missing' or 'other' in the department column
ambiguous_department_products = merged_df_unique[merged_df_unique['department'].isin(['missing', 'other'])].head(10000)

# To display these products, you could use:
print(ambiguous_department_products)

# Adjust the department of "Organic Riced Cauliflower" to 'frozen'
merged_df_unique.loc[merged_df_unique['product_name'] == 'Organic Riced Cauliflower', 'department'] = 'frozen'
merged_df_unique.loc[merged_df_unique['product_name'] == 'Roasted Almond Butter', 'department'] = 'pantry'


# -------------------------
# Categorize the Products
# -------------------------

# Categroize the departments based on contraints. Classify each product as either frozen, refridgerated, or other.
frozen_products = ['frozen']
refrigerated_products = ['dairy eggs', 'produce', 'meat seafood', 'deli']


# Assign category to each product in the dataset
def product_label(department):
  if department in frozen_products:
      return 'frozen'
  elif department in refrigerated_products:
      return 'refrigerated'
  else:
    return 'other'


# Apply the labels to each product
df['product_category'] = df['department'].apply(product_label)

# Count and sort each product along with its new label
product_frequency_with_category = df.groupby(['product_id', 'product_category', 'product_name', 'department']).size().reset_index(name='frequency')

product_frequency_with_category_sorted = product_frequency_with_category.sort_values(by='frequency', ascending=False)

# Display top 20 results
product_frequency_with_category_sorted.head(20)


# -----------------------
# Selecting top 1000 products
# -----------------------


#Selecting the top 1000 products based on the constraints
top_frozen = product_frequency_with_category_sorted[product_frequency_with_category_sorted['product_category'] == 'frozen'].head(100)
top_refrigerated = product_frequency_with_category_sorted[product_frequency_with_category_sorted['product_category'] == 'refrigerated'].head(100)
top_other = product_frequency_with_category_sorted[product_frequency_with_category_sorted['product_category'] == 'other'].head(800)

# Take the top 100 frozen products, then the top 100 refridgerated products, then fill the remainder with 800 'other' products to get a total of 1000 products
top_1000_products = pd.concat([top_frozen, top_refrigerated, top_other])

# view the top 100
top_1000_products.head(10)

# Check which of the top 1000 products have 'missing' or 'other' in the department column
ambiguous_department_top_1000 = top_1000_products[top_1000_products['department'].isin(['missing', 'other'])]

# Count and display the number of such products
count_ambiguous_department_top_1000 = len(ambiguous_department_top_1000)
print("Count of products with 'missing' or 'other' department:", count_ambiguous_department_top_1000)
print(ambiguous_department_top_1000)


# -----------------------
# Count # of Orders with top 1000 products
# -----------------------

# Create a set of top 1000 product IDs for faster lookup
top_1000_product_ids = set(top_1000_products['product_id'])

# Group the original DataFrame by order IDs and aggregate the product IDs
grouped_transactions = df.groupby('order_id')['product_id'].apply(list).reset_index()

# Initialize the counter
count_orders_with_in_aisle_items = 0

# Iterate through the grouped transactions
for _, row in grouped_transactions.iterrows():
    products_in_this_order = set(row['product_id'])
    
    # Check for intersection between products in this order and the top 1000 products
    if products_in_this_order & top_1000_product_ids:
        count_orders_with_in_aisle_items += 1

# Print the count
print(f"Number of orders containing at least one top 1000 product: {count_orders_with_in_aisle_items}")

# -----------------------
# Pre-Calculation Setup
# -----------------------

# Calculate the total number of unique orders
total_orders = df['order_id'].nunique()
print(total_orders)


# -----------------------
# Calculate Metric 1
# -----------------------

# Calculate the percentage of orders that contain at least one item from the top 1000 products
percentage_orders_with_in_aisle_items = (count_orders_with_in_aisle_items / total_orders) * 100

# Print the calculated metric
print(f"Percentage of orders with at least one item from top 1000 products: {percentage_orders_with_in_aisle_items:.2f}%")


# -----------------------
# Calculate Metric 2
# -----------------------

# Calculate the total number of product entries in all orders
total_product_entries = df['product_id'].count()

# Calculate the total number of in-aisle product entries in all orders
total_in_aisle_items_in_orders = df[df['product_id'].isin(top_1000_product_ids)]['product_id'].count()


# Calculate the metrics
average_in_aisle_items_per_order = total_in_aisle_items_in_orders / total_orders
average_items_per_order = total_product_entries / total_orders
percentage_in_aisle_items_per_order = (average_in_aisle_items_per_order / average_items_per_order) * 100

# Print the calculated metrics
print(f"Average number of in-aisle items per order: {average_in_aisle_items_per_order:.2f}")
print(f"Average number of total items per order: {average_items_per_order:.2f}")
print(f"Percentage of in-aisle items per order: {percentage_in_aisle_items_per_order:.2f}%")




# -----------------------
# Finding Substitution Products
# -----------------------



# -----------------------
# FP-Growth Method
# -----------------------

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from mlxtend.frequent_patterns import fpgrowth, association_rules


# Filter out transactions to only include top 1000 products
df_filtered = df[df['product_id'].isin(top_1000_product_ids)]

# Group by order_id and list all products
grouped_df_filtered = df_filtered.groupby('order_id')['product_id'].apply(list).reset_index(name='products')

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Fit and transform the 'products' column to one-hot encode
one_hot_mlb = mlb.fit_transform(grouped_df_filtered['products'])

# Convert to DataFrame
one_hot_df = pd.DataFrame(one_hot_mlb, columns=mlb.classes_)

# Run FP-growth algorithm
min_support = 0.005  # Lowered minimum support threshold
frequent_itemsets_filtered = fpgrowth(one_hot_df, min_support=min_support, use_colnames=True)

# Generate association rules
rules_filtered = association_rules(frequent_itemsets_filtered, metric="lift", min_threshold=1)

# Filter rules to get only those where the antecedent has only one item (i.e., potential substitutes)
filtered_rules_filtered = rules_filtered[rules_filtered['antecedents'].apply(lambda x: len(x) == 1)]

# Display the first few rules to confirm they look correct
print(filtered_rules_filtered)



# -----------------------
# Apriori Method
# -----------------------

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Step 1: Sample 10% of the data
sampled_df = df.sample(frac=0.3, random_state=1)

# Step 2: One-Hot Encoding on the sampled data
oht_df_sample = pd.get_dummies(sampled_df['product_id'])

# Step 3: Run Apriori algorithm
frequent_itemsets = apriori(oht_df_sample, min_support=0.01, use_colnames=True)

# Step 4: Generate association rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

# Step 5: Filter the rules
filtered_rules = rules[(rules['lift'] >= 1.2) & (rules['confidence'] >= 0.7)]

# Step 6: Create a list of substitutes for top 1000 products
substitute_list = []
for product in top_1000_product_ids:
    substitutes = filtered_rules[filtered_rules['antecedents'] == {product}]['consequents'].tolist()
    substitute_list.extend([list(x)[0] for x in substitutes][:5])

# Step 7: Create extended_top_products list
extended_top_products = list(set(top_1000_product_ids).union(set(substitute_list)))
print("Length of extended_top_products:", len(extended_top_products))

# Step 8: Rerun the metrics
count_orders_with_extended_top_products = df[df['product_id'].isin(extended_top_products)]['order_id'].nunique()
percentage_orders_with_extended_top_products = (count_orders_with_extended_top_products / total_orders) * 100

total_extended_top_products_in_orders = df[df['product_id'].isin(extended_top_products)]['product_id'].count()
average_extended_top_products_per_order = total_extended_top_products_in_orders / total_orders
percentage_extended_top_products_per_order = (average_extended_top_products_per_order / average_items_per_order) * 100

# Step 9: Print the metrics
print(f"Percentage of orders with at least one item from extended top products: {percentage_orders_with_extended_top_products:.2f}%")
print(f"Average number of extended top products per order: {average_extended_top_products_per_order:.2f}")
print(f"Percentage of extended top products per order: {percentage_extended_top_products_per_order:.2f}%")


# -----------------------
# Item-Item Collaborative Filtering
# -----------------------



import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random

# Assume df is your original dataframe.
# Replace this placeholder with your actual data.

# Step 1: Sample the dataset
sample_fraction = 0.1  # Adjust this based on how much of your data you want to sample
sampled_df = df.sample(frac=sample_fraction, random_state=42)

# Step 2: Create the user-item matrix using the sampled dataset
user_item_matrix = pd.pivot_table(sampled_df, index='order_id', columns='product_id', values='product_name', 
                                  aggfunc='count', fill_value=0)

# Step 3: Calculate item-item cosine similarity
item_similarity = cosine_similarity(user_item_matrix.T)

# Convert the results into a DataFrame for easier interpretation
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def find_top_n_substitutes_with_names(product_id, n=5):
    # Get the similarity values for the given product
    similarities = item_similarity_df.loc[product_id]
    # Sort them in descending order
    sorted_similarities = similarities.sort_values(ascending=False)
    # Get the top N most similar products (excluding the product itself)
    top_n = sorted_similarities.iloc[1:n+1]

    # Fetch product names for the top N products
    product_name_mapping = sampled_df[['product_id', 'product_name']].drop_duplicates().set_index('product_id')
    top_n_names = product_name_mapping.loc[top_n.index]

    # Create a DataFrame to show both product_id and product_name
    result = pd.DataFrame({
        'product_id': top_n.index,
        'similarity_score': top_n.values,
        'product_name': top_n_names['product_name'].values
    })

    return result

# Replace 34 with the product_id for which you want to find substitutes
top_5_substitutes = find_top_n_substitutes_with_names(24852, n=5)
print(top_5_substitutes)

# -----------------------
# Pre-Calculation Setup with Substitutes 
# -----------------------


# Step 1: Generate a list of substitute products for each of the top 1000 products
substitute_list = []
for product in top_1000_product_ids:
    top_5_substitutes = find_top_n_substitutes_with_names(product, n=5)
    substitute_list.extend(top_5_substitutes['product_id'].tolist())

# Combine the original top 1000 products with their substitutes
extended_top_products = list(set(top_1000_product_ids).union(set(substitute_list)))



# Step 3: Re-calculate the metrics based on this new list

# Metric 1
count_orders_with_extended_top_products = df[df['product_id'].isin(extended_top_products)]['order_id'].nunique()
percentage_orders_with_extended_top_products = (count_orders_with_extended_top_products / total_orders) * 100
print(f"Percentage of orders with at least one item from extended top products: {percentage_orders_with_extended_top_products:.2f}%")

# Metric 2
total_extended_top_products_in_orders = df[df['product_id'].isin(extended_top_products)]['product_id'].count()
average_extended_top_products_per_order = total_extended_top_products_in_orders / total_orders
percentage_extended_top_products_per_order = (average_extended_top_products_per_order / average_items_per_order) * 100

print(f"Average number of extended top products per order: {average_extended_top_products_per_order:.2f}")
print(f"Percentage of extended top products per order: {percentage_extended_top_products_per_order:.2f}%")
