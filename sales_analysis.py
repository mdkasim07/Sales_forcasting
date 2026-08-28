# ============================================
# SALES DATA ANALYSIS & FORECASTING
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. LOAD DATA
# --------------------------------------------

df = pd.read_csv("data/sales_data.csv")

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("\nFirst 5 records:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nColumn information:")
print(df.info())

# --------------------------------------------
# 2. DATA CLEANING
# --------------------------------------------

print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)

# Convert date column
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicate records
duplicates = df.duplicated().sum()
print("\nDuplicate records:", duplicates)

df = df.drop_duplicates()

# Check invalid quantities
df = df[df["Quantity"] > 0]

# Check invalid sales
df = df[df["Sales_INR"] >= 0]

print("\nCleaned dataset shape:", df.shape)

# --------------------------------------------
# 3. BASIC STATISTICS
# --------------------------------------------

print("\n" + "=" * 50)
print("DESCRIPTIVE STATISTICS")
print("=" * 50)

print(df[[
    "Quantity",
    "Unit_Price_INR",
    "Discount",
    "Sales_INR"
]].describe())

# --------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# --------------------------------------------

print("\n" + "=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# Total sales
total_sales = df["Sales_INR"].sum()

# Total quantity
total_quantity = df["Quantity"].sum()

# Average order value
average_sales = df["Sales_INR"].mean()

print(f"\nTotal Sales: ₹{total_sales:,.2f}")
print(f"Total Quantity Sold: {total_quantity:,}")
print(f"Average Order Value: ₹{average_sales:,.2f}")

# --------------------------------------------
# 5. SALES BY REGION
# --------------------------------------------

region_sales = (
    df.groupby("Region")["Sales_INR"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by Region:")
print(region_sales)

# --------------------------------------------
# 6. SALES BY PRODUCT
# --------------------------------------------

product_sales = (
    df.groupby("Product")["Sales_INR"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by Product:")
print(product_sales)

# --------------------------------------------
# 7. SALES BY CATEGORY
# --------------------------------------------

category_sales = (
    df.groupby("Category")["Sales_INR"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by Category:")
print(category_sales)

# --------------------------------------------
# 8. MONTHLY SALES
# --------------------------------------------

df["Month"] = df["Order_Date"].dt.to_period("M")

monthly_sales = (
    df.groupby("Month")["Sales_INR"]
    .sum()
)

print("\nMonthly Sales:")
print(monthly_sales)

# --------------------------------------------
# 9. VISUALIZATION - MONTHLY SALES
# --------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales.index.astype(str),
    monthly_sales.values,
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales (INR)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("visualizations/monthly_sales.png")
plt.show()

# --------------------------------------------
# 10. VISUALIZATION - REGION
# --------------------------------------------

plt.figure(figsize=(8, 5))

region_sales.plot(kind="bar")

plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales (INR)")
plt.tight_layout()

plt.savefig("visualizations/sales_by_region.png")
plt.show()

# --------------------------------------------
# 11. VISUALIZATION - PRODUCT
# --------------------------------------------

plt.figure(figsize=(10, 6))

product_sales.head(10).plot(kind="bar")

plt.title("Top Products by Sales")
plt.xlabel("Product")
plt.ylabel("Sales (INR)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("visualizations/top_products.png")
plt.show()

# --------------------------------------------
# 12. SALES FORECASTING
# --------------------------------------------

# Simple moving-average forecast

monthly_sales_df = monthly_sales.reset_index()
monthly_sales_df.columns = ["Month", "Sales"]

monthly_sales_df["Moving_Average"] = (
    monthly_sales_df["Sales"]
    .rolling(window=3)
    .mean()
)

print("\n" + "=" * 50)
print("SALES FORECAST")
print("=" * 50)

print(monthly_sales_df.tail())

# Forecast next month using last 3 months average
forecast = monthly_sales_df["Sales"].tail(3).mean()

print(
    f"\nPredicted Sales for Next Month: "
    f"₹{forecast:,.2f}"
)

print("\nAnalysis completed successfully!")
