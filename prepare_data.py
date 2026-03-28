import pandas as pd
import os

print("🚀 Loading dataset...")

# Load dataset
data = pd.read_csv(
    r"C:\Users\Dell\OneDrive\Desktop\Mohit\Code_2\data.csv",
    low_memory=False
)

print("✅ Data loaded")
print("Columns:", data.columns)
print("Rows:", len(data))

# ===============================
# BASIC CLEANING
# ===============================

# Convert Date column
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date'])

# Rename column
if 'Total Amount' in data.columns:
    data.rename(columns={'Total Amount': 'Sales'}, inplace=True)

# ===============================
# GROUP DATA (VERY IMPORTANT)
# ===============================
# Aggregate daily sales
data = data.groupby(['Date'])['Sales'].sum().reset_index()

# Sort data
data = data.sort_values(by=['Date'])

print("✅ Date converted & data sorted")

# ===============================
# FEATURE ENGINEERING
# ===============================

print("⚙️ Creating features...")

# Date features
data['day'] = data['Date'].dt.day
data['month'] = data['Date'].dt.month
data['weekday'] = data['Date'].dt.weekday

# Weekend flag
data['is_weekend'] = data['weekday'].apply(lambda x: 1 if x >= 5 else 0)

# Lag features
data['lag_1'] = data['Sales'].shift(1)
data['lag_2'] = data['Sales'].shift(2)

# Rolling mean
data['rolling_mean'] = data['Sales'].rolling(3).mean()

print("✅ Features created")

# ===============================
# REMOVE NULL VALUES
# ===============================

data = data.dropna()

print("✅ Null values removed")
print("Final rows:", len(data))

# ===============================
# SAVE CLEAN DATA
# ===============================

output_path = os.path.join(os.path.dirname(__file__), "clean_data.csv")
data.to_csv(output_path, index=False)

print("💾 Clean data saved at:", output_path)

# Preview
print("\nSample Data:")
print(data.head())