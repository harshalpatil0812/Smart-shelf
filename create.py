import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# LOAD DATA
# ===============================
data = pd.read_csv(
    r"C:\Users\Dell\OneDrive\Desktop\Mohit\Code_2\data.csv",
    low_memory=False
)

print("Columns:", data.columns)

# ===============================
# FIX COLUMN NAME (IMPORTANT)
# ===============================
if 'Total Amount' in data.columns:
    data.rename(columns={'Total Amount': 'Sales'}, inplace=True)

# ===============================
# CONVERT DATE
# ===============================
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date'])

# ===============================
# GROUP BY DATE (VERY IMPORTANT)
# ===============================
data = data.groupby('Date')['Sales'].sum().reset_index()

# Sort data
data = data.sort_values('Date')

# ===============================
# VISUALIZATION
# ===============================
plt.figure(figsize=(8,5))
plt.plot(data['Date'], data['Sales'], marker='o')

plt.title("Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid()

plt.savefig("graph.png")
print("Graph saved ✅")

# ===============================
# BASIC ANALYSIS
# ===============================
print("\n--- Basic Analysis ---")
print("Average Sales:", data['Sales'].mean())
print("Max Sales:", data['Sales'].max())
print("Min Sales:", data['Sales'].min())

# ===============================
# FEATURE ENGINEERING
# ===============================
data['day'] = data['Date'].dt.day
data['month'] = data['Date'].dt.month

# Lag features
data['lag_1'] = data['Sales'].shift(1)
data['lag_2'] = data['Sales'].shift(2)

# Rolling mean
data['rolling_mean'] = data['Sales'].rolling(3).mean()

# Drop null values
data = data.dropna()

print("\n--- Feature Engineered Data ---")
print(data.head())

# ===============================
# MACHINE LEARNING MODEL
# ===============================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

X = data[['day','month','lag_1','lag_2','rolling_mean']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ===============================
# RESULTS
# ===============================
print("\n--- Model Results ---")
print("MAE:", mean_absolute_error(y_test, predictions))