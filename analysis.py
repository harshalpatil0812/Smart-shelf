import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

data = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\Mohit\Code_2\data.csv")

print("\nColumns in dataset:", data.columns)



if 'Date' not in data.columns:
    if 'date' in data.columns:
        data.rename(columns={'date': 'Date'}, inplace=True)

if 'Total Amount' in data.columns:
    data.rename(columns={'Total Amount': 'Weekly_Sales'}, inplace=True)

elif 'sales' in data.columns:
    data.rename(columns={'sales': 'Weekly_Sales'}, inplace=True)

# Check again
if 'Date' not in data.columns or 'Weekly_Sales' not in data.columns:
    print("❌ ERROR: Required columns not found!")
    print("Available columns:", data.columns)
    exit()

print(" Columns fixed")


data["Date"] = pd.to_datetime(data["Date"])


data = data.sort_values("Date")


data["Week"] = data["Date"].dt.to_period("W").dt.start_time

weekly = (
    data.groupby("Week")["Weekly_Sales"]
    .sum()
    .reset_index()
    .rename(columns={"Week": "Date"})
)

print("\nAggregated weekly data:")
print(weekly.head())


plt.figure(figsize=(8, 5))
plt.plot(weekly["Date"], weekly["Weekly_Sales"], marker="o")
plt.title("Sales Trend")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.grid()
plt.savefig(r"C:\Users\Dell\OneDrive\Desktop\Mohit\Code_2\graph.jpeg")
print("📊 Graph saved")


print("\n--- Basic Analysis ---")
print("Average Sales:", weekly["Weekly_Sales"].mean())
print("Max Sales:", weekly["Weekly_Sales"].max())
print("Min Sales:", weekly["Weekly_Sales"].min())


weekly["day"] = weekly["Date"].dt.day
weekly["month"] = weekly["Date"].dt.month
weekly["weekday"] = weekly["Date"].dt.weekday
weekly["is_weekend"] = weekly["weekday"].apply(lambda x: 1 if x >= 5 else 0)

weekly["lag_1"] = weekly["Weekly_Sales"].shift(1)
weekly["lag_2"] = weekly["Weekly_Sales"].shift(2)
weekly["rolling_mean"] = weekly["Weekly_Sales"].rolling(3).mean()

weekly = weekly.dropna()

print("\n--- Feature Data ---")
print(weekly.head())


X = weekly[
    ["day", "month", "weekday", "is_weekend", "lag_1", "lag_2", "rolling_mean"]
]
y = weekly["Weekly_Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

print("\n🔥 Training model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n--- Model Results ---")
print("Predictions:", predictions)
print("Actual:", y_test.values)
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))


pickle.dump(model, open(r"C:\Users\Dell\OneDrive\Desktop\Mohit\Code_2\model.pkl", "wb"))
print("💾 Model saved")


plt.figure(figsize=(8, 5))
plt.plot(y_test.index, y_test.values, label="Actual")
plt.plot(y_test.index, predictions, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted")
plt.grid()
plt.savefig(r"C:\Users\Dell\OneDrive\Desktop\Mohit\Code_2\prediction.jpeg")

print("📈 Prediction graph saved")
