import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

path = r'C:\Users\harry\Desktop\Projects\Price Forecasting Project\Dataset_Creation\processed\final_dataset.csv'
df = pd.read_csv(path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
# ΔΙΟΡΘΩΣΗ: το 'hour' (και τα 'month', 'day_of_week', 'is_weekend') υπάρχουν
# ήδη έτοιμα στο final_dataset.csv (στάδιο 4 του build_dataset.py) — δεν
# χρειάζεται να υπολογιστούν ξανά εδώ.

# --- Feature Engineering ---
df['rolling_mean_1day'] = df['mcp_eur_per_mwh'].shift(24).rolling(24).mean()
df['rolling_mean_7days'] = df['mcp_eur_per_mwh'].shift(24).rolling(24*7).mean()
df['lag_24h'] = df['mcp_eur_per_mwh'].shift(24)
df['lag_25h'] = df['mcp_eur_per_mwh'].shift(25)
df['lag_48'] = df['mcp_eur_per_mwh'].shift(48)
df['lag_168h'] = df['mcp_eur_per_mwh'].shift(168)
df.dropna(inplace=True)

# --- Split βασισμένο σε μήνες ---
months_used = 24  # Παράμετρος για ευελιξία
test_horizon = 24 # 24 ώρες
# Μετατροπή μηνών σε ώρες (κατά προσέγγιση)
train_size = months_used * 30 * 24 

index = -(train_size + test_horizon)
days_back = 0
offset = days_back * 24

df_train = df.iloc[index : -(test_horizon + offset)] if offset > 0 else df.iloc[index:-test_horizon]
df_test = df.iloc[-(test_horizon + offset) : -offset] if offset > 0 else df.iloc[-test_horizon:]

X_train = df_train.drop(['timestamp', 'mcp_eur_per_mwh', 'net_out'], axis=1)
y_train = df_train['mcp_eur_per_mwh']
X_test = df_test.drop(['timestamp', 'mcp_eur_per_mwh', 'net_out'], axis=1)
y_test = df_test['mcp_eur_per_mwh']

# --- GridSearch ---
tscv = TimeSeriesSplit(n_splits=3)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
param_grid = {
    'max_depth': [5],
    'learning_rate': [0.05],
    'n_estimators': [150],
    'subsample': [1.0]
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=tscv, 
                           scoring='neg_mean_absolute_error', verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Καλύτερες παράμετροι: {grid_search.best_params_}")

y_predicted = best_model.predict(X_test)

mae = abs(y_predicted - y_test.values).mean()

print(f"MAE: {mae}")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Πραγματική Τιμή (Actual)', color='blue', marker='o')
plt.plot(y_predicted, label='Πρόβλεψη (Predicted)', color='red', linestyle='--', marker='x')

plt.title('Σύγκριση Πραγματικής Τιμής vs Πρόβλεψης (Τελευταίο 24ωρο)')
plt.xlabel('Ώρα (0-23)')
plt.ylabel('Τιμή (€/MWh)')
plt.legend()
plt.grid(True)
plt.show()