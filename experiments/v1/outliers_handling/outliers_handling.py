from pathlib import Path as _Path
REPO = _Path(__file__).resolve().parents[3]  # repository root

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 1. Φόρτωση δεδομένων (Βάλε το path για το csv με τα 562 outliers)
path = str(REPO / "outliers.csv") # Άλλαξε το όνομα αν χρειάζεται
df = pd.read_csv(path)
print(df)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Feature Engineering (Μόνο Ώρα, Ημέρα, Μήνας - Κυκλικά)
df['hour'] = df['timestamp'].dt.hour

df['day_of_week'] = df['timestamp'].dt.dayofweek

df['month'] = df['timestamp'].dt.month

# Επιλογή features (ΧΩΡΙΣ lags)
features = ['hour', 'day_of_week', 'month']
# ΠΡΟΣΟΧΗ: Αν έχεις άλλα εξωγενή features στο csv (π.χ. Load, RES), πρόσθεσέ τα στη λίστα 'features'

X = df[features]
y = df['mcp_eur_per_mwh']

# 3. Τυχαίο Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Εκπαίδευση Μοντέλου
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Πρόβλεψη στο Test Set
y_pred = model.predict(X_test)

# Υπολογισμός MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE στα Outliers (Test Set): {mae:.2f} €/MWh\n")

# 5. Feature Importance (Βάσει Gain)
importances = model.feature_importances_
df_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
df_importances = df_importances.sort_values(by='Importance', ascending=False)

print("--- Feature Importances ---")
print(df_importances)

# Οπτικοποίηση
xgb.plot_importance(model, importance_type='gain', max_num_features=10, title='Σημαντικότητα Χαρακτηριστικών στα Outliers')
plt.show()

# --- 6. Baseline / "Χαζή" Εκτίμηση (Μέση Τιμή) ---
# Η χαζή εκτίμηση είναι απλά η μέση τιμή των outliers στο train set
dummy_prediction = y_train.mean()

# Δημιουργούμε μια λίστα με αυτή τη σταθερή τιμή για όλο το test set
y_pred_dummy = [dummy_prediction] * len(y_test)

# Υπολογισμός Baseline MAE
mae_dummy = mean_absolute_error(y_test, y_pred_dummy)

print(f"MAE 'Χαζής' Εκτίμησης (Baseline): {mae_dummy:.2f} €/MWh")
print(f"Βελτίωση XGBoost έναντι Baseline: {mae_dummy - mae:.2f} €/MWh")