from pathlib import Path as _Path
REPO = _Path(__file__).resolve().parents[2]  # repository root

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# 1. Φόρτωση
path = str(REPO / "Dataset_Creation" / "after_exports.csv")
df = pd.read_csv(path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Feature Engineering (Η εκδοχή που έδινε 17 MAE)
df['hour'] = df['timestamp'].dt.hour
df['month'] = df['timestamp'].dt.month
df['day_of_week'] = df['timestamp'].dt.dayofweek

df['rolling_mean_1day'] = df['mcp_eur_per_mwh'].shift(24).rolling(24).mean()
df['rolling_mean_7days'] = df['mcp_eur_per_mwh'].shift(24).rolling(24*7).mean()

df['lag_24h'] = df['mcp_eur_per_mwh'].shift(24)
df['lag_25h'] = df['mcp_eur_per_mwh'].shift(25)
df['lag_26h'] = df['mcp_eur_per_mwh'].shift(26)
df['lag_48h'] = df['mcp_eur_per_mwh'].shift(48)
df['lag_1week'] = df['mcp_eur_per_mwh'].shift(24*7)

df.dropna(inplace=True)

# 3. Split (Χωρίς leakage, όπως το είχαμε ορίσει αρχικά)
years_used = 3
train_size = years_used * 365 * 24 
test_horizon = 24

# Διαχωρισμός
df_train = df.iloc[-(train_size + test_horizon) : -2*test_horizon]
df_test = df.iloc[-2*test_horizon:-test_horizon]

X_train = df_train.drop(['timestamp', 'mcp_eur_per_mwh'], axis=1)
y_train = df_train['mcp_eur_per_mwh']

X_test = df_test.drop(['timestamp', 'mcp_eur_per_mwh'], axis=1)
y_test = df_test['mcp_eur_per_mwh']

# 1. Ορισμός του TimeSeriesSplit (n_splits=3 είναι ασφαλές για αρχή)
tscv = TimeSeriesSplit(n_splits=3)

# 2. Αρχικοποίηση μοντέλου
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

# 3. Ορισμός του grid των παραμέτρων (πρόσθεσε/αφαίρεσε ανάλογα με τον χρόνο)
param_grid = {
    'max_depth': [5],
    'learning_rate': [0.05],
    'n_estimators': [100],
    'subsample': [1.0]
}

# 4. GridSearch
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    verbose=1,
    n_jobs=-1 # Χρήση όλων των πυρήνων του επεξεργαστή
)

# 5. Fit στο training set
grid_search.fit(X_train, y_train)

# 6. Αποτελέσματα
print(f"Καλύτερες παράμετροι: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# 5. Πρόβλεψη
y_predicted = best_model.predict(X_test)

# Υπολογισμός MAE
mae = np.mean(np.abs(y_predicted - y_test.values))
print(f"MAE: {mae}")

import matplotlib.pyplot as plt

def plot_forecast(y_test, y_predicted):
    plt.figure(figsize=(14, 7))
    
    # Μετατροπή σε numpy αν είναι pandas series/dataframe
    actual = y_test.values if hasattr(y_test, 'values') else y_test
    predicted = y_predicted
    
    plt.plot(actual, label='Πραγματική Τιμή (Actual)', color='blue', marker='o', linewidth=2)
    plt.plot(predicted, label='Πρόβλεψη (Predicted)', color='red', linestyle='--', marker='x', linewidth=2)
    
    plt.title('Σύγκριση Πρόβλεψης vs Πραγματικής Τιμής (24h Horizon)', fontsize=14)
    plt.xlabel('Ώρα της ημέρας', fontsize=12)
    plt.ylabel('Τιμή (€/MWh)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Προσθήκη annotation για το MAE
    mae = np.mean(np.abs(actual - predicted))
    plt.text(0, min(min(actual), min(predicted)), f'MAE: {mae:.2f}', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Κλήση της συνάρτησης μετά το predict
plot_forecast(y_test, y_predicted)

xgb.plot_importance(best_model, importance_type='weight', max_num_features=10)
plt.show()

df_3finaldays = df[-30*24:]
plt.figure()
plt.plot(df_3finaldays['mcp_eur_per_mwh'])
plt.show()