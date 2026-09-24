from pathlib import Path as _Path
REPO = _Path(__file__).resolve().parents[3]  # repository root

import pandas as pd

# Ορισμός των paths
input_path = str(REPO / "ready_to_use.csv")
output_path = str(REPO / "outliers.csv")

# Διάβασμα του αρχικού dataset
df = pd.read_csv(input_path)

# Φιλτράρισμα για τιμές μεγαλύτερες του 250
df_outliers = df[df['mcp_eur_per_mwh'] > 250]

# Αποθήκευση στο νέο αρχείο (χωρίς το index)
df_outliers.to_csv(output_path, index=False)

print(f"Εντοπίστηκαν {len(df_outliers)} outliers. Αποθηκεύτηκαν στο: {output_path}")