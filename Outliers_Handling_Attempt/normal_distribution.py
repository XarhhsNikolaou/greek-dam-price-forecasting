import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

base_file = r'C:\Users\harry\Desktop\Price Forecasting Project\ready_to_use.csv'
df = pd.read_csv(base_file)
data = df['mcp_eur_per_mwh']
# Υπολόγισε το μέσο όρο και την τυπική απόκλιση του δείγματός σου
mu = np.mean(data)
sigma = np.std(data)

# Σχεδίαση Histogram
plt.hist(data, bins=30, density=True, alpha=0.6, color='b')

# Σχεδίαση της κανονικής καμπύλης με τις δικές σου παραμέτρους
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, sigma) # Εδώ μπαίνουν οι τιμές σου
plt.plot(x, p, 'k', linewidth=2)
plt.title(f"Normal Distribution ($\mu={mu:.2f}, \sigma={sigma:.2f}$)")
plt.show()