import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# 1. Load dataset diamonds
df = sns.load_dataset('diamonds')
real_prices = df['price'].values

# 2. Lakukan parameter fitting Distribusi Lognormal teoretis
shape, loc, scale = stats.lognorm.fit(real_prices, floc=300)

# 3. Definisikan persentil fokus ekor kanan (90% s.d. 99.9%)
percentiles = np.linspace(90, 99.9, 100)

# 4. Hitung kuantil empiris (data riil) vs kuantil teoretis (model lognormal)
empirical_quantiles = np.percentile(real_prices, percentiles)
theoretical_quantiles = stats.lognorm.ppf(percentiles / 100.0, s=shape, loc=loc, scale=scale)

# 5. Plotting
plt.figure(figsize=(8, 6), dpi=150)
plt.scatter(theoretical_quantiles, empirical_quantiles, color='#1f77b4', alpha=0.7, edgecolors='none', s=40, label='Kuantil Harga Berlian')

# Menggambar garis ideal y = x
max_val = max(np.max(theoretical_quantiles), np.max(empirical_quantiles))
min_val = min(np.min(theoretical_quantiles), np.min(empirical_quantiles))
plt.plot([min_val, max_val], [min_val, max_val], color='#d62728', linestyle='--', linewidth=2, label='Garis Ideal (y=x)')

# Kustomisasi Grafik
plt.title("Tail Q-Q Plot (Persentil 90% - 99.9%)\nEvaluasi Risiko Ekor Distribusi Lognormal", fontsize=12, fontweight='bold', pad=10)
plt.xlabel("Kuantil Teoretis Lognormal ($)", fontsize=11)
plt.ylabel("Kuantil Empiris Data Riil ($)", fontsize=11)
plt.legend(loc='upper left', frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('qq_plot_ekor_diamonds.png')
plt.show()