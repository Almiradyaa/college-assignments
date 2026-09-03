import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

print("Mengambil data berlian...")
df = sns.load_dataset('diamonds')
prices = df['price'].values

loc_shift = 300
shifted_prices = prices - loc_shift
log_prices = np.log(shifted_prices[shifted_prices > 0]) 

plt.figure(figsize=(8, 6), dpi=150)
stats.probplot(log_prices, dist="norm", plot=plt)

plt.title("Lognormal Q-Q Plot: Harga Berlian vs Teoretis", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Kuantil Teoretis (Distribusi Normal)", fontsize=12)
plt.ylabel("Kuantil Empiris (Log-Harga Berlian)", fontsize=12)

ax = plt.gca()
lines = ax.get_lines()
if lines:
	for line in lines:
		line.set_color('#1f77b4')
		line.set_alpha(0.5)
	if len(lines) > 1:
		lines[1].set_color('#d62728')
		lines[1].set_linewidth(2)

plt.tight_layout()
plt.savefig('qq_plot_standar.png')
plt.show()