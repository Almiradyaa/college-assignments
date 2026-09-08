import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.style.use('seaborn-v0_8-whitegrid')
CHART_DPI = 150  

shape = 0.8925       
loc = 300.0         
scale = 2292.08      

x = np.linspace(301, 18823, 1000)

pdf_values = stats.lognorm.pdf(x, s=shape, loc=loc, scale=scale)

cdf_values = stats.lognorm.cdf(x, s=shape, loc=loc, scale=scale)

median_val = stats.lognorm.ppf(0.50, s=shape, loc=loc, scale=scale)

p90_val = stats.lognorm.ppf(0.90, s=shape, loc=loc, scale=scale)

print(f"Hasil Perhitungan Matematis Teoretis:")
print(f" - Median Teoretis (CDF = 0.50): ${median_val:,.2f}")
print(f" - Persentil 90    (CDF = 0.90): ${p90_val:,.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=CHART_DPI)
fig.suptitle("Analisis Teoretis PDF & CDF Harga Berlian (Model Lognormal)", 
fontsize=15, fontweight='bold', y=1.02)

ax1.plot(x, pdf_values, color='#1f77b4', linewidth=2.5, label='Kurva Kerapatan Peluang (PDF)')
ax1.fill_between(x, pdf_values, color='#1f77b4', alpha=0.15) 

ax1.set_title("Probability Density Function (PDF)\nKerapatan Peluang Harga Jual", fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel("Harga Jual Berlian ($ USD)", fontsize=10)
ax1.set_ylabel("Kepadatan (Density)", fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend(frameon=True, loc='upper right')

ax2.plot(x, cdf_values, color='#d62728', linewidth=2.5, label='Probabilitas Kumulatif (CDF)')
ax2.fill_between(x, cdf_values, color='#d62728', alpha=0.08) 

ax2.scatter(median_val, 0.50, color='green', s=85, zorder=5, 
            label=f'Median Teoretis: ${median_val:,.2f}')
ax2.axhline(0.50, color='green', linestyle=':', alpha=0.6)
ax2.axvline(median_val, color='green', linestyle=':', alpha=0.6)

ax2.scatter(p90_val, 0.90, color='orange', s=85, zorder=5, 
            label=f'Persentil 90 (P90): ${p90_val:,.2f}')
ax2.axhline(0.90, color='orange', linestyle=':', alpha=0.6)
ax2.axvline(p90_val, color='orange', linestyle=':', alpha=0.6)

ax2.set_title("Cumulative Distribution Function (CDF)\nKurva Probabilitas Kumulatif S-Curve", fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel("Harga Jual Berlian ($ USD)", fontsize=10)
ax2.set_ylabel("Probabilitas Kumulatif (P <= x)", fontsize=10)
ax2.set_ylim(-0.02, 1.05) 
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.legend(frameon=True, loc='lower right')

plt.tight_layout()
output_filename = 'pdf_cdf_diamonds.png'
plt.savefig(output_filename, bbox_inches='tight')
print(f"\nVisualisasi berhasil dibuat dan disimpan sebagai '{output_filename}'!")
plt.show()