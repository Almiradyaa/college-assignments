import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='colorblind')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def run_monte_carlo_pi(N_samples, seed=42):
    """
    Fungsi utama untuk mengestimasi nilai Pi dengan Simulasi Monte Carlo.
    """
    np.random.seed(seed)
    true_pi = np.pi
    results = []

    for N in N_samples:
        # Pembangkitan koordinat acak seragam (x, y) dalam persegi 1x1
        x = np.random.uniform(0, 1, N)
        y = np.random.uniform(0, 1, N)
        
        # Pengujian kondisi titik di dalam seperempat lingkaran: x^2 + y^2 <= 1
        inside = (x**2 + y**2) <= 1.0
        M = np.sum(inside)
        
        # Estimasi Pi = 4 * (M / N)
        pi_hat = 4.0 * M / N
        abs_err = np.abs(pi_hat - true_pi)
        rel_err = (abs_err / true_pi) * 100.0
        
        results.append({
            'N (Jumlah Sampel)': N,
            'M (Titik di Dalam)': M,
            'Estimasi Pi (π̂)': pi_hat,
            'Galat Absolut': abs_err,
            'Galat Relatif (%)': rel_err
        })

    return pd.DataFrame(results)

N_list = [10000, 100000, 1000000]
df_results = run_monte_carlo_pi(N_list, seed=42)

print("\n" + "="*60)
print("       TABEL HASIL ESTIMASI MONTE CARLO PI")
print("="*60)
print(df_results.to_string(index=False))
print("="*60 + "\n")

df_results.to_csv("monte_carlo_pi_results.csv", index=False)
print("--> Tabel hasil telah disimpan sebagai 'monte_carlo_pi_results.csv'\n")

# =====================================================================
# GRAFIK 1: SCATTER PLOT SEBARAN TITIK (N = 10,000)
# =====================================================================
N_plot = 10000
np.random.seed(42)
x_plot = np.random.uniform(0, 1, N_plot)
y_plot = np.random.uniform(0, 1, N_plot)
inside_plot = (x_plot**2 + y_plot**2) <= 1.0
pi_plot = 4.0 * np.sum(inside_plot) / N_plot

fig1, ax1 = plt.subplots(figsize=(8, 8), dpi=150)

ax1.scatter(x_plot[inside_plot], y_plot[inside_plot], color='#1f77b4', s=3, alpha=0.6, label='Di Dalam Lingkaran')
ax1.scatter(x_plot[~inside_plot], y_plot[~inside_plot], color='#d62728', s=3, alpha=0.6, label='Di Luar Lingkaran')

theta = np.linspace(0, np.pi/2, 200)
ax1.plot(np.cos(theta), np.sin(theta), color='black', linewidth=2, linestyle='--', label='Kurva Seperempat Lingkaran')

ax1.set_title(f'Monte Carlo Pi Estimation (N = {N_plot:,}, π̂ = {pi_plot:.4f})', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Sumbu X', fontsize=10)
ax1.set_ylabel('Sumbu Y', fontsize=10)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_aspect('equal')
ax1.legend(loc='lower left', frameon=True)
plt.tight_layout()

plt.savefig('monte_carlo_pi_scatter.png', bbox_inches='tight')
print("--> Grafik 1 (Scatter Plot) disimpan sebagai 'monte_carlo_pi_scatter.png'")

# =====================================================================
# GRAFIK 2: GRAFIK KONVERGENSI ESTiMASI PI (N = 100,000)
# =====================================================================
N_conv = 100000
np.random.seed(42)
x_conv = np.random.uniform(0, 1, N_conv)
y_conv = np.random.uniform(0, 1, N_conv)
inside_conv = (x_conv**2 + y_conv**2) <= 1.0

cumulative_m = np.cumsum(inside_conv)
iterations = np.arange(1, N_conv + 1)
pi_convergence = 4.0 * cumulative_m / iterations

fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=150)
ax2.plot(iterations, pi_convergence, color='#1f77b4', linewidth=1.2, label='Estimasi Kumulatif π̂')
ax2.axhline(np.pi, color='red', linestyle='--', linewidth=2, label=f'Nilai Sejati π ({np.pi:.6f})')

ax2.set_title('Konvergensi Estimasi Monte Carlo Pi terhadap Nilai Sejati', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Jumlah Iterasi (N)', fontsize=10)
ax2.set_ylabel('Nilai Estimasi π', fontsize=10)
ax2.set_ylim(2.8, 3.5)
ax2.legend(loc='upper right', frameon=True)
plt.tight_layout()

plt.savefig('monte_carlo_pi_convergence.png', bbox_inches='tight')
print("--> Grafik 2 (Konvergensi) disimpan sebagai 'monte_carlo_pi_convergence.png'\n")

plt.show()