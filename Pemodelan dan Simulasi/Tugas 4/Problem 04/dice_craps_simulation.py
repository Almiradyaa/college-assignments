import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='colorblind', font='DejaVu Sans')

# =====================================================================
# PROBABILITAS TEORETIS
# =====================================================================
theoretical_counts = {
    2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
    8: 5, 9: 4, 10: 3, 11: 2, 12: 1
}
theoretical_probs = {k: v / 36.0 for k, v in theoretical_counts.items()}
THEO_CRAPS_WIN_PROB = 244.0 / 495.0  

# =====================================================================
# FUNGSI SIMULASI PELEMPARAN DUA DADU
# =====================================================================
def simulate_dice_rolls(N, seed=42):
    np.random.seed(seed)
    die1 = np.random.randint(1, 7, size=N)
    die2 = np.random.randint(1, 7, size=N)
    sums = die1 + die2
    
    unique, counts = np.unique(sums, return_counts=True)
    empirical_probs = dict(zip(unique, counts / N))
    
    results = []
    for s in range(2, 13):
        emp = empirical_probs.get(s, 0.0)
        theo = theoretical_probs[s]
        abs_err = abs(emp - theo)
        rel_err = (abs_err / theo) * 100.0
        results.append({
            'Sum': s,
            'Kemungkinan (dari 36)': theoretical_counts[s],
            'Probabilitas Teoretis': theo,
            'Probabilitas Empiris': emp,
            'Galat Absolut': abs_err,
            'Galat Relatif (%)': rel_err
        })
    return pd.DataFrame(results)

# =====================================================================
# FUNGSI SIMULASI PERMAINAN CRAPS
# =====================================================================
def play_craps_game():
    # Lemparan Pertama (Come-out Roll)
    roll = np.random.randint(1, 7) + np.random.randint(1, 7)
    if roll in (7, 11):
        return 1  # Menang langsung (Natural)
    elif roll in (2, 3, 12):
        return 0  # Kalah langsung (Craps)
    else:
        point = roll
        while True:
            next_roll = np.random.randint(1, 7) + np.random.randint(1, 7)
            if next_roll == point:
                return 1  # Menang (Mencapai Point)
            elif next_roll == 7:
                return 0  # Kalah (Seven-out)

def simulate_craps_games(N_games_list, seed=42):
    np.random.seed(seed)
    max_N = max(N_games_list)
    outcomes = np.array([play_craps_game() for _ in range(max_N)])
    
    results = []
    for N in N_games_list:
        sub_outcomes = outcomes[:N]
        wins = np.sum(sub_outcomes)
        win_rate = wins / N
        abs_err = abs(win_rate - THEO_CRAPS_WIN_PROB)
        rel_err = (abs_err / THEO_CRAPS_WIN_PROB) * 100.0
        results.append({
            'N (Games Played)': N,
            'Jumlah Menang': wins,
            'Jumlah Kalah': N - wins,
            'Peluang Menang (Empiris)': win_rate,
            'Peluang Menang (Teoretis)': THEO_CRAPS_WIN_PROB,
            'Galat Absolut': abs_err,
            'Galat Relatif (%)': rel_err
        })
    return pd.DataFrame(results), outcomes

N_dice = 1_000_000
N_craps_list = [10_000, 100_000, 1_000_000]

df_dice = simulate_dice_rolls(N_dice, seed=42)
df_craps, craps_outcomes_1m = simulate_craps_games(N_craps_list, seed=42)

print("\n" + "="*65)
print("       PROBABILITAS JUMLAH DADU (N = 1,000,000)")
print("="*65)
print(df_dice.to_string(index=False))

print("\n" + "="*65)
print("       HASIL SIMULASI PERMAINAN CRAPS")
print("="*65)
print(df_craps.to_string(index=False))
print("="*65 + "\n")

# Ekspor CSV
df_dice.to_csv('dice_sum_probabilities.csv', index=False)
df_craps.to_csv('craps_simulation_results.csv', index=False)
print("--> Berkas 'dice_sum_probabilities.csv' dan 'craps_simulation_results.csv' berhasil disimpan.\n")

# =====================================================================
# GRAFIK 1: DIAGRAM BATANG DISTRIBUSI JUMLAH DADU
# =====================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
x_sums = df_dice['Sum']
width = 0.35

ax1.bar(x_sums - width/2, df_dice['Probabilitas Empiris'], width, label='Empiris (Monte Carlo N=1M)', color='#1f77b4', alpha=0.85)
ax1.bar(x_sums + width/2, df_dice['Probabilitas Teoretis'], width, label='Teoretis (Eksak)', color='#ff7f0e', alpha=0.85)

ax1.set_title('Perbandingan Distribusi Probabilitas Jumlah Dua Dadu', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Jumlah Mata Dadu (Sum)', fontsize=10)
ax1.set_ylabel('Probabilitas', fontsize=10)
ax1.set_xticks(x_sums)
ax1.legend(loc='upper right', frameon=True)
plt.tight_layout()

plt.savefig('dice_sum_distribution.png', bbox_inches='tight')
print("--> Grafik 1 'dice_sum_distribution.png' berhasil disimpan.")

# =====================================================================
# GRAFIK 2: KONVERGENSI WIN RATE CRAPS
# =====================================================================
fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=150)
N_plot = 100_000
cum_wins = np.cumsum(craps_outcomes_1m[:N_plot])
iters = np.arange(1, N_plot + 1)
running_win_rate = cum_wins / iters

ax2.plot(iters, running_win_rate, color='#1f77b4', linewidth=1.2, label='Tingkat Kemenangan Kumulatif')
ax2.axhline(THEO_CRAPS_WIN_PROB, color='red', linestyle='--', linewidth=2, label=f'Win Rate Teoretis ({THEO_CRAPS_WIN_PROB:.4f})')

ax2.set_title('Konvergensi Win Rate Permainan Craps (N = 100,000)', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Jumlah Permainan (Games Played)', fontsize=10)
ax2.set_ylabel('Tingkat Kemenangan (Win Rate)', fontsize=10)
ax2.set_ylim(0.40, 0.60)
ax2.legend(loc='upper right', frameon=True)
plt.tight_layout()

plt.savefig('craps_win_rate_convergence.png', bbox_inches='tight')
print("--> Grafik 2 'craps_win_rate_convergence.png' berhasil disimpan.\n")

plt.show()