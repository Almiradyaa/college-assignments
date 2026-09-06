import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Atur tema visual agar bersih dan profesional
plt.style.use('seaborn-v0_8-whitegrid')

# 2. Ambil data asli dari seaborn
df = sns.load_dataset('diamonds')
prices = df['price'].values

# 3. Lakukan transformasi log10 pada data harga
log10_prices = np.log10(prices)

# 4. Gambar histogram dengan KDE (Kernel Density Estimate) untuk memperhalus kurva
plt.figure(figsize=(8, 5), dpi=150)
sns.histplot(log10_prices, bins=100, color='#f79420', edgecolor='white', kde=True)

# 5. Kustomisasi sumbu agar menampilkan label nilai harga asli ($) demi kemudahan membaca
# Nilai log10 dari 1000 adalah 3, nilai log10 dari 10000 adalah 4
ticks = [2.5, 3.0, 3.5, 4.0, 4.25]
tick_labels = [f"${int(10**t):,}" for t in ticks]
plt.xticks(ticks, tick_labels)

# 6. Tambahkan judul dan label akademis
plt.title("Histogram Distribusi Harga Berlian dalam Skala Log10\nMenunjukkan Bukti Empiris Bimodalitas (Dua Puncak)", 
fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Harga Jual Berlian (Skala Log10 USD)", fontsize=10)
plt.ylabel("Jumlah Berlian (Frequency Count)", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('histogram_log10_bimodal.png')
plt.show()
print("Grafik bimodal log10 berhasil disimpan sebagai 'histogram_log10_bimodal.png'!")