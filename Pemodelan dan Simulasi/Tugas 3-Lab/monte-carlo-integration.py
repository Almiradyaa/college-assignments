"""
=============================================================================
PRAKTIKUM PEMODELAN DAN SIMULASI - MODUL 3
Lab Session 1: Integration using Monte Carlo Simulation
-----------------------------------------------------------------------------
Fungsi        : f(x) = x^2
Batas         : a = 0.0, b = 3.0
PRNG          : modul `random` bawaan Python, seed = 2
Iterasi       : NumSteps = 1.000.000 (deteksi min-max)
                N        = 1.000.000 (titik acak Monte Carlo)
Nilai eksak   : integral x^2 dx dari 0 s.d. 3 = 9.0
=============================================================================
"""

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt

# TASK 1 - NUMERICAL SOLUTION 
def f(x):
    """Fungsi yang akan diintegrasikan: f(x) = x^2."""
    return x ** 2

# Parameter simulasi 
a = 0.0                 # batas bawah integrasi
b = 3.0                 # batas atas integrasi
NumSteps = 1_000_000    # jumlah langkah untuk deteksi nilai minimum-maksimum
N = 1_000_000           # jumlah titik acak Monte Carlo
SEED = 2                # seed PRNG agar hasil reproducible
EXACT_INTEGRAL = 9.0    # solusi analitis: [x^3 / 3] dari 0 s.d. 3 = 9.0

# Inisialisasi PRNG 
random.seed(SEED)

PLOT_SAMPLE = 8_000     # jumlah titik yang benar-benar digambar pada grafik
plot_stride = N // PLOT_SAMPLE   # ambil 1 titik setiap `plot_stride` iterasi

x_inside, y_inside = [], []      
x_outside, y_outside = [], []    

print("=" * 66)
print(" TASK 1 - INISIALISASI")
print("=" * 66)
print(f" f(x)            : x^2")
print(f" Batas integrasi : [{a}, {b}]")
print(f" PRNG seed       : {SEED}")
print(f" NumSteps        : {NumSteps:,}")
print(f" N (titik acak)  : {N:,}\n")


# TASK 2 - MIN-MAX DETECTION
step = (b - a) / NumSteps
ymin = f(a)
ymax = f(a)

for i in range(NumSteps + 1):
    x = a + i * step
    y = f(x)
    if y < ymin:
        ymin = y
    if y > ymax:
        ymax = y

print("=" * 66)
print(" TASK 2 - DETEKSI MIN-MAX")
print("=" * 66)
print(f" ymin = {ymin:.6f}")
print(f" ymax = {ymax:.6f}\n")

# TASK 3 - MONTE CARLO METHOD
A = (b - a) * (ymax - ymin)

M = 0   

for i in range(N):
    x = random.uniform(a, b)        # x ~ U[a, b]
    y = random.uniform(ymin, ymax)  # y ~ U[ymin, ymax]

    if y <= f(x):                   
        M += 1
        if i % plot_stride == 0:    
            x_inside.append(x)
            y_inside.append(y)
    else:
        if i % plot_stride == 0:
            x_outside.append(x)
            y_outside.append(y)

# Estimasi nilai integral
NumericalIntegral = (M / N) * A

# Analisis galat terhadap solusi analitis
absolute_error = abs(EXACT_INTEGRAL - NumericalIntegral)
relative_error = (absolute_error / EXACT_INTEGRAL) * 100

print("=" * 66)
print(" TASK 3 - SIMULASI MONTE CARLO")
print("=" * 66)
print(f" Luas bounding box (A)    : {A:.6f}")
print(f" Titik di bawah kurva (M) : {M:,}")
print(f" Rasio M/N                : {M / N:.6f}")
print(f" Integral numerik         : {NumericalIntegral:.6f}")
print(f" Integral eksak           : {EXACT_INTEGRAL:.6f}")
print(f" Galat absolut            : {absolute_error:.6f}")
print(f" Galat relatif            : {relative_error:.6f} %\n")

# TASK 4 - VISUALIZATION
x_curve = np.linspace(a, b, 500)
y_curve = f(x_curve)

plt.figure(figsize=(9, 6))

# Sebaran titik acak (sub-sampel)
plt.scatter(x_outside, y_outside, s=3, color="yellow",
            label=f"Di atas kurva (sampel: {len(x_outside):,})")
plt.scatter(x_inside, y_inside, s=3, color="blue",
            label=f"Di bawah kurva (sampel: {len(x_inside):,})")

# Kurva fungsi 
plt.plot(x_curve, y_curve, color="red", linewidth=2.5, label="f(x) = x$^2$")

# Kotak pembatas (bounding box) 
plt.plot([a, b, b, a, a], [ymin, ymin, ymax, ymax, ymin],
        color="black", linestyle="--", linewidth=1.2, label="Bounding box")

plt.title("Integrasi Monte Carlo f(x) = x$^2$ pada [0, 3]\n"
        f"N = {N:,} | Estimasi = {NumericalIntegral:.4f} | "
        f"Eksak = {EXACT_INTEGRAL} | Galat = {relative_error:.4f}%")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend(loc="upper left", framealpha=0.9)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("monte_carlo_integration.png", dpi=150)
plt.show()

print("=" * 66)
print(" TASK 4 - Grafik disimpan sebagai 'monte_carlo_integration.png'")
print("=" * 66)