import numpy as np
import matplotlib.pyplot as plt
import time

def f(x):
    return 4*x**3 - 6*x**2 + 7*x - 2.3

def df(x):
    return 12*x**2 - 12*x + 7 # Turunan untuk Newton-Raphson

TOLERANSI = 0.001 
X_LOWER, X_UPPER = 0.0, 1.0 
X_GUESS = 0.0 

def run_method(method_name, *args):
    start = time.perf_counter()
    if method_name == "Bisection":
        xl, xu, tol = args
        x_vals, errors = [], [100.0]
        if f(xl) * f(xu) >= 0: return [], [], 0
        while True:
            xr = (xl + xu) / 2
            x_vals.append(xr)
            if len(x_vals) > 1:
                ea = abs((xr - x_vals[-2]) / xr) * 100
                errors.append(ea)
                if ea < tol: break
            if f(xl) * f(xr) < 0: xu = xr
            else: xl = xr

    elif method_name == "Newton-Raphson":
        x0, tol = args
        x_vals, errors = [x0], [100.0]
        while True:
            if df(x0) == 0: break
            x1 = x0 - f(x0)/df(x0)
            x_vals.append(x1)
            ea = abs((x1 - x0) / x1) * 100
            errors.append(ea)
            if ea < tol: break
            x0 = x1

    elif method_name == "Regula Falsi":
        xl, xu, tol = args
        x_vals, errors = [], [100.0]
        if f(xl) * f(xu) >= 0: return [], [], 0
        while True:
            xr = xu - (f(xu) * (xl - xu)) / (f(xl) - f(xu))
            x_vals.append(xr)
            if len(x_vals) > 1:
                ea = abs((xr - x_vals[-2]) / xr) * 100
                errors.append(ea)
                if ea < tol: break
            if f(xl) * f(xr) < 0: xu = xr
            else: xl = xr

    elif method_name == "Secant":
        x0, x1, tol = args
        x_vals, errors = [x0, x1], [100.0, 100.0]
        while True:
            f_x0, f_x1 = f(x0), f(x1)
            if f_x1 - f_x0 == 0: break
            x_new = x1 - (f_x1 * (x1 - x0)) / (f_x1 - f_x0)
            x_vals.append(x_new)
            ea = abs((x_new - x1) / x_new) * 100
            errors.append(ea)
            if ea < tol: break
            x0, x1 = x1, x_new

    exec_time = (time.perf_counter() - start) * 1000
    return x_vals, errors, exec_time

methods_data = {
    "1": ("Bisection", run_method("Bisection", X_LOWER, X_UPPER, TOLERANSI)),
    "2": ("Newton-Raphson", run_method("Newton-Raphson", X_GUESS, TOLERANSI)),
    "3": ("Regula Falsi", run_method("Regula Falsi", X_LOWER, X_UPPER, TOLERANSI)),
    "4": ("Secant", run_method("Secant", X_LOWER, X_UPPER, TOLERANSI))
}

def plot_single(name, data):
    x_vals = data[0]
    x_plot = np.linspace(-0.2, 1.2, 400)
    y_plot = f(x_plot)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, label='f(x) = 4x^3 - 6x^2 + 7x - 2.3', color='blue', linewidth=1.5)
    plt.axhline(0, color='black', linewidth=1)

    if len(x_vals) > 2:
        if name == "Newton-Raphson":
            x0, x1 = x_vals[0], x_vals[1]
            plt.plot([x0, x1], [f(x0), 0], 'k--', linewidth=1.2, label='Garis Singgung (Turunan)')
            plt.plot([x0, x0], [0, f(x0)], 'k:', alpha=0.5)
        elif name in ["Secant", "Regula Falsi"]:
            x0, x1, x2 = x_vals[0], x_vals[1], x_vals[2]
            plt.plot([x0, x2], [f(x0), 0], 'k--', linewidth=1.2, label='Garis Potong (Secant)')
            plt.plot([x1, x1], [0, f(x1)], 'k:', alpha=0.5)

    limit = min(5, len(x_vals))
    colors = ['red', 'green', 'purple', 'orange', 'cyan']
    
    for i in range(limit):
        x = x_vals[i]
        if name == "Newton-Raphson" and i > 0:
            plt.plot(x, 0, marker='o', markersize=9, color=colors[i], markeredgecolor='black', zorder=5, label=f'Iterasi {i}')
        elif name in ["Secant", "Regula Falsi", "Bisection"] and i > 1:
            plt.plot(x, 0, marker='o', markersize=9, color=colors[i], markeredgecolor='black', zorder=5, label=f'Iterasi {i-1}')

        if i < 2:
            plt.plot(x, f(x), marker='o', markersize=7, color='black', zorder=4)

    plt.title(f'Grafik Konvergensi: {name}', fontsize=16, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('Nilai x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-3.0, 3.0])
    plt.legend()
    plt.tight_layout()
    plt.show()

while True:
    print("\n" + "="*40)
    print("   MENU PLOT METODE NUMERIK")
    print("="*40)
    print("1. Plot Bisection")
    print("2. Plot Newton-Raphson")
    print("3. Plot Regula Falsi")
    print("4. Plot Secant")
    print("5. Cetak Tabel Perbandingan Waktu")
    print("0. Keluar")
    print("="*40)
    
    pilihan = input("Masukkan angka pilihan Anda (0-5): ")
    
    if pilihan in ["1", "2", "3", "4"]:
        name, data = methods_data[pilihan]
        print(f"\nMembuat grafik untuk {name}...")
        plot_single(name, data)
    elif pilihan == "5":
        print("\n" + "="*65)
        print(f"{'METODE':<16} | {'ITERASI':<8} | {'WAKTU (ms)':<12} | {'AKAR AKHIR':<15}")
        print("="*65)
        for key, (name, data) in methods_data.items():
            x_vals, errs, t = data
            iters = len(errs) - 1 if name != "Secant" else len(errs) - 2
            print(f"{name:<16} | {iters:<8} | {t:<12.5f} | {x_vals[-1]:.6f}")
        print("="*65)
    elif pilihan == "0":
        print("Program dihentikan.")
        break
    else:
        print("Pilihan tidak valid. Silakan coba lagi.")