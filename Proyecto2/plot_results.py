#!/usr/bin/env python3
"""
plot_results.py  —  Genera gráficas de Speedup y Tiempo de Ejecución
a partir del CSV producido por run_experiments.sh

Uso: python3 plot_results.py results/resultados_<timestamp>.csv
"""
import sys, csv, collections, math

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Instala matplotlib: pip install matplotlib")
    sys.exit(1)

if len(sys.argv) < 2:
    print(f"Uso: {sys.argv[0]} <resultados.csv>")
    sys.exit(1)

csv_path = sys.argv[1]

# Leer datos
data = collections.defaultdict(list)   # (version, N, D, threads) → [time_ms]

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['version'], int(row['N']), int(row['D']), row['threads'])
        try:
            data[key].append(float(row['time_ms']))
        except ValueError:
            pass

def mean(lst): return sum(lst) / len(lst) if lst else float('nan')
def stddev(lst):
    if len(lst) < 2: return 0.0
    m = mean(lst)
    return math.sqrt(sum((x - m)**2 for x in lst) / (len(lst) - 1))

# Organizar por N
Ns = sorted(set(k[1] for k in data))
thread_counts = sorted(set(int(k[3]) for k in data if k[3] != 'seq'))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ─── Gráfica 1: Tiempo vs Número de Hilos (por tamaño N) ──────────────────
ax = axes[0]
for N in Ns:
    D = next(k[3] for k in data if k[1] == N)  # tomar D de cualquier key
    D_val = next(k[2] for k in data if k[1] == N)

    # Tiempo secuencial (base)
    seq_key = ('sequential', N, D_val, '1')
    seq_t = mean(data.get(seq_key, [float('nan')]))

    times = []
    stds  = []
    threads_plot = [1] + thread_counts
    for t in threads_plot:
        if t == 1:
            times.append(seq_t)
            stds.append(stddev(data.get(seq_key, [])))
        else:
            key = ('parallel', N, D_val, str(t))
            times.append(mean(data.get(key, [float('nan')])))
            stds.append(stddev(data.get(key, [])))

    ax.errorbar(threads_plot, times, yerr=stds, marker='o', label=f'N={N}',
                capsize=4, linewidth=1.8)

ax.set_xlabel('Número de Hilos (OMP_NUM_THREADS)')
ax.set_ylabel('Tiempo de Ejecución (ms)')
ax.set_title('Tiempo vs. Número de Hilos')
ax.legend(); ax.grid(True, alpha=0.4)
ax.set_xticks([1] + thread_counts)

# ─── Gráfica 2: Speedup vs Número de Hilos ────────────────────────────────
ax = axes[1]
for N in Ns:
    D_val = next(k[2] for k in data if k[1] == N)
    seq_key = ('sequential', N, D_val, '1')
    seq_t = mean(data.get(seq_key, [float('nan')]))

    speedups = [1.0]
    for t in thread_counts:
        key = ('parallel', N, D_val, str(t))
        par_t = mean(data.get(key, [float('nan')]))
        speedups.append(seq_t / par_t if par_t > 0 else float('nan'))

    ax.plot([1] + thread_counts, speedups, marker='s', label=f'N={N}',
            linewidth=1.8)

# Línea ideal (speedup lineal)
ax.plot([1] + thread_counts, [1] + thread_counts, 'k--', alpha=0.4,
        label='Speedup ideal')
ax.set_xlabel('Número de Hilos (OMP_NUM_THREADS)')
ax.set_ylabel('Speedup (T_seq / T_par)')
ax.set_title('Speedup vs. Número de Hilos')
ax.legend(); ax.grid(True, alpha=0.4)
ax.set_xticks([1] + thread_counts)

plt.suptitle('Análisis de Rendimiento — Algoritmo TRIM (OpenMP)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out_png = csv_path.replace('.csv', '_plots.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"Gráficas guardadas en: {out_png}")
plt.show()
