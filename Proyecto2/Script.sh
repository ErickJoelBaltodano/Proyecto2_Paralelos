set -euo pipefail

# ─────────────────────────── Configuración ────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
DATASETS_DIR="${SCRIPT_DIR}/datasets"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CSV_OUT="${RESULTS_DIR}/resultados_${TIMESTAMP}.csv"
SUMMARY_OUT="${RESULTS_DIR}/resumen_${TIMESTAMP}.txt"

SEQ_BIN="${SCRIPT_DIR}/src/Secuential"
PAR_BIN="${SCRIPT_DIR}/src/Paralelo"
SEQ_SRC="${SCRIPT_DIR}/src/Secuencial.c"
PAR_SRC="${SCRIPT_DIR}//src/Paralelo.c"

# Tamaños de dataset a probar (N = número de puntos, D = características)
# Formato: "N:D"
DATASET_SIZES=("5000:8" "20000:8" "80000:8")

# Configuraciones de hilos para la versión paralela
THREAD_COUNTS=(1 2 4 8)

# Número de repeticiones por combinación (para estadísticas)
N_RUNS=5

# Hiperparámetros del algoritmo TRIM
EPSILON="1e-6"
MAX_ITER="200"
LR="1e-4"
BGD_ITER="500"

# Fracción de puntos envenenados (el resto son limpios; TRIM debe recuperarlos)
POISON_FRACTION="0.2"

# Colores para la terminal
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ─────────────────────────── Funciones auxiliares ─────────────────────────
log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_title() { echo -e "\n${BOLD}${CYAN}══════ $* ══════${NC}"; }

# Extraer el tiempo (ms) de la salida del programa
extract_time() {
    # Busca la línea "Tiempo (ms)      : <valor>"
    grep -E "^Tiempo \(ms\)" <<< "$1" | awk '{print $NF}'
}

# Extraer MSE de la salida del programa
extract_mse() {
    grep -E "^MSE" <<< "$1" | awk '{print $NF}'
}

# Calcular media aritmética con awk
calc_mean() {
    local values=("$@")
    local sum=0
    for v in "${values[@]}"; do
        sum=$(awk "BEGIN{print $sum + $v}")
    done
    awk "BEGIN{printf \"%.4f\", $sum / ${#values[@]}}"
}

# Calcular desviación estándar con awk
calc_stddev() {
    local values=("$@")
    local n=${#values[@]}
    local mean
    mean=$(calc_mean "${values[@]}")
    local sq_sum=0
    for v in "${values[@]}"; do
        sq_sum=$(awk "BEGIN{d=$v-$mean; print $sq_sum + d*d}")
    done
    awk "BEGIN{printf \"%.4f\", sqrt($sq_sum / ($n > 1 ? $n-1 : 1))}"
}

# Verificar si un comando existe
require_cmd() {
    if ! command -v "$1" &>/dev/null; then
        log_error "Comando '$1' no encontrado. Instálalo y vuelve a intentarlo."
        exit 1
    fi
}

# ─────────────────────────── Crear directorios ────────────────────────────
mkdir -p "${RESULTS_DIR}" "${DATASETS_DIR}"

# ─────────────────────────── Compilación ──────────────────────────────────
log_title "Compilación"

CFLAGS="-O2 -march=native -Wall -Wextra -std=c11 -D_GNU_SOURCE"

log_info "Compilando versión secuencial..."
if gcc ${CFLAGS} -o "${SEQ_BIN}" "${SEQ_SRC}" -lm; then
    log_ok "Secuencial compilado exitosamente."
else
    log_error "Fallo al compilar Secuencial.c"
    exit 1
fi

log_info "Compilando versión paralela (OpenMP)..."
if gcc ${CFLAGS} -fopenmp -o "${PAR_BIN}" "${PAR_SRC}" -lm; then
    log_ok "Paralelo compilado exitosamente."
else
    log_error "Fallo al compilar Paralelo.c"
    exit 1
fi

# ─────────────────────────── Generación de datasets ───────────────────────
log_title "Generación de Datasets Sintéticos"

generate_dataset() {
    local N=$1
    local D=$2
    local outfile=$3
    local poison_frac="${POISON_FRACTION}"

    log_info "Generando dataset: N=${N}, D=${D}, poison=${poison_frac} → ${outfile}"

    python3 - <<PYEOF
import math, random, sys

random.seed(42)
N = ${N}
D = ${D}
poison_frac = ${poison_frac}
n_poison = int(N * poison_frac)
n_clean  = N - n_poison

# Coeficientes verdaderos (theta*)
theta_true = [random.uniform(-2.0, 2.0) for _ in range(D + 1)]  # +1 sesgo

outfile = "${outfile}"
with open(outfile, 'w') as f:
    # Puntos limpios (ruido gaussiano pequeño)
    for _ in range(n_clean):
        x = [random.gauss(0, 1) for _ in range(D)]
        y = theta_true[0] + sum(theta_true[j+1]*x[j] for j in range(D))
        y += random.gauss(0, 0.05)   # Ruido legítimo
        f.write(','.join(f'{v:.6f}' for v in x) + f',{y:.6f}\n')

    # Puntos envenenados (etiquetas corrompidas arbitrariamente)
    for _ in range(n_poison):
        x = [random.gauss(0, 1) for _ in range(D)]
        y = random.uniform(-50.0, 50.0)   # Etiqueta maliciosa
        f.write(','.join(f'{v:.6f}' for v in x) + f',{y:.6f}\n')

print(f"Dataset generado: {n_clean} limpios + {n_poison} envenenados = {N} total")
PYEOF
}

DATASET_FILES=()
for size_spec in "${DATASET_SIZES[@]}"; do
    N="${size_spec%%:*}"
    D="${size_spec##*:}"
    fname="${DATASETS_DIR}/dataset_N${N}_D${D}.csv"
    generate_dataset "${N}" "${D}" "${fname}"
    DATASET_FILES+=("${fname}")
done
log_ok "Todos los datasets generados."

# ─────────────────────────── Inicializar CSV de resultados ────────────────
log_title "Iniciando Experimentos"

cat > "${CSV_OUT}" << 'CSVHDR'
version,N,D,threads,run,time_ms,mse
CSVHDR
log_info "Resultados brutos → ${CSV_OUT}"

# ─────────────────────────── Función de experimento ───────────────────────
run_experiment() {
    local binary="$1"
    local dataset="$2"
    local threads="$3"       # 0 = secuencial
    local version_label="$4"
    local N="$5"
    local D="$6"

    local times=()
    local mse_vals=()

    for run in $(seq 1 ${N_RUNS}); do
        local output

        if [[ "$threads" -eq 0 ]]; then
            # Versión secuencial (sin OMP_NUM_THREADS)
            output=$("${binary}" "${dataset}" "${EPSILON}" "${MAX_ITER}" \
                                 "${LR}" "${BGD_ITER}" 2>&1)
        else
            # Versión paralela con N hilos
            output=$(OMP_NUM_THREADS="${threads}" "${binary}" "${dataset}" \
                     "${EPSILON}" "${MAX_ITER}" "${LR}" "${BGD_ITER}" 2>&1)
        fi

        local t
        t=$(extract_time "${output}")
        local m
        m=$(extract_mse "${output}")

        if [[ -z "$t" || -z "$m" ]]; then
            log_warn "Ejecución ${run} no produjo salida válida."
            t="NaN"; m="NaN"
        fi

        times+=("$t")
        mse_vals+=("$m")

        local t_label="${threads}"
        [[ "$threads" -eq 0 ]] && t_label="seq"
        echo "${version_label},${N},${D},${t_label},${run},${t},${m}" \
            >> "${CSV_OUT}"

        printf "    Run %d/%d: %.4f ms  MSE=%.6f\n" "${run}" "${N_RUNS}" \
               "${t}" "${m}"
    done

    # Calcular estadísticos (excluir NaN)
    local valid_times=()
    for v in "${times[@]}"; do
        [[ "$v" != "NaN" ]] && valid_times+=("$v")
    done

    if [[ ${#valid_times[@]} -gt 0 ]]; then
        local mean_t stddev_t
        mean_t=$(calc_mean  "${valid_times[@]}")
        stddev_t=$(calc_stddev "${valid_times[@]}")
        echo "  → Media: ${mean_t} ms  ±  ${stddev_t} ms"
    fi
}

# ─────────────────────────── Loop principal de experimentos ───────────────
{
    echo "======================================================================"
    echo "  RESUMEN EXPERIMENTAL TRIM — ${TIMESTAMP}"
    echo "  Configuración: epsilon=${EPSILON} max_iter=${MAX_ITER}"
    echo "                 lr=${LR} bgd_iter=${BGD_ITER}"
    echo "                 N_RUNS=${N_RUNS} POISON_FRAC=${POISON_FRACTION}"
    echo "======================================================================"
    echo ""
    printf "%-12s %-8s %-8s %-8s %-12s %-12s %-10s\n" \
           "Version" "N" "D" "Threads" "Media(ms)" "StdDev(ms)" "Speedup"
    echo "----------------------------------------------------------------------"
} | tee "${SUMMARY_OUT}"

for size_spec in "${DATASET_SIZES[@]}"; do
    N="${size_spec%%:*}"
    D="${size_spec##*:}"
    dataset="${DATASETS_DIR}/dataset_N${N}_D${D}.csv"

    log_title "Dataset N=${N}, D=${D}"

    # ── Versión secuencial ──────────────────────────────────────────────
    log_info "Ejecutando TRIM Secuencial (${N_RUNS} repeticiones)..."
    seq_times=()
    for run in $(seq 1 ${N_RUNS}); do
        output=$("${SEQ_BIN}" "${dataset}" "${EPSILON}" "${MAX_ITER}" \
                              "${LR}" "${BGD_ITER}" 2>&1)
        t=$(extract_time "${output}")
        m=$(extract_mse  "${output}")
        [[ -z "$t" ]] && { log_warn "Run ${run}: sin tiempo"; t="NaN"; }
        [[ -z "$m" ]] && m="NaN"
        seq_times+=("$t")
        echo "sequential,${N},${D},1,${run},${t},${m}" >> "${CSV_OUT}"
        printf "    Run %d/%d: %s ms  MSE=%s\n" "${run}" "${N_RUNS}" "${t}" "${m}"
    done

    # Estadísticos de la versión secuencial
    valid_seq=()
    for v in "${seq_times[@]}"; do [[ "$v" != "NaN" ]] && valid_seq+=("$v"); done
    seq_mean="N/A"; seq_std="N/A"
    if [[ ${#valid_seq[@]} -gt 0 ]]; then
        seq_mean=$(calc_mean   "${valid_seq[@]}")
        seq_std=$(calc_stddev  "${valid_seq[@]}")
    fi
    log_ok "Secuencial: media=${seq_mean} ms ± ${seq_std} ms"
    printf "%-12s %-8s %-8s %-8s %-12s %-12s %-10s\n" \
           "sequential" "${N}" "${D}" "1" "${seq_mean}" "${seq_std}" "1.00x" \
           | tee -a "${SUMMARY_OUT}"

    # ── Versión paralela con distintos números de hilos ────────────────
    for threads in "${THREAD_COUNTS[@]}"; do
        log_info "Ejecutando TRIM Paralelo (threads=${threads}, ${N_RUNS} repeticiones)..."
        par_times=()
        for run in $(seq 1 ${N_RUNS}); do
            output=$(OMP_NUM_THREADS="${threads}" "${PAR_BIN}" "${dataset}" \
                     "${EPSILON}" "${MAX_ITER}" "${LR}" "${BGD_ITER}" 2>&1)
            t=$(extract_time "${output}")
            m=$(extract_mse  "${output}")
            [[ -z "$t" ]] && { log_warn "Run ${run}: sin tiempo"; t="NaN"; }
            [[ -z "$m" ]] && m="NaN"
            par_times+=("$t")
            echo "parallel,${N},${D},${threads},${run},${t},${m}" >> "${CSV_OUT}"
            printf "    Run %d/%d: %s ms  MSE=%s\n" "${run}" "${N_RUNS}" "${t}" "${m}"
        done

        # Estadísticos y speedup
        valid_par=()
        for v in "${par_times[@]}"; do [[ "$v" != "NaN" ]] && valid_par+=("$v"); done
        par_mean="N/A"; par_std="N/A"; speedup="N/A"
        if [[ ${#valid_par[@]} -gt 0 ]]; then
            par_mean=$(calc_mean   "${valid_par[@]}")
            par_std=$(calc_stddev  "${valid_par[@]}")
            if [[ "$seq_mean" != "N/A" && "$par_mean" != "N/A" ]]; then
                speedup=$(awk "BEGIN{printf \"%.2fx\", ${seq_mean}/${par_mean}}")
            fi
        fi
        log_ok "Paralelo t=${threads}: media=${par_mean} ms ± ${par_std} ms  Speedup=${speedup}"
        printf "%-12s %-8s %-8s %-8s %-12s %-12s %-10s\n" \
               "parallel" "${N}" "${D}" "${threads}" "${par_mean}" "${par_std}" "${speedup}" \
               | tee -a "${SUMMARY_OUT}"
    done
    echo "----------------------------------------------------------------------" | tee -a "${SUMMARY_OUT}"
done

# ─────────────────────────── Cierre del resumen ───────────────────────────
{
    echo ""
    echo "======================================================================"
    echo "  Resultados brutos: ${CSV_OUT}"
    echo "  Resumen:           ${SUMMARY_OUT}"
    echo "  Fecha/Hora:        $(date)"
    echo "======================================================================"
} | tee -a "${SUMMARY_OUT}"

log_ok "Experimentos completados."
log_info "Para generar gráficas con Python, ejecuta:"
echo "  python3 plot_results.py ${CSV_OUT}"

# ─────────────────────────── Script de graficación (auto-generado) ────────
cat > "${SCRIPT_DIR}/plot_results.py" << 'PYEOF'
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
PYEOF
chmod +x "${SCRIPT_DIR}/plot_results.py"
log_info "Script de graficación generado: plot_results.py"