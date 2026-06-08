#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─────────────────────────── Parámetros por defecto ─────────────────────── */
#define DEFAULT_EPSILON   1e-6
#define DEFAULT_MAX_ITER  200
#define DEFAULT_LR        1e-4
#define DEFAULT_BGD_ITER  500
#define MAX_FEATURES      256
#define MAX_LINE          65536

/* ─────────────────────────── Estructuras ────────────────────────────────── */
typedef struct {
    double *X;   /* [N × (D+1)], columna 0 = término de sesgo              */
    double *y;   /* [N]                                                      */
    int     N;
    int     D;
} Dataset;

typedef struct {
    double residual;
    int    idx;
} ResidualEntry;

/* ─────────────────────────── Utilidades ─────────────────────────────────── */
static int cmp_residual(const void *a, const void *b)
{
    const ResidualEntry *ra = (const ResidualEntry *)a;
    const ResidualEntry *rb = (const ResidualEntry *)b;
    if (ra->residual < rb->residual) return -1;
    if (ra->residual > rb->residual) return  1;
    return 0;
}

/* ─────────────────────────── Carga del dataset ──────────────────────────── */
static int load_dataset(const char *path, Dataset *ds)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: no se pudo abrir '%s'\n", path);
        return -1;
    }

    char line[MAX_LINE];
    int  n_lines = 0;
    int  D_det   = -1;

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (D_det < 0) {
            int cols = 0;
            char *tmp = strdup(line);
            char *tok = strtok(tmp, ",\t ");
            while (tok) { cols++; tok = strtok(NULL, ",\t "); }
            free(tmp);
            if (cols < 2) {
                fprintf(stderr, "Error: se necesitan al menos 2 columnas.\n");
                fclose(fp); return -1;
            }
            D_det = cols - 1;
        }
        n_lines++;
    }
    rewind(fp);

    if (n_lines == 0) {
        fprintf(stderr, "Error: dataset vacío.\n");
        fclose(fp); return -1;
    }
    if (D_det > MAX_FEATURES) {
        fprintf(stderr, "Error: demasiadas características (%d > %d).\n",
                D_det, MAX_FEATURES);
        fclose(fp); return -1;
    }

    ds->N = n_lines;
    ds->D = D_det;
    ds->X = (double *)malloc((size_t)ds->N * (ds->D + 1) * sizeof(double));
    ds->y = (double *)malloc((size_t)ds->N * sizeof(double));
    if (!ds->X || !ds->y) {
        fprintf(stderr, "Error: memoria insuficiente.\n");
        fclose(fp); return -1;
    }

    int row = 0;
    while (fgets(line, sizeof(line), fp) && row < ds->N) {
        if (line[0] == '#' || line[0] == '\n') continue;
        double *xrow = ds->X + (size_t)row * (ds->D + 1);
        xrow[0] = 1.0;
        char *tok = strtok(line, ",\t \n");
        for (int c = 1; c <= ds->D && tok; c++) {
            xrow[c] = atof(tok);
            tok = strtok(NULL, ",\t \n");
        }
        ds->y[row] = tok ? atof(tok) : 0.0;
        row++;
    }
    fclose(fp);
    ds->N = row;
    return 0;
}

static void free_dataset(Dataset *ds)
{
    free(ds->X); free(ds->y);
    ds->X = NULL; ds->y = NULL;
}

/* ─────────────────────────── Predicción puntual ─────────────────────────── */
static inline double predict(const double *xrow, const double *theta, int Dp1)
{
    double s = 0.0;
    for (int j = 0; j < Dp1; j++) s += xrow[j] * theta[j];
    return s;
}

/* ─────────────────────────── BGD paralelo sobre subconjunto ────────────── */
/*
 * Misma lógica que la versión secuencial.
 * Paralelismo:
 *   - El bucle interno de acumulación del gradiente usa reduction(+:grad_j)
 *     a través de un arreglo de reducción manual (OpenMP < 4.5 no soporta
 *     reduction sobre arreglos; usamos critical + local acumulador).
 *   - Para compiladores con OpenMP ≥ 4.5 podría usarse:
 *       #pragma omp parallel for reduction(+:grad[:Dp1])
 *     pero el enfoque con atomic es más portable.
 */
static void bgd_parallel(const Dataset *ds,
                         const int     *subset,
                         int            R,
                         double        *theta,
                         int            Dp1,
                         double         lr,
                         int            bgd_iter)
{
    double *grad = (double *)calloc((size_t)Dp1, sizeof(double));
    if (!grad) { fprintf(stderr, "Error alloc grad\n"); return; }

    for (int it = 0; it < bgd_iter; it++) {
        memset(grad, 0, (size_t)Dp1 * sizeof(double));

        /*
         * Paralelización del gradiente:
         * Cada hilo acumula en un buffer local y luego reduce al grad global
         * con #pragma omp critical (correcto y portable para cualquier D).
         *
         * Alternativa OpenMP 4.5+:
         *   #pragma omp parallel for reduction(+:grad[:Dp1])
         */
        #pragma omp parallel
        {
            double *grad_local = (double *)calloc((size_t)Dp1, sizeof(double));
            if (!grad_local) {
                #pragma omp critical
                { fprintf(stderr, "Error alloc grad_local\n"); }
            } else {
                #pragma omp for schedule(static)
                for (int s = 0; s < R; s++) {
                    int i = subset[s];
                    const double *xrow = ds->X + (size_t)i * Dp1;
                    double err = predict(xrow, theta, Dp1) - ds->y[i];
                    for (int j = 0; j < Dp1; j++)
                        grad_local[j] += err * xrow[j];
                }

                /* Reducción al gradiente global */
                #pragma omp critical
                {
                    for (int j = 0; j < Dp1; j++)
                        grad[j] += grad_local[j];
                }
                free(grad_local);
            }
        } /* fin parallel */

        /* Actualizar theta – independiente por elemento: paralelo */
        double scale = lr / R;
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < Dp1; j++)
            theta[j] -= scale * grad[j];
    }
    free(grad);
}

/* ─────────────────────────── Algoritmo TRIM paralelo ───────────────────── */
static int trim_parallel(const Dataset *ds,
                         int            R,
                         double         epsilon,
                         int            max_iter,
                         double         lr,
                         int            bgd_iter,
                         double        *theta_out)
{
    int N   = ds->N;
    int Dp1 = ds->D + 1;

    double *theta     = (double *)calloc((size_t)Dp1, sizeof(double));
    double *theta_old = (double *)calloc((size_t)Dp1, sizeof(double));
    int    *subset    = (int    *)malloc((size_t)R * sizeof(int));
    ResidualEntry *res_table = (ResidualEntry *)malloc(
                                   (size_t)N * sizeof(ResidualEntry));

    if (!theta || !theta_old || !subset || !res_table) {
        fprintf(stderr, "Error: memoria insuficiente en trim_parallel\n");
        free(theta); free(theta_old); free(subset); free(res_table);
        return -1;
    }

    /* Subconjunto inicial: primeros R puntos */
    for (int i = 0; i < R; i++) subset[i] = i;

    int iter = 0;
    for (iter = 0; iter < max_iter; iter++) {
        memcpy(theta_old, theta, (size_t)Dp1 * sizeof(double));

        /* ── Paso 1: BGD paralelo ────────────────────────────────────── */
        bgd_parallel(ds, subset, R, theta, Dp1, lr, bgd_iter);

        /* ── Paso 2: Calcular residuos en paralelo (sección más pesada) ─ */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            const double *xrow = ds->X + (size_t)i * Dp1;
            double err = predict(xrow, theta, Dp1) - ds->y[i];
            res_table[i].residual = err * err;
            res_table[i].idx      = i;
            /* Cada hilo escribe en su propia celda → sin data-race */
        }

        /* ── Paso 3: Ordenar (secuencial; fracción pequeña del total) ── */
        qsort(res_table, (size_t)N, sizeof(ResidualEntry), cmp_residual);
        for (int k = 0; k < R; k++) subset[k] = res_table[k].idx;

        /* ── Paso 4: Convergencia ────────────────────────────────────── */
        double delta = 0.0;
        #pragma omp parallel for reduction(+:delta) schedule(static)
        for (int j = 0; j < Dp1; j++) {
            double d = theta[j] - theta_old[j];
            delta += d * d;
        }
        delta = sqrt(delta);

        if (delta < epsilon) { iter++; break; }
    }

    memcpy(theta_out, theta, (size_t)Dp1 * sizeof(double));
    free(theta); free(theta_old); free(subset); free(res_table);
    return iter;
}

/* ─────────────────────────── MSE ───────────────────────────────────────── */
static double compute_mse(const Dataset *ds, const double *theta)
{
    int Dp1 = ds->D + 1;
    double mse = 0.0;

    #pragma omp parallel for reduction(+:mse) schedule(static)
    for (int i = 0; i < ds->N; i++) {
        const double *xrow = ds->X + (size_t)i * Dp1;
        double err = predict(xrow, theta, Dp1) - ds->y[i];
        mse += err * err;
    }
    return mse / ds->N;
}

/* ─────────────────────────── main ───────────────────────────────────────── */
int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr,
            "Uso: %s <dataset.csv> [epsilon] [max_iter] [lr] [bgd_iter]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    const char *path = argv[1];
    double epsilon   = (argc > 2) ? atof(argv[2]) : DEFAULT_EPSILON;
    int    max_iter  = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITER;
    double lr        = (argc > 4) ? atof(argv[4]) : DEFAULT_LR;
    int    bgd_iter  = (argc > 5) ? atoi(argv[5]) : DEFAULT_BGD_ITER;

    /* ── Cargar datos (excluido del tiempo de algoritmo) ───────────────── */
    Dataset ds;
    memset(&ds, 0, sizeof(ds));
    if (load_dataset(path, &ds) != 0) return EXIT_FAILURE;

    int num_threads = omp_get_max_threads();
    int R = (int)(ds.N * 0.8);
    if (R < 2) R = 2;

    printf("=== TRIM Paralelo (OpenMP) ===\n");
    printf("Dataset        : %s\n", path);
    printf("N=%d  D=%d  R=%d\n", ds.N, ds.D, R);
    printf("Hilos OpenMP   : %d\n", num_threads);
    printf("epsilon=%.2e  max_iter=%d  lr=%.2e  bgd_iter=%d\n",
           epsilon, max_iter, lr, bgd_iter);
    fflush(stdout);

    double *theta_out = (double *)calloc((size_t)(ds.D + 1), sizeof(double));
    if (!theta_out) {
        fprintf(stderr, "Error alloc theta_out\n");
        free_dataset(&ds);
        return EXIT_FAILURE;
    }

    /* ── Medir tiempo EXCLUSIVO del algoritmo con omp_get_wtime() ──────── */
    double t0 = omp_get_wtime();

    int iters = trim_parallel(&ds, R, epsilon, max_iter, lr, bgd_iter,
                              theta_out);

    double t1      = omp_get_wtime();
    double elapsed = (t1 - t0) * 1000.0; /* ms */

    double mse = compute_mse(&ds, theta_out);

    printf("\n--- Resultados ---\n");
    printf("Iteraciones TRIM : %d\n", iters);
    printf("Tiempo (ms)      : %.4f\n", elapsed);
    printf("MSE (full data)  : %.6f\n", mse);
    printf("Theta            : [");
    for (int j = 0; j <= ds.D; j++) {
        printf("%.6f", theta_out[j]);
        if (j < ds.D) printf(", ");
    }
    printf("]\n");
    printf("Hilos usados     : %d\n", num_threads);
    fflush(stdout);

    free(theta_out);
    free_dataset(&ds);
    return EXIT_SUCCESS;
}