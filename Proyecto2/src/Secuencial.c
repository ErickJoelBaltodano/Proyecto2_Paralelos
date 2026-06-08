#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ─────────────────────────── Parámetros por defecto ─────────────────────── */
#define DEFAULT_EPSILON   1e-6   /* Umbral de convergencia de TRIM           */
#define DEFAULT_MAX_ITER  200    /* Iteraciones máximas de TRIM              */
#define DEFAULT_LR        1e-4   /* Tasa de aprendizaje para BGD             */
#define DEFAULT_BGD_ITER  500    /* Pasos de BGD por iteración de TRIM       */
#define MAX_FEATURES      256    /* Dimensionalidad máxima soportada         */
#define MAX_LINE          65536  /* Longitud máxima de una línea CSV         */

/* ─────────────────────────── Estructuras ────────────────────────────────── */
typedef struct {
    double *X;   /* Matriz de características [N × (D+1)], columna 0 = 1   */
    double *y;   /* Vector de etiquetas [N]                                  */
    int     N;   /* Número total de puntos                                   */
    int     D;   /* Número de características (sin el sesgo)                 */
} Dataset;

typedef struct {
    double residual;
    int    idx;
} ResidualEntry;

/* ─────────────────────────── Utilidades ─────────────────────────────────── */
static double elapsed_ms(struct timespec t0, struct timespec t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0
         + (t1.tv_nsec - t0.tv_nsec) / 1.0e6;
}

/* Comparador para qsort (orden ascendente de residuo) */
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

    /* Primera pasada: contar líneas y detectar dimensionalidad */
    char line[MAX_LINE];
    int  n_lines = 0;
    int  D_det   = -1;

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;

        if (D_det < 0) {
            /* Detectar número de columnas en la primera línea de datos */
            int cols = 0;
            char *tmp = strdup(line);
            char *tok = strtok(tmp, ",\t ");
            while (tok) { cols++; tok = strtok(NULL, ",\t "); }
            free(tmp);
            if (cols < 2) {
                fprintf(stderr, "Error: se necesitan al menos 2 columnas.\n");
                fclose(fp);
                return -1;
            }
            D_det = cols - 1; /* Última columna = y */
        }
        n_lines++;
    }
    rewind(fp);

    if (n_lines == 0) {
        fprintf(stderr, "Error: dataset vacío.\n");
        fclose(fp);
        return -1;
    }
    if (D_det > MAX_FEATURES) {
        fprintf(stderr, "Error: demasiadas características (%d > %d).\n",
                D_det, MAX_FEATURES);
        fclose(fp);
        return -1;
    }

    ds->N = n_lines;
    ds->D = D_det;

    /* Allocate: columna 0 reservada para el término de sesgo (bias = 1) */
    ds->X = (double *)malloc((size_t)ds->N * (ds->D + 1) * sizeof(double));
    ds->y = (double *)malloc((size_t)ds->N * sizeof(double));
    if (!ds->X || !ds->y) {
        fprintf(stderr, "Error: memoria insuficiente.\n");
        fclose(fp);
        return -1;
    }

    /* Segunda pasada: leer valores */
    int row = 0;
    while (fgets(line, sizeof(line), fp) && row < ds->N) {
        if (line[0] == '#' || line[0] == '\n') continue;

        double *xrow = ds->X + (size_t)row * (ds->D + 1);
        xrow[0] = 1.0; /* Término de sesgo */

        char *tok = strtok(line, ",\t \n");
        for (int c = 1; c <= ds->D && tok; c++) {
            xrow[c] = atof(tok);
            tok = strtok(NULL, ",\t \n");
        }
        ds->y[row] = tok ? atof(tok) : 0.0;
        row++;
    }
    fclose(fp);
    ds->N = row; /* Ajuste si hubo líneas vacías */
    return 0;
}

static void free_dataset(Dataset *ds)
{
    free(ds->X);
    free(ds->y);
    ds->X = NULL;
    ds->y = NULL;
}

/* ─────────────────────────── Predicción puntual ─────────────────────────── */
static inline double predict(const double *xrow, const double *theta, int Dp1)
{
    double s = 0.0;
    for (int j = 0; j < Dp1; j++) s += xrow[j] * theta[j];
    return s;
}

/* ─────────────────────────── BGD sobre subconjunto ─────────────────────── */
/*
 * Ejecuta Batch Gradient Descent sobre el subconjunto indicado por `subset`
 * (arreglo de índices de longitud R).
 * Actualiza theta in-place.
 */
static void bgd(const Dataset *ds,
                const int     *subset,
                int            R,
                double        *theta,
                double         lr,
                int            bgd_iter)
{
    int Dp1 = ds->D + 1;
    double *grad = (double *)calloc((size_t)Dp1, sizeof(double));
    if (!grad) { fprintf(stderr, "Error alloc grad\n"); return; }

    for (int it = 0; it < bgd_iter; it++) {
        /* Calcular gradiente sobre el subconjunto */
        memset(grad, 0, (size_t)Dp1 * sizeof(double));

        for (int s = 0; s < R; s++) {
            int i = subset[s];
            const double *xrow = ds->X + (size_t)i * Dp1;
            double err = predict(xrow, theta, Dp1) - ds->y[i];
            for (int j = 0; j < Dp1; j++)
                grad[j] += err * xrow[j];
        }

        /* Actualizar theta */
        double scale = lr / R;
        for (int j = 0; j < Dp1; j++)
            theta[j] -= scale * grad[j];
    }
    free(grad);
}

/* ─────────────────────────── Algoritmo TRIM ─────────────────────────────── */
/*
 * Implementa el Algoritmo 2 de Jagielski et al.
 *
 * Parámetros:
 *   ds       - Dataset completo
 *   R        - Tamaño del subconjunto limpio estimado (= N * fraccion_limpia)
 *   epsilon  - Umbral de convergencia (diferencia en theta)
 *   max_iter - Máximo de iteraciones
 *   lr, bgd_iter - Hiperparámetros de BGD
 *   theta_out - Salida: coeficientes estimados [D+1]
 *
 * Retorna: número de iteraciones realizadas
 */
static int trim(const Dataset *ds,
                int            R,
                double         epsilon,
                int            max_iter,
                double         lr,
                int            bgd_iter,
                double        *theta_out)
{
    int N   = ds->N;
    int Dp1 = ds->D + 1;

    /* Inicializar theta en cero */
    double *theta     = (double *)calloc((size_t)Dp1, sizeof(double));
    double *theta_old = (double *)calloc((size_t)Dp1, sizeof(double));
    if (!theta || !theta_old) {
        fprintf(stderr, "Error alloc theta\n");
        free(theta); free(theta_old);
        return -1;
    }

    /* Subconjunto inicial: primeros R puntos */
    int *subset = (int *)malloc((size_t)R * sizeof(int));
    if (!subset) {
        fprintf(stderr, "Error alloc subset\n");
        free(theta); free(theta_old);
        return -1;
    }
    for (int i = 0; i < R; i++) subset[i] = i;

    /* Tabla de residuos para todos los N puntos */
    ResidualEntry *res_table = (ResidualEntry *)malloc(
                                   (size_t)N * sizeof(ResidualEntry));
    if (!res_table) {
        fprintf(stderr, "Error alloc res_table\n");
        free(theta); free(theta_old); free(subset);
        return -1;
    }

    int iter = 0;
    for (iter = 0; iter < max_iter; iter++) {
        /* Guardar theta anterior */
        memcpy(theta_old, theta, (size_t)Dp1 * sizeof(double));

        /* ── Paso 1: Estimar theta con BGD sobre el subconjunto actual ── */
        bgd(ds, subset, R, theta, lr, bgd_iter);

        /* ── Paso 2: Calcular residuos para TODOS los N puntos ────────── */
        for (int i = 0; i < N; i++) {
            const double *xrow = ds->X + (size_t)i * Dp1;
            double err = predict(xrow, theta, Dp1) - ds->y[i];
            res_table[i].residual = err * err;
            res_table[i].idx      = i;
        }

        /* ── Paso 3: Seleccionar los R puntos con menor residuo ────────── */
        qsort(res_table, (size_t)N, sizeof(ResidualEntry), cmp_residual);
        for (int k = 0; k < R; k++) subset[k] = res_table[k].idx;

        /* ── Paso 4: Verificar convergencia ────────────────────────────── */
        double delta = 0.0;
        for (int j = 0; j < Dp1; j++) {
            double d = theta[j] - theta_old[j];
            delta += d * d;
        }
        delta = sqrt(delta);

        if (delta < epsilon) {
            iter++;   /* Contar la iteración que convergió */
            break;
        }
    }

    memcpy(theta_out, theta, (size_t)Dp1 * sizeof(double));

    free(theta);
    free(theta_old);
    free(subset);
    free(res_table);

    return iter;
}

/* ─────────────────────────── Cálculo de MSE ─────────────────────────────── */
static double compute_mse(const Dataset *ds, const double *theta)
{
    int Dp1 = ds->D + 1;
    double mse = 0.0;
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

    const char *path     = argv[1];
    double epsilon       = (argc > 2) ? atof(argv[2]) : DEFAULT_EPSILON;
    int    max_iter      = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITER;
    double lr            = (argc > 4) ? atof(argv[4]) : DEFAULT_LR;
    int    bgd_iter      = (argc > 5) ? atoi(argv[5]) : DEFAULT_BGD_ITER;

    /* ── Cargar datos (NO incluido en el tiempo de algoritmo) ──────────── */
    Dataset ds;
    memset(&ds, 0, sizeof(ds));
    if (load_dataset(path, &ds) != 0) return EXIT_FAILURE;

    printf("=== TRIM Secuencial ===\n");
    printf("Dataset : %s\n", path);
    printf("N=%d  D=%d  R=N*0.8=%d\n", ds.N, ds.D, (int)(ds.N * 0.8));
    printf("epsilon=%.2e  max_iter=%d  lr=%.2e  bgd_iter=%d\n",
           epsilon, max_iter, lr, bgd_iter);
    fflush(stdout);

    int R = (int)(ds.N * 0.8); /* 80 % del dataset como estimado limpio */
    if (R < 2) R = 2;

    double *theta_out = (double *)calloc((size_t)(ds.D + 1), sizeof(double));
    if (!theta_out) {
        fprintf(stderr, "Error alloc theta_out\n");
        free_dataset(&ds);
        return EXIT_FAILURE;
    }

    /* ── Medir tiempo EXCLUSIVO del algoritmo ──────────────────────────── */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int iters = trim(&ds, R, epsilon, max_iter, lr, bgd_iter, theta_out);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = elapsed_ms(t0, t1);

    /* ── Calcular MSE final ─────────────────────────────────────────────── */
    double mse = compute_mse(&ds, theta_out);

    /* ── Reportar resultados ────────────────────────────────────────────── */
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
    fflush(stdout);

    free(theta_out);
    free_dataset(&ds);
    return EXIT_SUCCESS;
}