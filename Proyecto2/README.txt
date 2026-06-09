========================================================================
PROYECTO FINAL: IMPLEMENTACIÓN Y ANÁLISIS DEL ALGORITMO ROBUSTO TRIM
Materia: Algoritmos Paralelos - Facultad de Ciencias, UNAM
Profesor: Mario Arturo Nieto Butrón
Fecha de Entrega: Viernes 12 de Junio, 2026
Autor: Erick Joel Baltodano Cuevas
========================================================================

Este repositorio contiene la implementación secuencial y paralela (OpenMP)
del algoritmo de defensa robusta TRIM, basado en el artículo científico:
"Manipulating Machine Learning: Poisoning Attacks and Countermeasures 
for Regression Learning" (Jagielski et al.).

El objetivo del proyecto es mitigar ataques de envenenamiento de datos
(Poisoning Attacks) en modelos de regresión lineal y analizar de forma
rigurosa el rendimiento, Speedup y Eficiencia al paralelizar las regiones
más pesadas en cómputo mediante memoria compartida.

────────────────────────────────────────────────────────────────────────
1. ESTRUCTURA DEL DIRECTORIO
────────────────────────────────────────────────────────────────────────

El directorio principal contiene los siguientes componentes esenciales:

├── datasets/      - Carpeta creada automáticamente donde se guardan los
│                    archivos CSV con los datos sintéticos (limpios y 
│                    envenenados) utilizados en los experimentos.
│
├── results/       - Carpeta destino de los datos recopilados de rendimiento.
│                    Contiene los archivos CSV de salida y resúmenes de texto
│                    con las medias aritméticas y desviaciones estándar.
│
├── src/           - Contiene el código fuente puro en lenguaje C:
│   ├── Secuencial.c  -> Versión base monocore que ejecuta TRIM de forma lineal.
│   └── Paralelo.c    -> Versión optimizada con directivas y reducciones OpenMP.
│
├── Script.sh      - Script de automatización en Bash (diseñado para Fedora/Debian)
│                    que compila los códigos con optimizaciones, genera los
│                    datasets sintéticos y corre la suite experimental 5 veces.
│
├── plot_results.py- Script auxiliar en Python encargado de procesar el archivo
│                    CSV de resultados para generar de forma automática las
│                    gráficas reglamentarias de Tiempo vs Hilos y Speedup vs Hilos.
│
└── README.txt     - Este archivo de instrucciones y documentación del sistema.


────────────────────────────────────────────────────────────────────────
2. DETALLE DE LOS COMPONENTES EN 'src/'
────────────────────────────────────────────────────────────────────────

* src/Secuencial.c
  Implementa el bucle jerárquico del algoritmo TRIM. En cada superpaso,
  calcula los parámetros óptimos del modelo (theta) mediante Descenso de
  Gradiente de Lotes (BGD) sobre un conjunto estimado limpio de tamaño R. 
  Posteriormente evalúa los residuos para los N puntos y clasifica los mejores
  mediante Quicksort secuencial. Mide el tiempo de ejecución exclusivo del
  algoritmo utilizando 'clock_gettime(CLOCK_MONOTONIC)'.

* src/Paralelo.c
  Mantiene una equivalencia matemática exacta con la versión secuencial para
  garantizar que no haya divergencia de resultados ni data races. Explota el
  paralelismo multinúcleo aplicando:
  - '#pragma omp parallel for private(i)' para el cálculo masivo e independiente
    de los residuos de los N puntos.
  - '#pragma omp parallel for reduction(+:grad[:D+1])' para resolver de forma
    concurrente y segura las acumulaciones vectoriales en el descenso de gradiente.
  Mide el tiempo de ejecución mediante 'omp_get_wtime()'.


────────────────────────────────────────────────────────────────────────
3. CÓMO EJECUTAR EL PROYECTO (PASO A PASO)
────────────────────────────────────────────────────────────────────────

Para replicar las pruebas experimentales y generar todas las métricas de la 
rúbrica de manera automatizada y segura, sigue estos pasos desde tu terminal
en Fedora o Debian:

Paso 1: Dar permisos de ejecución al script principal
$ chmod +x Script.sh

Paso 2: Ejecutar el script de automatización
$ ./Script.sh

Este comando realizará las siguientes tareas de forma transparente:
  1. Creará los directorios necesarios ('src/src', 'datasets', 'results').
  2. Compilará automáticamente ambos códigos C usando GCC con el flag 
     de optimización agresiva '-O3' (y la bandera '-fopenmp' para el paralelo).
  3. Generará de forma aleatoria 3 conjuntos de datos de escalas crecientes
     (N=5000, N=20000, N=80000) con un 20% de datos envenenados.
  4. Ejecutará 5 veces consecutivas la versión secuencial y la paralela 
     variando el número de hilos de OpenMP (1, 2, 4 y 8 hilos), incluyendo
     pausas térmicas (sleep) y límites de tiempo (timeout) para evitar
     congelamientos de la CPU en el procesador AMD 3020e.

Paso 3: Visualizar los resultados en consola y archivos
Al finalizar, podrás ver un resumen detallado en pantalla con la Media y
la Desviación Estándar. Todos los datos duros quedarán guardados de forma
permanente dentro de la carpeta 'results/' en un formato estructurado CSV.

Paso 4: Generar las gráficas obligatorias del reporte
Para obtener de manera automática las curvas de Tiempo vs Hilos y Speedup vs Hilos,
ejecuta el script de Python:
$ python3 plot_results.py


────────────────────────────────────────────────────────────────────────
4. REQUISITOS DEL SISTEMA Y DEPENDENCIAS
────────────────────────────────────────────────────────────────────────
- Sistema Operativo: GNU/Linux (Probado con éxito en Fedora Workstation y Debian).
- Compilador: GCC (Gnu Compiler Collection) con soporte nativo para OpenMP.
- Bibliotecas estándar de C: math.h, time.h, stdio.h, stdlib.h, string.h.
- Para las gráficas (Opcional): Python 3 con los módulos 'matplotlib' y 'pandas'.
========================================================================
