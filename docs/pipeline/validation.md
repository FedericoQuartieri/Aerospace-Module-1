# Validazione delle modifiche alla pipeline

Base: `5c7aa7b4b2db0fe691acda24f96c6c08bdc9a55d`, verificata nuovamente sul remoto il 16 settembre 2026. Nessun push eseguito.

## Correttezza

**648 confronti completi dei campi superati**, oltre ai test manufactured, ai residui direzionali e alle verifiche degli strumenti.

| Ambiente e variante | Griglie | Confronti | Massima differenza scalata |
|---|---|---:|---:|
| macOS ARM64, double, OpenMP/SIMD | 16³; 17×19×13 | 72 | 0 |
| macOS ARM64, float, batch 3 | 7×9×5 | 12 | 6.56e-7 |
| macOS ARM64, float SIMD, batch 64 | 17×19×13 | 12 | 5.96e-7 |
| macOS ARM64, double, OpenMP senza SIMD | 7×9×5 | 12 | 0 |
| macOS ARM64, AddressSanitizer + UBSan | 7×9×5 | 12 | 2.36e-15 |
| Linux x86_64, 4 rank MPI, double | 8³; 7×9×5 | 288 | 2.00e-15 |
| Linux x86_64, blocchi MPI sottili | 4×5×6 | 132 | 6.19e-15 |
| Linux x86_64, MPI float, batch 3 | 7×9×5 | 48 | 8.42e-7 |
| Linux x86_64, MPI float SIMD, batch 64 | 17×19×13 | 60 | 5.96e-7 |

Riferimento: Schur seriale scalare della stessa revisione. Confronto di eta, zeta, u (tre componenti ciascuno), pressione e pressure_star, cella per cella. La differenza scalata è `abs(a-b)/max(1,abs(a),abs(b))`; soglie 1e-10 in double e 1e-4 in float. I residui delle equazioni hanno soglie 1e-11 e 2e-5 rispettivamente.

Scenari: `paper_variable` con K variabile nello spazio e nel tempo, `zero_pressure`, `constant_forcing`; ulteriori eseguibili manufactured `paper_man`, `zero_pressure`, `constant_forcing_man`. Ogni run usa quattro passi fino a T=0.1.

La matrice macOS principale usa 1/4 thread e batch auto/1/3/64/1024. MPI usa 1/2 thread per rank e le forme 4×1×1, 1×4×1, 1×1×4, 2×2×1, 1×2×2; la matrice double usa batch auto/1/3/64. I casi con un solo punto locale lungo un asse vengono verificati tramite residui e riferimento seriale anche dove Schur distribuito non supporta la forma.

Ulteriori verifiche superate:

- 9 test Python: cambio backend/batch e ritorno alla configurazione precedente, impronta di sorgenti non committate, inizializzazione concorrente di quattro fasi, casi mancanti/falliti/non finiti/forma errata, retry, roundoff e differenze tra campi con uguale norma.
- Test del layout 12³: nessuna differenza tra campi contigui e campi con padding.
- Brinkman, griglia 6×64×6, K=0.01, 200 passi: errore relativo L2 7.3585e-4 per K uniforme e 9.0123e-3 per strato poroso; entrambi sotto il limite 0.05. Verificati anche i default specifici del test dopo la riorganizzazione degli oggetti del Makefile.
- Batch automatico a 4 thread pari a 1024; override runtime a 3 riconosciuto.
- Fase 15 in dry-run: 53 casi enumerati; nessun job PBS inviato.
- Validatore di `run_equivalence.sh`: accetta 16 righe valide anche con norme nulle; rifiuta righe mancanti, NaN e discrepanze.
- Sintassi Bash/Python e `git diff --check`.

La verifica sanitizer usa Clang, SIMD e un processo senza OpenMP; leak detection è disattivata perché gli eseguibili manufactured preesistenti non liberano tutti i campi alla fine del programma. Non sono stati osservati errori AddressSanitizer/UBSan. Non è una verifica ThreadSanitizer.

Le build iniziali nella cartella del progetto e nel container hanno superato alcuni timeout di compilazione. Quei tentativi non sono conteggiati come verifiche superate. Le esecuzioni complete sopra riportate sono state concluse usando copie temporanee e riuso degli oggetti. I 56 file C/header/Makefile della copia macOS sono identici byte per byte alla versione consegnata; la loro impronta è registrata in `validation-macos.json`.

## Confronto locale con la base pulita

GCC 15.2, ARM64, `TRIDIAG=pipeline OMP=1 SIMD=1`, batch automatico, `OMP_DYNAMIC=FALSE`, `OMP_WAIT_POLICY=passive`. Tre ripetizioni alternate per versione, tre passi per esecuzione fino a T=0.1. Si riportano le mediane del tempo per passo e della RSS di picco. La base è una copia pulita di `5c7aa7b`, non il checkout iniziale obsoleto.

| Griglia | Thread | Base ms/passo | Nuovo ms/passo | Rapporto | RSS MiB base → nuova |
|---|---:|---:|---:|---:|---:|
| 64³ | 1 | 33.465 | 23.297 | 1.44× | 46.1 → 34.3 |
| 64³ | 4 | 139.834 | 17.065 | 8.19× | 46.6 → 38.1 |
| 128³ | 1 | 254.937 | 190.904 | 1.34× | 345.4 → 250.1 |
| 128³ | 4 | 1147.901 | 92.657 | 12.39× | 346.2 → 257.7 |

Queste misure sono indicative: macchina non dedicata, poche ripetizioni e variabilità visibile nei dati grezzi. Il confronto con MPI e la taratura sul cluster non sono stati eseguiti. La RSS della base è normalizzata correggendo il suo precedente errore di unità su macOS. Dati grezzi: `benchmark.csv`; parametri e ambiente: `benchmark-host.json`.

## Riproduzione

~~~sh
python3 scripts/test_pipeline_tools.py
CC=gcc-15 OMP=1 SIMD=1 THREADS="1 4" PIPELINE_BATCHES="auto 1 3 64 1024" T_END=0.1 ./scripts/check_pipeline.sh
MPI=1 OMP=1 SIMD=1 THREADS="1 2" RANKS=4 GRIDS="8;7 9 5" PIPELINE_BATCHES="auto 1 3 64" T_END=0.1 MPIEXEC_ARGS="--oversubscribe" OMP_WAIT_POLICY=passive ./scripts/check_pipeline.sh
MPI=1 OMP=1 SIMD=1 THREADS="1 2" RANKS=4 GRIDS="4 5 6" PIPELINE_BATCHES="auto 1 3 64" T_END=0.1 MPIEXEC_ARGS="--oversubscribe" OMP_WAIT_POLICY=passive ./scripts/check_pipeline.sh
MPI=1 OMP=1 SIMD=1 PRECISION=float THREADS="1 2" RANKS=4 GRIDS="17 19 13" PIPELINE_BATCHES="64" T_END=0.1 MPIEXEC_ARGS="--oversubscribe" OMP_WAIT_POLICY=passive ./scripts/check_pipeline.sh
~~~

Il workflow GitHub Actions è stato aggiunto ma non eseguito da remoto: partirà dopo il push. Prima di interpretare nuove campagne di performance sul cluster, eseguire la fase 15 e usare la nuova directory di campagna, senza mescolare le misure storiche.
