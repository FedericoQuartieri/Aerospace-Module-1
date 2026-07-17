# Navier–Stokes–Brinkman solver

Il progetto implementa lo stesso metodo numerico con due backend selezionati a
compile time:

- `solver_standard`: kernel scalari di riferimento;
- `solver_optimized`: kernel ottimizzati con lo stesso contratto numerico.

La precisione predefinita è `double`. I campi vettoriali usano un layout SoA e
la direzione X è contigua in memoria.

La descrizione completa delle formule, dei sistemi di Thomas, degli accessi in
memoria e delle implementazioni `STANDARD`/`OPTIMIZED` è disponibile in
[`docs/kernel-design.md`](docs/kernel-design.md). L'architettura generale è
descritta in [`docs/solver-design.md`](docs/solver-design.md).

## Configurazione e build

Dalla radice del repository:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
```

Per cambiare i valori predefiniti incorporati negli eseguibili:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGRID_WIDTH=64 -DGRID_HEIGHT=64 -DGRID_DEPTH=64 \
  -DTIME_STEP=0.0007 -DSIM_TOTAL_TIME=0.007 \
  -DSOLVER_PRECISION=DOUBLE
cmake --build build --parallel 4
```

`SOLVER_PRECISION` accetta `DOUBLE` oppure `FLOAT`. Dopo aver modificato una
variabile CMake è sufficiente riconfigurare e ricompilare la stessa directory di
build.

È disponibile anche il Makefile:

```sh
make -j4
make test
make test-convergence
```

CMake/CTest resta il percorso consigliato perché espone separatamente tutti i
gruppi di test.

## Eseguire una simulazione

I parametri passati sulla riga di comando sostituiscono i valori predefiniti
della build:

```sh
./build/solver_standard  --grid 64 --dt 0.0007 --steps 10
./build/solver_optimized --grid 64 --dt 0.0007 --steps 10
```

Le opzioni disponibili sono:

```text
--grid N                 usa una griglia cubica N x N x N
--dt DT                  imposta il passo temporale
--steps N                imposta il numero di timestep
--output-frequency N     scrive ogni N timestep; 0 disabilita l'output
--output-directory PATH  imposta la directory dei file VTI
```

Il tempo finale è `steps * dt`. La CLI accetta solo griglie cubiche; usando
l'API C si possono assegnare separatamente i tre elementi di `config.extent`.
La spaziatura staggered è calcolata come `2L / (2N - 1)`.

Al termine l'eseguibile stampa backend, timestep completati e statistiche
temporali, inclusi microsecondi e nanosecondi per timestep e per cella. Il tempo
di output è riportato separatamente dal tempo di calcolo e non entra nella
metrica normalizzata.

## Test di correttezza

I test di correttezza eseguono i casi manufactured `paper`, `zero-pressure` e
`variable-permeability` su entrambi i backend. Verificano:

- norme L2 e Linf della velocità;
- norme L2 e Linf della pressione;
- norma L2 della divergenza;
- errore Linf al bordo;
- assenza di valori non finiti, inizializzazione e modello di memoria.

Per eseguirli tramite CTest:

```sh
cmake --build build \
  --target test_correctness_standard test_correctness_optimized \
  --parallel 4
ctest --test-dir build -L correctness --output-on-failure
```

Un test superato produce la riga `Passed`; `--output-on-failure` mostra norme e
backend soltanto in caso di errore. Per vedere sempre i valori numerici:

```sh
./build/test_correctness_standard --verbose
./build/test_correctness_optimized --verbose
```

Ogni caso deve stampare `[PASS]`. Inoltre il comando deve terminare con codice
zero, verificabile in una shell con:

```sh
echo $?
```

La configurazione di riferimento dei test è definita da
`BASE_CONFIG_INITIALIZER` in `test/manufactured_cases.c`: 64³, `dt=0.0007` e
10 step. Le soglie si trovano nello stesso file, dentro ogni
`ManufacturedCase`. Se si cambia griglia o `dt` per un nuovo esperimento occorre
modificare quella configurazione e ricompilare; non si devono allentare le
soglie solo per ottenere un test verde. I test del progetto non devono superare
64 punti per direzione.

### Test tunnel e cavity

[`test/test_variable_permeability_cavity.c`](test/test_variable_permeability_cavity.c)
contiene due casi:

- `tunnel`: il flusso uniforme attraversa il volume e incontra un ostacolo
  sferico a bassa permeabilità;
- `cavity`: un cubo unitario con apertura quadrata centrata in ingresso; il
  condotto conserva la sezione dell'apertura fino a metà cubo, quindi si allarga
  linearmente verso un'apertura quadrata più grande in uscita.

Nel caso `cavity`, il condotto interno ha permeabilità `1`, mentre il volume
esterno ha permeabilità `0.002`. La velocità uniforme in uscita viene ridotta in
proporzione al rapporto fra le aree, mantenendo la portata. Pressione iniziale e
forzante sono nulle in entrambi i casi.

`tunnel` usa una griglia 48x32x32; `cavity` usa una griglia cubica 32x32x32.
Entrambi eseguono 2000 timestep da `0.001` e scrivono uno snapshot ogni 5 step
nella directory `output`, incluso `solution_000000.vti`, quando sono avviati
direttamente. CTest aggiunge `--no-output` per verificare permeabilità,
condizioni al bordo, valori finiti e attraversamento del condotto senza creare
snapshot:

```sh
cmake --build build \
  --target test_variable_permeability_cavity_standard \
           test_variable_permeability_cavity_optimized \
  --parallel 4
ctest --test-dir build -R variable_permeability --output-on-failure
```

Lo stesso eseguibile seleziona il caso con `--case`:

```sh
./build/test_variable_permeability_cavity_standard --case tunnel
./build/test_variable_permeability_cavity_standard --case cavity
./build/test_variable_permeability_cavity_optimized --case tunnel
./build/test_variable_permeability_cavity_optimized --case cavity
```

Senza opzioni, oppure con `--case all`, vengono eseguiti prima `tunnel` e poi
`cavity`. Un'esecuzione superata non stampa nulla e termina con codice zero.

## Test di convergenza

La convergenza spaziale usa 16³, 32³ e 64³ con `dt=0.0007` e 10 step. L'ordine
usato per il pass/fail è quello della coppia 32→64:

- velocità L2 combinata: ordine minimo `1.70`;
- pressione L2: ordine minimo `1.95`.

La convergenza temporale usa una griglia 64³, tempo finale 0.5 e
`dt={0.1, 0.05, 0.025, 0.0125, 0.00625}`. Le coppie verificate sono
0.05→0.025 e 0.025→0.0125, con ordini minimi 1.80 per la velocità e 1.35 per la
pressione.

Per eseguire entrambi gli studi con CTest:

```sh
cmake --build build --target test_convergence --parallel 4
ctest --test-dir build -L convergence --output-on-failure
```

Per stampare errori e ordini misurati:

```sh
./build/test_convergence --spatial --verbose
./build/test_convergence --temporal --verbose
```

Senza selezionare una modalità vengono eseguiti entrambi:

```sh
./build/test_convergence --verbose
```

La tabella mostra griglia o `dt`, errore L2 combinato della velocità, errore L2
della pressione, divergenza e errore al bordo. L'ultima riga mostra gli ordini
misurati e le soglie minime: gli errori devono diminuire e ciascun ordine deve
essere maggiore o uguale alla propria soglia.

Lo script seguente configura una build dedicata, compila il test e lo esegue:

```sh
python3 run_convergence_study.py \
  --build-dir /tmp/nsb-convergence \
  --mode all --build-type Release --verbose
```

`--mode` accetta `spatial`, `temporal` oppure `all`.

I parametri dello studio sono intenzionalmente espliciti in
`test/test_convergence.c`:

- modificare `extents[]` per le griglie spaziali;
- modificare `config.dt` e `config.steps` in `spatial_study` per passo e durata;
- modificare `dt[]`, `TEMPORAL_GRID_EXTENT` e il valore `0.5` in
  `temporal_study` per lo studio temporale.

Dopo una modifica ricompilare `test_convergence`. Conservare almeno tre livelli
di raffinamento e non usare griglie oltre 64³ nei test.

## Benchmark dei kernel X

Il benchmark prestazionale è separato da CTest: non è un test numerico e può
usare 128³, mentre correttezza e convergenza restano limitate a 64³. Compilare
i due probe con:

```sh
cmake --build build \
  --target benchmark_x_standard benchmark_x_optimized \
  --parallel 4
```

Un'esecuzione singola stampa CSV con il costo del momentum X, della pressione X
e dell'intero timestep in nanosecondi per step e per cella:

```sh
./build/benchmark_x_optimized \
  --grid 128 --warmup 2 --steps 10 --workload paper
```

`--workload` accetta `paper` oppure `synthetic`. Il secondo usa callback
algebriche leggere per rendere più visibile il costo del solver tridiagonale.
Il warmup modifica normalmente lo stato ma viene escluso dalle misure; l'output
VTI è sempre disabilitato.

Per valutare una modifica confrontare l'eseguibile candidato con un eseguibile
ottimizzato congelato prima della modifica:

```sh
python3 benchmark_x.py \
  --baseline /percorso/baseline/benchmark_x_optimized \
  --candidate build/benchmark_x_optimized \
  --target momentum --repeats 5 \
  --csv benchmark-x-raw.csv
```

In alternativa si possono usare `benchmark_x_standard` e
`benchmark_x_optimized` della stessa build per confrontare i backend. Lo script
alterna l'ordine delle esecuzioni, riporta mediana e MAD relativa su 64³ e 128³
e termina con `gate=PASS` soltanto se:

- il rumore massimo è al più 3%;
- la metrica X scelta migliora almeno del 15% a 128³ su entrambi i workload;
- il timestep paper migliora almeno del 5% a 128³;
- metrica scelta e timestep non regrediscono oltre il 2% a 64³.

Il comando restituisce codice 0 quando il gate passa e 2 quando fallisce. Non
sono previste esecuzioni a 256³. Motivazioni, varianti scartate e misure della
soluzione adottata sono raccolte in `docs/x-kernel-performance.md`.

## Tutti i test e le etichette CTest

Per eseguire l'intera suite:

```sh
ctest --test-dir build --output-on-failure
```

I gruppi possono essere selezionati con `-L`:

```sh
ctest --test-dir build -L correctness --output-on-failure
ctest --test-dir build -L kernel      --output-on-failure
ctest --test-dir build -L convergence --output-on-failure
ctest --test-dir build -L output      --output-on-failure
ctest --test-dir build -L stats       --output-on-failure
ctest --test-dir build -L sanitizer   --output-on-failure
```

I test numerici forzano sempre `output_frequency=0`: non producono snapshot e
non includono I/O nelle misure. Il writer è verificato separatamente dal test
con etichetta `output`.

## Scrivere l'output VTK/VTI

Il solver scrive file `.vti`, cioè il formato XML ImageData di VTK. Non produce
il vecchio formato legacy `.vtk`. Per abilitare gli snapshot:

```sh
./build/solver_optimized \
  --grid 64 --dt 0.0007 --steps 10 \
  --output-frequency 5 \
  --output-directory output
```

Il comando crea la directory `output` se necessario e scrive, in modo sincrono:

```text
output/solution_000000.vti
output/solution_000005.vti
output/solution_000010.vti
```

Quando l'output è abilitato, `solution_000000.vti` viene scritto sempre e
contiene la condizione iniziale a `t=0`. La frequenza si applica ai timestep
successivi: con `--output-frequency 1` viene scritto ogni step; con 0 l'output è
completamente disabilitato. La directory padre deve già esistere se il percorso
richiesto è annidato. Un errore di apertura o scrittura termina la solve con
`SOLVER_OUTPUT_ERROR` e un codice di uscita diverso da zero.

Ogni file contiene:

- `Pressure`, array scalare sul reticolo di pressione;
- `Velocity`, array a tre componenti ricavato dai tre campi SoA;
- `TimeValue`, tempo della velocità: `0` nel file iniziale, poi `n*dt`;
- `PressureTime`, tempo della pressione: `0` nel file iniziale, poi
  `(n-1/2)*dt`.

I file possono essere aperti direttamente con ParaView usando **File → Open**.
Per controllare rapidamente che il file sia stato creato e contenga gli array:

```sh
ls -lh output/*.vti
strings output/solution_000000.vti | grep 'Name='
```

Da un programma C l'output si abilita nella configurazione prima di
`solver_init`:

```c
SolverConfig config = solver_default_config();

config.extent[DIRECTION_X] = 64;
config.extent[DIRECTION_Y] = 64;
config.extent[DIRECTION_Z] = 64;
config.dt = (Real)0.0007;
config.steps = 10;
config.output_frequency = 5;
config.output_directory = "output";
```

La serializzazione è implementata in `src/output.c`. Il buffer per trasformare
la velocità da SoA a tuple XYZ viene allocato una sola volta in `solver_init`;
non avvengono allocazioni durante i timestep.
