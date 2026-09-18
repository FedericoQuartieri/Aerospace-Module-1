# Profiling della pipeline del solver: risultati reali

Verifica sul cluster Polimi, 17 settembre 2026, account `u10830290`, branch
`unified` al commit `70e9233` ("ready for cluster"). Questo documento riporta
i risultati **effettivamente ottenuti** eseguendo il workflow descritto in
[pipeline/README.md](pipeline/README.md) sul solver vero (non piu' il
programma di controllo delle verifiche precedenti). Distingue esplicitamente
tre categorie:

- **Misura**: numero osservato direttamente (timer interni del solver,
  campionamento nativo gprofng, chiamate MPI contate da mpiP, byte heap di
  Massif).
- **Simulazione**: numero prodotto da un modello (istruzioni/cache/branch di
  Callgrind: l'esecuzione e' reale, i miss di cache e i mispredict sono
  calcolati da un modello di CPU generico, non dai contatori hardware dello
  Xeon Gold 6238R).
- **Interpretazione**: lettura o ipotesi che collega piu' misure; e' quella
  piu' esposta a essere sbagliata e va marcata come tale.

Non ripete i risultati gia' verificati in
[PROFILING_SENZA_ROOT.md](PROFILING_SENZA_ROOT.md) (Callgrind, Massif,
gprofng, mpiP funzionano senza privilegi amministrativi; perf/LIKWID restano
bloccati su `cpu01` da `perf_event_paranoid=4`): qui si applicano al solver.

## 0. Due errori trovati e corretti durante l'esecuzione

Onesta' sul processo, non solo sul risultato:

1. **Callgrind**: il primo tentativo di isolare `momentum_step`/`pressure_step`
   usava `--instr-atstart=no`, che spegne l'istrumentazione stessa (non solo
   la raccolta). Risultato: 0 eventi raccolti in entrambi i casi, silenziosamente,
   nessun errore. Corretto in `--collect-atstart=no --toggle-collect=<funzione>`,
   che tiene l'istrumentazione sempre attiva e isola solo la raccolta. Rifatto
   con successo (job `32601`).
2. **gprofng**: il primo tentativo per l'esperimento MPI+OMP passava
   `-o mpi.er` esplicito; sotto `mpirun -np 2` i due rank hanno provato a
   scrivere nella stessa directory, il secondo ha fallito
   ("in use and cannot be updated") e mpirun ha abortito tutto il job,
   lasciando un profilo quasi vuoto (solo l'inizializzazione MPI, 0.003s).
   Corretto omettendo `-o`: gprofng nomina da solo per rank
   (`test.1.er`, `test.2.er`), come nel probe gia' validato in precedenza.
   Rifatto con successo (job `32603`).

Entrambi gli script in [pipeline/](pipeline/) sono stati corretti per le
prossime esecuzioni; i log grezzi di prima e dopo la correzione sono in
[pipeline/results/](pipeline/results/).

## 1. Correttezza e overhead (job 32591)

- **Misura**: `scripts/check_pipeline.py` (MPI=1 OMP=1 SIMD=1, 4 rank, 2
  thread, griglie 24³ e 17×19×13) — **PASSED, 150 confronti campo-intero,
  differenza scalata massima 1.998e-15** contro il riferimento seriale Schur.
- **Misura**: `bench BENCH_NORMS=1`, stesso caso (64³, 50 passi, 4 rank),
  build release vs build debug (`-g -fno-omit-frame-pointer -no-pie`):
  norme L2 **identiche bit per bit** (`u_x 5.61618744760938524e-04` etc. in
  entrambe le build). Le flag di debug non alterano la codegen numerica.
- **Interpretazione, non misura pulita**: il tempo per passo e' cambiato
  parecchio fra le due build (release 193.3 ms/passo, 62.1% in MPI; debug
  134.8 ms/passo, 54.7% in MPI) — ma nella direzione opposta a quella
  attesa (la build "piu' pesante" e' risultata piu' veloce). Su un nodo
  condiviso una singola esecuzione A/B non basta a isolare l'effetto delle
  flag dal rumore di contesa.

**Aggiornamento (job 32606, 5 repliche per build, stesso caso)**: il tempo
per passo oscilla fra 65.7 ms e 812.9 ms **all'interno della stessa identica
build**, un fattore ~12x dovuto al solo rumore del nodo condiviso (`mpi per
step` passa dal 24% al 67% del tempo fra repliche identiche). Le medie sulle
5 repliche sono pero' quasi indistinguibili: **release 434.7 ms, debug
424.8 ms**. Conclusione: **nessun overhead sistematico misurabile delle flag
di debug** — la variabilita' osservata prima era rumore di contesa, non un
effetto delle flag. Punto chiuso.

Nello stesso job, `BENCH_NORMS=1` sul caso "native" (128³, 100 passi, build
debug) non era mai stato eseguito: **misura**, norme finite e piu' piccole
di quelle a 64³ (`u_x` 1.06e-04, `u_y` 1.59e-04, `u_z` 1.44e-04, `p` 5.19e-03
contro 5.6e-04/8.4e-04/7.1e-04/2.0e-02 a 64³), coerenti con la convergenza
attesa a griglia piu' fine. Punto chiuso.

## 2. Timeline nativa della pipeline (misura, gia' presente nel codice)

`src/utils.c`/`include/types.h` misurano gia' a runtime, per ogni passo, il
tempo di parete di ciascuna fase. Tre configurazioni molto diverse mostrano
quanto la ripartizione dipenda dalla configurazione, non solo dalla fisica:

| Fase | 32³, 1 rank, 1 thread | 64³, 4 rank, 2 thread (release) | 128³, 2 rank, 2 thread |
|---|---|---|---|
| eta (g term) | 41.9% (17.4%) | 24.4% (2.9%) | 74.3% (10.1%) |
| zeta | 22.3% | 37.1% | 3.4% |
| u | 27.2% | 1.5% | 4.5% |
| psi | 2.9% | 8.9% | 9.8% |
| phi low/high | 2.6% / 2.7% | 17.1% / 0.5% | 2.0% / 2.2% |
| pressure | 0.3% | 0.1% | 0.3% |
| unaccounted | 0.0% | 23.1% | 4.2% |
| mpi per step | 0.0% | **62.1%** | 26.6% |

**Interpretazione**: a 4 rank su una griglia 64³ (blocco locale 32×32×64) il
solutore e' dominato dalla comunicazione (62.1% del tempo), non dal calcolo:
qualunque ottimizzazione dei kernel numerici a questa configurazione avrebbe
un ritorno limitato. Il 23.1% di "unaccounted" a 4 rank (contro ~0% a 1 rank)
segnala inoltre che con piu' processi qualcosa non viene cronometrato dai
timer interni (probabile sincronizzazione implicita non strumentata) — non e'
un errore, ma un limite di visibilita' dei timer attuali da tenere presente.

## 3. Callgrind: istruzioni, accessi, cache/branch simulati (job 32591 + 32601 fix)

Caso piccolo, 32³, 1 rank, 1 thread, backend schur, senza SIMD.

**Misura** (esecuzione intera): 589,121,195 istruzioni, D1 miss rate 1.6%,
LL miss rate 0.0%, branch mispredict 3.4%. Norme identiche al run nativo.

**Misura, isolata per funzione** (`--collect-atstart=no --toggle-collect=`):

| | momentum_step | pressure_step |
|---|---|---|
| Istruzioni | 399,208,723 (67.8% del totale) | 25,887,964 (4.4% del totale) |
| D1 miss rate | 2.0% | **3.7%** |
| Branch mispredict | 3.5% | 3.1% |

Nonostante pesi 15x meno in istruzioni, la fase di pressione ha un tasso di
D1 miss quasi doppio: **simulazione**, non contatore hardware, ma coerente
con un accesso meno regolare alla memoria in quella fase.

**Simulazione, per funzione** (`callgrind_annotate`):
- `momentum_step`: `__cos_fma`+`__sin_fma` (libm, valutazione del termine
  forzante) = **29.3%** delle istruzioni; `momentum_assemble_line` 14.4%;
  `thomas_solve` 11.5% ma **33.35%/18.81%** dei D1 miss di lettura/scrittura.
- `pressure_step`: `thomas_solve` **59.2%** delle istruzioni e **48.46%**
  dei branch mispredict della fase; `pressure_gather_line`/`scatter_line`
  ~11.5% ciascuna; `compute_div` solo 8.3% delle istruzioni ma **93.68%**
  dei DLmr (LL read miss) della fase.

**Interpretazione**: `thomas_solve` e' il singolo hotspot piu' costoso in
entrambe le fasi ed e' anche la fonte principale di cache miss simulati; il
termine forzante trigonometrico (`g term`, gia' isolato nei timer nativi)
pesa quasi un terzo delle istruzioni della quantita' di moto, confermando
quantitativamente quanto il commento in `include/types.h` gia' segnalava
qualitativamente.

**Compatibilita' Valgrind/AVX2 (verificata, era un punto aperto)**: stesso
caso con `SIMD=1`, Callgrind gira senza errori, 458,265,788 istruzioni
(-22% rispetto alla versione scalare) e **norme bit-identiche**. Il punto
"la compatibilita' della build SIMD con Valgrind va ancora provata" in
PROFILING_SENZA_ROOT.md e' chiuso.

## 4. Massif: allocazioni reali (job 32591)

**Misura**: picco heap 5.664 MB (32³, backend schur). Il picco pero' e'
**dominato dall'allocazione temporanea di `compute_solver_error_norms`**
(il calcolo delle norme di `BENCH_NORMS=1`), identico byte per byte
(5,664,352 B) in tutte le configurazioni testate — un artefatto della
scelta di misurare con le norme attive, non una proprieta' del backend.

Guardando lo snapshot **durante** il ciclo di soluzione (prima del calcolo
delle norme), l'effetto del batch del backend pipeline diventa visibile:

| PIPELINE_BATCH_LINES | Heap durante il solve |
|---|---|
| 1 | 4,722,880 B |
| 8 | 4,726,592 B |
| 32 | 4,739,456 B |

Crescita reale e monotona (+16.576 B, +0.35%, da batch 1 a 32) ma modesta a
questa taglia di griglia. Lo scratch del backend schur (le matrici Schur
fattorizzate) non compare mai sopra la soglia dell'1% di ms_print a questa
taglia: e' piccolo rispetto alla sola memoria dei campi, coerente con
l'idea che lo Schur complement tenga solo l'interfaccia, non l'intero
blocco.

**Aggiornamento (job 32607, griglia 64³, senza `BENCH_NORMS`)**: rimossa la
maschera dell'allocazione delle norme, il picco vero emerge presto
nell'esecuzione (durante `backend_init`/primo passo, non alla fine) ed e'
molto piu' leggibile:

| Configurazione | Picco heap (B) |
|---|---|
| schur | 34,504,424 |
| pipeline, batch=1 | 34,509,776 |
| pipeline, batch=8 | 34,517,088 |
| pipeline, batch=32 | 34,542,240 |
| pipeline, batch=128 | 34,642,848 |

**Misura**: crescita reale e monotona di 133.072 B (+0.39%) da batch=1 a
batch=128, circa 8 volte piu' netta in valore assoluto che a 32³. Il picco
di schur resta quasi identico al picco di pipeline a batch minimo
(+5.352 B, cioe' pipeline con batch=1): conferma che, a questa taglia di
griglia, lo scratch specifico di **entrambi** i backend e' piccolo rispetto
alla sola memoria dei campi (i sei campi vettoriali/scalari piu' il buffer
di pressione, tutti allocati in `solver_init`/`solver_solve`, dominano il
picco in ogni configurazione testata). Punto chiuso: l'effetto del batch e'
reale, monotono e ora quantificato con margine, anche se resta piccolo in
termini assoluti a queste taglie di griglia.

## 5. gprofng: campionamento nativo (job 32593 + 32603 fix)

Caso piu' grande (128³, 100 passi), nessuna serializzazione dei thread.

**OMP-only (1 rank, 4 thread, 111.85s CPU campionati)**: `libm` (trig)
20.2%, `thomas_solve` 18.5%, **libgomp (runtime OpenMP) 16.4%**,
`gamma_from_k` 8.7%, `momentum_assemble_line` 8.0% (40.1% inclusivo),
kernel SIMD (`update_u_simd`/`update_zeta_simd`) ~7% ciascuno.
`momentum_direction` pesa il 50.5% inclusivo del tempo totale.

**MPI+OMP (2 rank sullo stesso nodo, 2 thread ciascuno) — risultato piu'
rilevante di questo job**: **73.99% (rank 0) e 72.26% (rank 1) del tempo CPU
e' speso dentro il runtime OpenMP (libgomp)**, non nel calcolo. La calltree
lo localizza con precisione: dentro `momentum_direction`, 60.45% del tempo
di quella chiamata e' in libgomp (attesa/sincronizzazione dei thread),
contro 12.79% in `momentum_assemble_line` (il lavoro vero); dentro
`pressure_direction`, 96.96% del tempo e' in libgomp.

**Interpretazione**: a 2 thread per rank su questa taglia di blocco locale,
i thread OpenMP passano la maggior parte del tempo ad aspettarsi a vicenda
(barriere di fine regione parallela), non a calcolare. Questo e' coerente
con — ma piu' esteso di — il 24.85%/26.6% di tempo che mpiP attribuisce
alle sole chiamate MPI (§6): mpiP misura solo il tempo dentro le funzioni
MPI, gprofng cattura anche l'attesa dei thread che non fanno la chiamata
MPI ma restano bloccati alla barriera OpenMP nel frattempo.

**Aggiornamento (job 32608, stesso caso, 1 e 4 thread per rank invece di
2)**: la quota di tempo in libgomp **cresce con il numero di thread**, non
resta costante:

| Thread/rank | CPU campionata | Quota in libgomp |
|---|---|---|
| 1 | 0.70 s (rank 1), 0.72 s (rank 2) | non attendibile — vedi sotto |
| 2 (job 32603) | 82.14 s / 83.54 s | 73.99% / 72.26% |
| 4 | 404.90 s | **92.69%** |

Il punto a 1 thread **non e' una misura attendibile**: gprofng ha campionato
solo ~0.7 secondi di CPU per rank contro 88.9 secondi di tempo di parete
reale (dai timer nativi dello stesso run, "wall per step" 889.2 ms × 100
passi) — meno del 2% dell'esecuzione, sistematico su entrambi i rank. La
causa non e' stata diagnosticata in questa sessione (verosimilmente
un'interazione fra il campionamento a clock di gprofng e l'assenza di un
team di thread OpenMP quando `OMP_NUM_THREADS=1`); il dato "0% di libgomp a
1 thread" **non va usato**, e' un artefatto di sotto-campionamento, non una
misura.

Il confronto **2 → 4 thread e' invece ben campionato** (82-84s e 405s di CPU
sono coerenti con lavoro reale) ed e' la risposta solida alla domanda
aperta: la quota di tempo in runtime OpenMP **cresce** con il numero di
thread (74% → 93%), cioe' aggiungere thread per rank in questa
configurazione ibrida (2 rank sullo stesso nodo, 128³) peggiora la frazione
di tempo passata in sincronizzazione invece di migliorarla. Non e' quindi
un effetto isolato del caso a 2 thread: e' un problema di scalabilita' che
si aggrava aumentando i thread, degno di essere approfondito prima di
aumentare ulteriormente il parallelismo OpenMP per rank su questo tipo di
nodo/taglia.

## 6. mpiP: chiamate, byte, simbolizzazione offline (job 32594)

**Misura**: MPI% aggregato 24.85% del tempo applicativo (128³, 100 passi,
2 rank). Dominano `MPI_Sendrecv` (due punti di chiamata, ~8.4%+8.1%
dell'AppTime, 38.400 chiamate ciascuno — lo scambio halo) e `MPI_Allgather`
(due punti, il collettivo Schur per gruppo di linee), poi `MPI_Waitall`.
Per byte, `Allgather` domina il volume (157 MB aggregati a un punto di
chiamata) davanti a `Sendrecv` (118 MB).

**Simbolizzazione offline (il limite segnalato in precedenza, ora
migliorato)**: compilando con `-no-pie` gli indirizzi dei nostri stessi
punti di chiamata sono risolvibili con `addr2line` senza bisogno di BFD:
`par_exchange_halo` risolto a `src/parallel.c:350,352,354,356,359`,
`par_topology_init`, `par_neighbor`, `par_line_allgather` a
`parallel.c:248`. Gli indirizzi piu' profondi nello stack (dentro
`libmpi.so`/`libopen-pal.so`, con ASLR) restano "not in executable": limite
accettato, non un errore dello script (vedi [pipeline/README.md](pipeline/README.md)).

**Limite gia' noto, ora diagnosticato fino in fondo (job 32609, 32613,
32614, 32615)**: PBS assegna correttamente due nodi distinti (verificato:
`cpu01`, `cpu02` nel `PBS_NODEFILE`), ma `mpirun -np 2 --map-by node`
fallisce con `Permission denied (publickey)` quando `cpu01` tenta di
collegarsi via SSH a `cpu02`. Tre passi per arrivare alla causa:

1. Generata una coppia di chiavi SSH dedicata, senza passphrase, nella
   propria area (`~/.ssh/id_ed25519` sul cluster, non quella del Mac),
   aggiunta ai propri `authorized_keys` senza toccare la voce gia'
   presente. Non ha risolto: stesso errore identico.
2. Un `ssh -vvv` diretto (non mediato da `mpirun`) mostra che il client
   offre correttamente la nuova chiave (`Offering public key: ... ED25519
   SHA256:5LV3...`), ma il server la **rifiuta prima ancora di chiedere la
   firma** (`Authentications that can continue: publickey` senza mai un
   `PK_OK`): non e' un problema della chiave in se'.
3. `pbsdsh` — il meccanismo nativo di PBS per eseguire su un altro nodo
   della propria allocazione, che **non passa per SSH** — fallisce in modo
   identico (`error 15010 on spawn`) provando a raggiungere `cpu02` da
   `cpu01`. E l'Open MPI installato qui (`ompi_info`) ha solo i componenti
   di lancio `rsh`, `slurm` e `isolated`: nessun `tm` (il lancio nativo per
   PBS), quindi `mpirun` non ha comunque un'alternativa a `rsh`/SSH su
   questa installazione.

**Conclusione**: non e' un prerequisito mancante nella propria area (la
chiave era corretta e offerta correttamente), ne' qualcosa risolvibile
installando altro nel proprio spazio personale — anche il canale nativo di
PBS fra questi due nodi non funziona. E' un limite di infrastruttura del
cluster fra `cpu01` e `cpu02`, coerente con (e ora piu' preciso di)
"i tentativi sugli altri nodi non hanno prodotto verifiche conclusive" gia'
annotato in PROFILING_SENZA_ROOT.md. Non ulteriormente perseguito in questa
sessione: risolverlo uscirebbe dall'area personale dell'utente e
dipenderebbe dagli amministratori del cluster, che questo lavoro non doveva
coinvolgere.

## 7. Convergenza fra metodi indipendenti

Un punto di robustezza del lavoro: tre metodi indipendenti (timer nativi,
Callgrind, gprofng) concordano sugli stessi hotspot pur misurando cose
diverse (tempo di parete, istruzioni simulate, campioni di CPU nativi):
`thomas_solve` e la valutazione trigonometrica del termine forzante (`g
term`/`libm`) emergono in tutti e tre. Questo non e' ridondanza: e' la
verifica incrociata che l'assenza di contatori hardware verificati (§ vedi
PROFILING_SENZA_ROOT.md) non ha impedito di localizzare i costi reali.

## 8. Cosa resta da verificare

Punti della prima stesura, ora chiusi con dati (job 32606-32609):

- ~~Overhead delle flag di debug~~ **chiuso**: 5 repliche per build,
  nessuna differenza sistematica misurabile, il rumore di nodo condiviso
  arriva a 12x (§1).
- ~~BENCH_NORMS sul caso native~~ **chiuso**: norme finite e convergenti a
  128³ (§1).
- ~~Effetto batch su Massif solo a 32³~~ **chiuso**: confermato a 64³ senza
  `BENCH_NORMS`, crescita monotona di 133 KB da batch=1 a 128, 8 volte piu'
  netta che a 32³ (§4).
- ~~Dominio di libgomp solo a 2 thread~~ **chiuso, con una riserva**: cresce
  con il numero di thread (74%→93% da 2 a 4), non e' un effetto isolato del
  caso a 2 thread (§5). Il punto a 1 thread e' invece un artefatto di
  sotto-campionamento di gprofng, non una misura — non riprovato in questa
  sessione perche' non centrale rispetto alla domanda originale (che
  riguardava se l'effetto *cresce* con i thread, e la risposta a quello e'
  gia' solida dal confronto 2→4).
- ~~Multi-nodo~~ **diagnosticato fino in fondo, non risolvibile dall'area
  utente**: ne' la chiave SSH ne' il lancio nativo PBS (`pbsdsh`)
  funzionano fra `cpu01` e `cpu02` (§6). Non e' un prerequisito mancante
  nella propria area: e' un limite di infrastruttura fra questi nodi.
  Nessuna ulteriore azione presa, per non far dipendere il lavoro dagli
  amministratori.

Punti ancora aperti, non affrontati in questa sessione:

- **La scomposizione per asse (eta/zeta/u) resta solo nei timer nativi**:
  Callgrind isola `momentum_step`/`pressure_step` nel loro insieme (i tre
  assi condividono la stessa funzione `momentum_direction`), non i singoli
  assi al suo interno.
- **perf/LIKWID restano bloccati** su `cpu01` esattamente come descritto in
  PROFILING_SENZA_ROOT.md: questa sessione non ha tentato ne' modificato
  nulla su quel fronte, per costruzione (nessun accesso a `perf_event` o
  MSR e' stato usato).
- **"unaccounted" al 23.1% a 4 rank (§2)**: la causa e' ora identificata per
  lettura del sorgente, non solo per differenza residua — `solver.c:109-136`
  esegue 13 scambi di halo per passo (`refresh_vector_halo` su eta/zeta/u
  piu' `par_exchange_halo` su `pressure_star`, e ancora `refresh_vector_halo`
  su `u` prima di `pressure_step`) fuori da qualunque timer di fase. Coerente
  con l'osservazione che l'unaccounted e' ~0% a 1 rank (dove
  `par_exchange_halo` e' un no-op locale) e cresce con il numero di rank.
  Non e' stato aggiunto un timer dedicato nel codice: sarebbe una modifica
  al solver vero e proprio, non solo al workflow di profiling, e non e'
  stata richiesta esplicitamente.
- **Il perche' preciso del dominio di libgomp** (load imbalance fra thread
  dentro `momentum_direction`/`pressure_direction`? costo fisso della
  barriera OpenMP indipendente dal lavoro? contesa fra i due rank sulle
  stesse risorse del nodo?) non e' stato isolato: sappiamo che c'e', che e'
  ingente e che cresce con i thread, non ancora esattamente perche'.

## Riproducibilita'

Workflow, script PBS e istruzioni di esecuzione in
[pipeline/](pipeline/) ([README.md](pipeline/README.md)), inclusi gli
script `solver_followup_*.pbs` usati per i punti della sez. 8. Log completi
di ogni job in [pipeline/results/](pipeline/results/). Le directory grezze
`.er` di gprofng (dati di campionamento binari, alcune centinaia di MB)
restano sul cluster in
`~/likwid-profiling/pipeline-solver/runs/gprofng-*/`, non copiate qui.
