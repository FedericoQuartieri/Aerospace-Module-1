# Design e implementazione dei kernel numerici

Questo documento descrive il comportamento effettivo dei kernel del solver
Navier–Stokes–Brinkman. È il riferimento per comprendere:

- la fattorizzazione direzionale del momentum e della pressione;
- i sistemi tridiagonali e le loro righe di bordo;
- il layout dei campi e l'ordine degli accessi in memoria;
- l'uso del workspace preallocato;
- le differenze fra backend `STANDARD` e `OPTIMIZED`;
- le trasformazioni matematiche usate dal percorso ottimizzato;
- i vincoli necessari per mantenere equivalenza numerica.

Il documento descrive il codice corrente in `src/kernels_standard.c`,
`src/kernels_optimized.c`, `src/physics.c`, `src/momentum.c` e
`src/pressure.c`. Le decisioni prestazionali sperimentali sono riportate anche
in `docs/x-kernel-performance.md`.

## 1. Contratto generale dei backend

I due backend esportano le stesse sei operazioni numeriche:

| Sottosistema | X | Y | Z |
|---|---|---|---|
| Momentum | `momentum_solve_x` | `momentum_solve_y` | `momentum_solve_z` |
| Pressione | `pressure_solve_x` | `pressure_solve_y` | `pressure_solve_z` |

La selezione avviene a compile-time in `src/kernels.h`. Gli alias
`backend_*` diventano direttamente i simboli `standard_*` oppure
`optimized_*`; nel timestep non esistono function pointer, branch sul backend
o dispatch runtime.

Ogni kernel riceve:

- una `Grid` già costruita;
- campi distinti per input e output;
- un `RealBuffer` allocato una volta in `solver_init()`;
- eventuali callback fisiche tramite `ProblemDefinition`.

I kernel non allocano e non liberano memoria. La capacità dello scratch è
calcolata dal backend attivo prima del primo timestep.

## 2. Layout dei dati e coordinate staggered

### 2.1 Indicizzazione

Tutti i campi scalari usano layout row-major con X contigua:

```text
q(i,j,k) = i + j*nx + k*nx*ny

stride_x = 1
stride_y = nx
stride_z = nx*ny
```

Incrementare `i` visita quindi elementi consecutivi. Incrementare `j` o `k`
lungo una singola linea produce invece accessi separati rispettivamente da
`nx` e `nx*ny` valori.

Un `VectorField` è una struttura SoA:

```text
component[X] -> allocazione indipendente di N valori
component[Y] -> allocazione indipendente di N valori
component[Z] -> allocazione indipendente di N valori
```

Le tre componenti non sono interlacciate. Ogni allocazione è allineata a 64
byte. Questo permette di leggere una componente come stream contiguo e rende
semplici i load SIMD lungo X.

### 2.2 Griglia staggered

Per ogni direzione `d` la spaziatura è:

```text
h_d = 2*L_d / (2*N_d - 1)
```

Le coordinate della pressione sono `index*h_d`. La componente di velocità `c`
è spostata di `h_c/2` soltanto lungo il proprio asse:

```text
x_d(i,c) = i*h_d + (d == c ? h_d/2 : 0)
```

I kernel usano i valori precomputati `1/h_d` e `1/h_d²` presenti in `Grid`.

## 3. Sequenza matematica di un timestep

### 3.1 Momentum

Per ogni componente della velocità vengono aggiornati tre campi persistenti:

```text
Eta  --solve X--> Eta nuova
Zeta --solve Y--> Zeta nuova, usando Eta nuova
U    --solve Z--> U nuova, usando Zeta nuova
```

La forma incrementale è:

```text
Xi - U = dt/beta * g

(I - gamma Dxx)(Eta_new  - Eta_old)  = Xi       - Eta_old
(I - gamma Dyy)(Zeta_new - Zeta_old) = Eta_new  - Zeta_old
(I - gamma Dzz)(U_new    - U_old)    = Zeta_new - U_old
```

Il loop sulle componenti è esterno a ogni kernel direzionale. Il workspace
full-grid può quindi essere riutilizzato dalla componente successiva.

Nel solve X:

```text
g = forcing
    - grad(pressure_star)
    - nu/K * U
    + nu * (Dxx(Eta) + Dyy(Zeta) + Dzz(U))

rhs_x = U + dt/beta*g - Eta
```

La forzante è valutata a `(timestep - 1/2)*dt`. I valori assoluti di velocità
usati nei ghost point sono valutati a `(timestep - 1)*dt`. La boundary del
sistema incrementale è invece:

```text
delta_bc = u_bc(timestep*dt) - u_bc((timestep - 1)*dt)
```

All'inizio del primo timestep `Eta`, `Zeta` e `U` contengono la velocità a
`t=0`; `pressure_star` è una copia della pressione iniziale, secondo lo startup
della baseline.

### 3.2 Pressione

Dopo il momentum si applica la fattorizzazione:

```text
(I - Dxx) Psi        = -div(U_new)/dt
(I - Dyy) Phi_lower  = Psi
(I - Dzz) correction = Phi_lower
```

La pipeline usa due soli campi full-grid:

```text
workspace.field  <- RHS della divergenza
pressure_star    <- Psi
workspace.field  <- Phi_lower
pressure_star    <- correction
```

Il vecchio `pressure_star` è morto dopo il momentum e può quindi contenere gli
intermedi della pressione. Alla fine, un traversal unit-stride esegue:

```text
pressure_next = pressure + correction
pressure      = pressure_next
pressure_star = pressure_next + correction
```

Il secondo assegnamento costruisce il predittore per il timestep successivo.

## 4. Algoritmo di Thomas usato dai kernel

Per una matrice tridiagonale con riga generica:

```text
a_l*x_(l-1) + b_l*x_l + c_l*x_(l+1) = d_l
```

la forward elimination memorizza:

```text
den_l = b_l - a_l*c'_(l-1)
c'_l  = c_l / den_l
d'_l  = (d_l - a_l*d'_(l-1)) / den_l
```

La back substitution è:

```text
x_last = d'_last
x_l    = d'_l - c'_l*x_(l+1)
```

Nei kernel:

- `tmp` o `modified_superdiagonal` contiene `c'`;
- il RHS, l'output o un buffer dedicato contiene `d'`;
- `increment` o l'output finale contiene la soluzione.

La ricorrenza lungo una singola linea è necessariamente sequenziale. Il
parallelismo disponibile è fra sistemi appartenenti a linee diverse.

## 5. Righe del sistema momentum

Per la direzione `d`:

```text
w_q = -gamma_q / h_d²
```

La riga interna è:

```text
w*x_(l-1) + (1 - 2w)*x_l + w*x_(l+1) = rhs_l
```

Poiché `gamma > 0`, `w` è negativo. Il codice conserva questa convenzione per
evitare cambi di segno fra formule e implementazione.

### 5.1 Boundary inferiore

L'incremento è imposto direttamente:

```text
x_0 = delta_bc_left
```

Per Thomas ciò equivale a inizializzare:

```text
c'_0 = 0
d'_0 = delta_bc_left
```

La prima riga interna incorpora quindi il valore sinistro attraverso
`w*d'_0`.

### 5.2 Boundary superiore normale

Se la componente risolta è normale alla direzione del solve:

```text
component == solve_direction
```

l'ultimo incremento viene imposto direttamente:

```text
x_last = delta_bc_right
```

L'ultima riga non viene eliminata con Thomas.

### 5.3 Boundary superiore tangenziale

Per una componente tangenziale la riga finale è:

```text
w*x_(last-1) + (1 - 3w)*x_last
    = rhs_last - 2w*delta_bc_right
```

La diagonale `1-3w` e il contributo `-2w*bc` derivano dal ghost point della
baseline e devono essere preservati esattamente.

## 6. Righe del sistema di pressione

Per il pressure solver:

```text
w = -1/h_d²
```

`w` è costante lungo tutta la griglia e non dipende da `gamma`.

Le condizioni di Neumann omogenee producono le righe:

```text
sinistra:
(1 - 2w)*x_0 + 2w*x_1 = rhs_0

interno:
w*x_(l-1) + (1 - 2w)*x_l + w*x_(l+1) = rhs_l

destra:
w*x_(last-1) + (1 - w)*x_last = rhs_last
```

La prima superdiagonale è quindi `2w`, mentre l'ultima diagonale è `1-w`.
Questa asimmetria è intenzionale e comune ai due backend.

## 7. Fisica e condizioni al bordo condivise

### 7.1 Supporto componente-specifico di `g`

`g` è zero fuori dal supporto valido della componente:

```text
componente X: i=1..nx-2, j=1..ny-1, k=1..nz-1
componente Y: i=1..nx-1, j=1..ny-2, k=1..nz-1
componente Z: i=1..nx-1, j=1..ny-1, k=1..nz-2
```

All'interno del supporto, il gradiente di pressione usa il punto positivo
lungo l'asse della componente:

```text
(pressure_star[q + stride_component] - pressure_star[q]) / h_component
```

Le tre parti del Laplaciano leggono campi differenti:

```text
Dxx(Eta) + Dyy(Zeta) + Dzz(U)
```

Questa separazione è importante per il batching X: `Dxx(Eta)` legge soltanto
la linea X corrente. Le linee con `j` diversi non leggono reciprocamente i loro
valori `Eta`; i contributi trasversali provengono da `Zeta` e `U`, che non sono
modificati durante il solve X. Elaborare insieme più linee X è quindi
numericamente equivalente al completarle una alla volta.

### 7.2 Seconda differenza sul bordo superiore

Quando una derivata raggiunge la faccia superiore si usa il valore assoluto
della boundary velocity:

```text
(field_(last-1) - 3*field_last + 2*u_bc) / h²
```

Il valore `u_bc` è valutato alle coordinate staggered della componente e al
tempo intero della velocità in ingresso.

### 7.3 Priorità di facce, spigoli e vertici

`evaluate_velocity_boundary_increment()` centralizza la semantica dei bordi.
I casi con più facce incidenti vengono risolti in ordine esplicito:

1. vertice inferiore comune;
2. spigoli con facce inferiori;
3. singole facce inferiori;
4. vertici e spigoli superiori;
5. singole facce superiori.

Le facce inferiori hanno quindi priorità. Sulle facce inferiori, la componente
normale include la correzione discreta costruita dalle derivate trasversali
degli incrementi, per mantenere il vincolo di divergenza della baseline.

## 8. Backend STANDARD

Il backend standard non usa intrinsics. I loop sono scalari e mantengono X come
inner loop ogni volta che più sistemi Y/Z possono essere attraversati insieme.

### 8.1 Capacità e partizione dello scratch

La capacità è:

```text
max(3*nx, 2*nx*max(ny,nz))
```

Le partizioni sono:

| Kernel | Segmenti nello scratch |
|---|---|
| Momentum X | `tmp[nx]`, `rhs[nx]`, `increment[nx]` |
| Momentum Y/Z | `tmp[nx*length]`, `increment[nx*length]` |
| Pressione X | `tmp[nx]` |
| Pressione Y/Z | `tmp[nx*length]` |

Il RHS full-grid di momentum Y/Z e pressione X usa `workspace.field`, non lo
scratch.

### 8.2 Momentum X standard

Traversal:

```text
for k
  for j
    forward Thomas lungo i=0..nx-1
    back substitution lungo i=nx-1..0
    aggiornamento Eta lungo i=0..nx-1
```

Per ogni linea `(j,k)`:

1. `tmp[0]=0` e `rhs[0]=delta_bc_left`;
2. per gli interni viene calcolato `w` da `gamma`;
3. `evaluate_momentum_x_rhs()` costruisce `g`, `Xi-Eta` e il RHS senza campi
   globali intermedi;
4. il RHS viene immediatamente ridotto dalla forward elimination;
5. viene applicata la riga superiore normale o tangenziale;
6. la back substitution riempie `increment`;
7. `increment` viene sommato in-place a `Eta`.

Tutti gli array principali avanzano con stride uno lungo X. Anche le letture
dei vicini Y/Z formano stream contigui traslati: per esempio, incrementando
`i`, gli indirizzi `q-nx` e `q+nx` avanzano comunque di uno. Il kernel legge
più stream, ma non esegue gather.

La forward elimination, il RHS e l'accesso a `gamma` sono fusi nello stesso
loop. Restano tuttavia tre passaggi sulla linea scratch: forward, backward e
applicazione dell'incremento.

### 8.3 Momentum Y/Z standard

Prima dei solve viene materializzato in un traversal unit-stride:

```text
rhs_workspace[q] = source[q] - stage[q]
```

Per Y `source=Eta` e `stage=Zeta`; per Z `source=Zeta` e `stage=U`.
Materializzare l'intero RHS prima di aggiornare `stage` impedisce che un valore
aggiornato possa contaminare sistemi non ancora risolti.

Il kernel elabora un piano alla volta:

```text
Y: outer=k, level=j, inner=i
Z: outer=j, level=k, inner=i
```

`level` segue la dipendenza di Thomas, mentre `i` attraversa sistemi
indipendenti e contigui. `tmp[level*nx+i]` e `increment[level*nx+i]` hanno lo
stesso layout del piano del campo. A ogni livello tutti i valori X sono letti e
scritti in ordine contiguo; il salto `stride_y` o `stride_z` avviene soltanto
passando al livello successivo.

Dopo la forward elimination, la riga superiore viene costruita per ogni `i`,
la back substitution procede a livelli decrescenti e un ultimo traversal
aggiunge l'incremento a `stage`.

### 8.4 Pressione X standard

Il kernel esegue due fasi full-grid.

La prima costruisce:

```text
rhs = -div(U)/dt
```

con differenze all'indietro sulle tre componenti. Se `i==0`, `j==0` o `k==0`
il RHS è posto a zero, perché una delle differenze staggered non dispone del
punto precedente.

La seconda fase risolve una linea X contigua per ogni `(j,k)` mediante
`pressure_thomas_line()`. Il RHS full-grid viene modificato in-place dalla
forward elimination e `Psi` viene scritto in `pressure_star`.

Questa implementazione scrive e rilegge il campo RHS. Una variante lineare che
precalcolava i coefficienti costanti ha migliorato il solo kernel, ma non il
timestep abbastanza da superare il gate end-to-end; per questo non è presente
nel percorso corrente.

### 8.5 Pressione Y/Z standard

Per ogni piano, `level` percorre la direzione del solve e `i` è l'inner loop
contiguo. `output` contiene direttamente il RHS ridotto `d'` e poi la soluzione:

1. livello sinistro: applicazione della riga Neumann con superdiagonale `2w`;
2. livelli interni: forward elimination;
3. livello destro: diagonale `1-w`;
4. back substitution in-place su `output`.

Lo scratch contiene soltanto `c'`; non serve un buffer soluzione separato.

## 9. Backend OPTIMIZED

Il backend ottimizzato conserva le stesse formule ma usa due strategie diverse:

- momentum X: batching scalare di più ricorrenze indipendenti e RHS
  strength-reduced;
- momentum e pressione Y/Z: SIMD esplicita fra sistemi adiacenti lungo X.

Pressione X resta uguale al percorso standard.

### 9.1 Astrazione SIMD

Su AArch64 con NEON:

```text
Real=float  -> float32x4_t, SIMD_LENGTH=4
Real=double -> float64x2_t, SIMD_LENGTH=2
```

Load, store, add, sub, mul e div sono mappati alle intrinsics NEON. Su altre
architetture le stesse macro operano su uno scalare con `SIMD_LENGTH=1`: il
backend resta corretto, ma non promette il vantaggio prestazionale della
piattaforma primaria ARM64.

### 9.2 Capacità dello scratch ottimizzato

La capacità è il massimo fra:

```text
momentum X batch:  2 * 4 * nx = 8*nx
percorso planare:  2 * nx * max(ny,nz)
percorso SIMD:     2*longest*(8*SIMD_LENGTH)
                   + 8*SIMD_LENGTH + 3*longest
```

L'ultimo termine copre il blocco maggiore del momentum Z; la pressione usa più
vettori per slice ma un solo piano di coefficienti, quindi resta coperta dallo
stesso massimo.

### 9.3 Momentum X ottimizzato

Il kernel continua a essere invocato una componente alla volta. Per ogni `k`
raggruppa quattro valori consecutivi di `j`:

```text
batch = {(j,k), (j+1,k), (j+2,k), (j+3,k)}
```

Il batch finale può contenere da una a tre linee. Lo scratch è partizionato in:

```text
modified_superdiagonal[4][nx]
rhs[4][nx]
```

Non viene effettuato il packing dei campi fisici: ogni linea continua a essere
letta direttamente dal layout SoA. Il loop principale è:

```text
for i = 1 .. nx-2
  for line = 0 .. line_count-1
    calcola RHS e avanza Thomas della linea
```

La ricorrenza di ogni linea dipende dal proprio valore a `i-1`, ma le quattro
linee non dipendono fra loro. Il core può quindi sovrapporre divisioni e altre
istruzioni delle quattro catene. È instruction-level parallelism, non SIMD e
non multithreading.

Per ciascun punto `gamma` viene letto una volta e usato sia per:

```text
w = -gamma/h_x²
source_scale = 2*gamma/nu
```

RHS e forward elimination restano fusi, evitando un secondo traversal e una
seconda lettura di `gamma`.

La back substitution mantiene in `last_increment[4]` il valore successivo di
ogni linea. Appena viene calcolato un incremento, viene sommato a `Eta`. Non
esiste quindi il buffer `increment[nx]` del backend standard e non serve il
traversal finale in avanti.

### 9.4 Strength reduction del RHS momentum X

In inizializzazione:

```text
beta  = 1 + dt*nu/(2K)
gamma = dt*nu/(2*beta)
```

Il percorso standard ricostruisce `beta` e `K` da `gamma` durante il calcolo di
`g`. Il percorso ottimizzato usa:

```text
source_scale = dt/beta = 2*gamma/nu

source_scale*(nu/K)
    = 2 - 2*source_scale/dt
    = 2 - 4*gamma/(nu*dt)
```

Definendo:

```text
source_without_drag = forcing
                      - grad(pressure_star)
                      + nu*(Dxx(Eta)+Dyy(Zeta)+Dzz(U))

scaled_drag = 2 - 2*source_scale/dt
```

il RHS diventa:

```text
rhs_x = U
        + source_scale*source_without_drag
        - scaled_drag*U
        - Eta
```

La trasformazione elimina le divisioni per punto necessarie a ricostruire
`beta` e `K`. Non cambia forcing, supporto, gradienti, Laplaciani, coordinate o
staggering temporale.

La funzione è `always_inline` con Clang/GCC perché la build ordinaria non usa
LTO. In questo modo non viene introdotta una chiamata per ogni cella. Anche il
calcolo delle coordinate staggered è espanso localmente per evitare tre piccole
chiamate esterne per punto.

### 9.5 Momentum Y/Z ottimizzato

Come nello standard, il RHS `source-stage` viene prima costruito nel campo
full-grid. La SIMD opera poi su più sistemi indipendenti con coordinate X
adiacenti.

Dimensioni dei blocchi:

```text
Y: slice_vectors=4  -> slice=4*SIMD_LENGTH sistemi
Z: slice_vectors=8  -> slice=8*SIMD_LENGTH sistemi
```

Per ogni piano e blocco X:

1. le boundary inferiori vengono valutate scalarmente per ogni lane;
2. per ogni `level`, `gamma`, RHS e coefficienti di `slice` sistemi sono letti
   con load contigui;
3. le forward recurrence indipendenti vengono eseguite con vettori NEON;
4. le boundary superiori sono valutate scalarmente;
5. ultima riga e back substitution sono vettoriali;
6. l'incremento viene sommato a `stage` con load/store vettoriali.

Lo scratch è organizzato come un piccolo piano packed:

```text
tmp[length][slice]
increment[length][slice]
boundary[slice]
tail_tmp[length]
tail_rhs[length]
tail_increment[length]
```

Per un dato `level`, le lane nello scratch e nel campo sono contigue. La SIMD
non segue una singola linea strided: ogni lane rappresenta una linea diversa,
mentre la ricorrenza avanza simultaneamente tutte le linee.

Se `nx` non è multiplo di `slice`, le X residue vengono risolte una alla volta
da `scalar_momentum_tail_line()`. Il fallback usa le stesse righe Thomas e gli
stessi contributi di bordo.

### 9.6 Pressione X ottimizzata

L'implementazione coincide intenzionalmente con quella standard:

1. materializzazione full-grid di `-div(U)/dt`;
2. Thomas scalare sulle linee X contigue.

La ricorrenza è sequenziale nella direzione cache-friendly. Non vengono usati
SIMD, batching o coefficienti persistenti.

### 9.7 Pressione Y/Z ottimizzata

La pressione usa blocchi più larghi:

```text
Y: slice_vectors=16
Z: slice_vectors=16
slice = 16*SIMD_LENGTH sistemi
```

Poiché `w=-1/h²` è costante, il vettore `w`, l'inverso della prima diagonale e
il primo `c'` vengono costruiti una volta fuori dai loop dei piani.

Per ogni blocco:

1. il primo livello legge `input`, applica la riga Neumann sinistra e scrive
   `d'` in `output`;
2. i livelli interni aggiornano `tmp` e `output` con operazioni SIMD;
3. l'ultimo livello usa la diagonale `1-w`;
4. la back substitution modifica `output` in-place.

Lo scratch contiene:

```text
tmp[length][slice]
tail_tmp[length]
tail_rhs[length]
tail_solution[length]
```

Le linee residue non adatte al blocco vengono copiate dal campo strided a
`tail_rhs`, risolte con `pressure_thomas_line()` e ricopiate in `output`. Il
packing riguarda quindi soltanto la coda, non la griglia completa.

## 10. Confronto degli accessi in memoria

| Kernel | STANDARD | OPTIMIZED |
|---|---|---|
| Momentum X | una linea X; forward, backward, update | quattro linee X; forward interlacciato, backward+update fusi |
| Momentum Y | piano `nx*ny`, X inner | blocchi di sistemi X adiacenti, SIMD |
| Momentum Z | piano `nx*nz`, X inner | blocchi più larghi di sistemi X adiacenti, SIMD |
| Pressione X | RHS full-grid + linee X scalari | uguale allo standard |
| Pressione Y/Z | piano con X inner, output in-place | blocchi SIMD, output in-place, tail scalare |

In entrambi i backend:

- i campi fisici restano SoA;
- non esistono trasposizioni full-grid;
- X rimane la dimensione contigua;
- input e output di un kernel non aliasano;
- gli intermedi full-grid vengono riutilizzati secondo la loro lifetime;
- non avvengono allocazioni nel timestep.

## 11. Vincoli di equivalenza numerica

Qualsiasi modifica ai kernel deve preservare:

1. sequenza `Eta -> Zeta -> U -> Psi -> Phi_lower -> correction`;
2. forcing a `(timestep-1/2)*dt`;
3. boundary assolute a `(timestep-1)*dt`;
4. boundary incrementali fra due tempi interi consecutivi;
5. supporto componente-specifico di `g`;
6. gradiente di pressione nella direzione positiva della componente;
7. righe momentum inferiori, normali superiori e tangenziali superiori;
8. righe Neumann della pressione con `2w` a sinistra e `1-w` a destra;
9. indipendenza dei coefficienti pressure da `gamma`;
10. aggiornamento in-place degli stadi soltanto dopo aver costruito i dati da
    cui dipende il sistema;
11. assenza di allocazioni nel call graph di `solver_step()`.

## 12. Verifica dei kernel

`test_kernel_equivalence` invoca direttamente i sei kernel STANDARD e
OPTIMIZED sugli stessi campi e confronta ogni valore. Le estensioni 17 e 31
esercitano batch e tail non multipli delle larghezze SIMD.

Le tolleranze dipendono da `Real`:

```text
double: assoluta e relativa 4e-12
float:  assoluta e relativa 4e-5
```

L'equivalenza puntuale è affiancata da:

- test manufactured di correttezza per entrambi i backend;
- convergenza spaziale e temporale;
- controllo di valori non finiti;
- sanitizer;
- benchmark separati di momentum X, pressione X e timestep in
  `ns/(step cell)`.

Le ottimizzazioni vengono mantenute soltanto se superano prima l'equivalenza
numerica e poi il gate prestazionale end-to-end.
