# Esperimenti prestazionali sui kernel X

Questo documento registra le decisioni prestazionali per i kernel X. Le
formule numeriche, lo staggering e le righe di bordo dei sistemi di Thomas non
sono criteri negoziabili: una candidata viene misurata solo dopo aver superato
correttezza ed equivalenza kernel.

## Metodo di misura

La piattaforma primaria è Apple/ARM64 NEON in precisione `double`, build
`Release`, senza multithreading e con output disabilitato. Il probe
`test/benchmark_x.c` misura separatamente:

- momentum X in `ns/(step cell)`;
- pressione X in `ns/(step cell)`;
- timestep completo in `ns/(step cell)`.

Si usano griglie 64³ e 128³ e due workload: il caso manufactured `paper` e un
caso sintetico con callback algebriche leggere. `benchmark_x.py` alterna
baseline e candidata, usa la mediana di cinque ripetizioni e controlla anche la
MAD relativa. Il gate richiede almeno 15% sul kernel obiettivo e 5% sul
timestep paper a 128³, senza regressioni oltre 2% a 64³. Non vengono eseguiti
benchmark a 256³.

## Ottimizzazione mantenuta: momentum X

Il kernel ottimizzato elabora quattro linee X indipendenti per batch. A ogni
indice X avanza le quattro ricorrenze scalari di Thomas: ogni singolo sistema
resta sequenziale, ma le divisioni di sistemi diversi non dipendono tra loro e
il core può sovrapporne la latenza. Le linee rimangono contigue in X; non sono
necessari gather/scatter né un buffer di packing full-grid.

Il RHS e la forward elimination restano fusi. La back substitution aggiorna
direttamente `eta`, evitando il buffer `increment` e un ulteriore traversal in
avanti. Lo scratch preallocato contiene coefficienti e RHS per quattro linee,
per un fabbisogno X di `8*nx` valori e nessuna allocazione nel timestep.

Il percorso caldo usa inoltre le identità già implicite nei coefficienti del
solver:

```text
gamma = dt*nu/(2*beta)
dt/beta = 2*gamma/nu
K = dt*nu/(2*(beta - 1))
(2*gamma/nu)*(nu/K) = 2 - 4*gamma/(nu*dt)
```

Questo evita di ricostruire `beta` e `K` con divisioni per ogni cella. Il
calcolo conserva forcing, differenze finite, supporto componente-specifico e
livelli temporali della baseline. Il codice è mantenuto inline nel backend
ottimizzato perché le build ordinarie non abilitano LTO; in questo modo non si
introduce una chiamata di funzione in ogni punto del RHS.

Mediane misurate su cinque ripetizioni:

| workload | griglia | momentum X baseline | momentum X candidato | guadagno X | timestep baseline | timestep candidato | guadagno timestep |
|---|---:|---:|---:|---:|---:|---:|---:|
| paper | 64³ | 98.068 | 72.683 | 25.9% | 129.438 | 104.016 | 19.6% |
| paper | 128³ | 102.468 | 72.415 | 29.3% | 139.941 | 110.247 | 21.2% |
| synthetic | 64³ | 57.607 | 30.134 | 47.7% | 84.442 | 56.806 | 32.7% |
| synthetic | 128³ | 50.480 | 30.973 | 38.6% | 86.787 | 67.439 | 22.3% |

I tempi sono in `ns/(step cell)`; la MAD relativa massima osservata è stata
1,87%. Il gate momentum è passato.

## Esperimenti scartati

### Pressione X: coefficienti invarianti e RHS per linea

La matrice della pressione X è a coefficienti costanti. Precalcolare una volta
le diagonali modificate di Thomas e costruire direttamente un RHS di linea ha
ridotto preliminarmente il solo kernel pressione X di circa 43% a 128³.
Tuttavia la pressione X rappresentava circa il 5–6% del timestep: il guadagno
globale osservabile restava intorno al 2–3%, sotto il gate del 5%. La modifica è
stata rimossa perché il beneficio complessivo non giustificava un secondo
percorso Thomas specializzato.

### Momentum X: fusione delle tre componenti

Elaborare X/Y/Z insieme riusava `gamma` e i coefficienti tridiagonali. Sul caso
paper il miglioramento preliminare era modesto, mentre il workload sintetico
regrediva sensibilmente a causa dei maggiori stream simultanei e della peggiore
ottimizzazione del loop interno sulle componenti. La candidata è stata rimossa.

### Momentum X: RHS separato dalla ricorrenza

Costruire quattro RHS completi e risolverli in una seconda fase esponeva le
ricorrenze indipendenti, ma introduceva un passaggio aggiuntivo sui buffer di
linea e perdeva la fusione cache-friendly del percorso originale. La variante
interlacciata e fusa mantenuta sopra è risultata nettamente migliore.
