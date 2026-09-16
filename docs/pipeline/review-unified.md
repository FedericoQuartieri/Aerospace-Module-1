# Revisione della pipeline di unified

Base: `5c7aa7b4b2db0fe691acda24f96c6c08bdc9a55d` (16 settembre 2026).
Il checkout iniziale era `ae625f8`, indietro di 33 commit. Le modifiche qui
descritte sono integrate sulla nuova base, dopo fetch e fast-forward del branch.

## Cosa era già stato corretto a monte

L'ultimo unified include già SIMD per la quantità di moto della pipeline,
`g_line` e forzante per linea, batch automatico dipendente dai thread, una
prima invalidazione della cache dei binari dello studio e i test fisici
Brinkman. Queste funzionalità sono conservate. La campagna corrente resta
quella delle fasi 10–15; le fasi 00–09 eliminate a monte non sono ripristinate.

## Come si svolge la pipeline

La decomposizione cartesiana MPI stabilisce quali assi sono locali e quali attraversano più processi. All'avvio il backend prepara geometria, scratch, coordinate per g e fattori della pressione. Ogni passo temporale aggiorna gli aloni dei campi e, se necessario, la permeabilità a metà passo; poi esegue:

~~~mermaid
flowchart LR
    U[Velocità precedente] --> X[Quantità di moto X: eta]
    X --> Y[Quantità di moto Y: zeta]
    Y --> Z[Quantità di moto Z: u]
    Z --> H[Scambio aloni di u e divergenza]
    H --> PX[Pressione X: psi]
    PX --> PY[Pressione Y: phi]
    PY --> PZ[Pressione Z: fi]
    PZ --> C[Aggiornamento di pressione e velocità]
~~~

La quantità di moto risolve un incremento da sommare al campo di arrivo. La pressione assegna invece la soluzione. Le tre direzioni sono dipendenti e restano in sequenza; il parallelismo è tra linee indipendenti dentro una direzione. Lungo un asse distribuito, Thomas passa c' e d' al vicino successivo in avanti, quindi la soluzione nel verso opposto. La pressione può omettere c' a runtime perché è già noto dall'inizializzazione.

I due backend condividono la fisica della riga (momentum_row.h) e la matrice della pressione; cambiano algoritmo di soluzione e layout dello scratch. Schur rimane il riferimento indipendente nei confronti numerici.

## Interventi sul solutore

- **OpenMP:** un team persiste per tutta una direzione. Il lavoro assegnato al
  thread comprende l'intera ricorrenza di una linea o di un gruppo SIMD di
  linee. Sono eliminate le barriere tra livelli Thomas. Sulle direzioni MPI
  restano sincronizzazioni ai confini dei batch; tutte le chiamate MPI passano
  dal master, nel rispetto di `MPI_THREAD_FUNNELED`. Si conserva l'ordine
  componenti 0, 1, 2 in avanti e 2, 1, 0 all'indietro.
- **Assi locali:** ogni thread conclude andata e ritorno del proprio batch e
  riutilizza lo scratch. Non serve conservare tre componenti dell'intero
  volume. Il batch locale viene limitato anche per lasciare lavoro a tutti
  i thread; tra vicini MPI il numero di linee rimane coerente.
- **Layout e SIMD:** X usa linee contigue nello scratch; Y/Z mantengono
  adiacenti le linee necessarie ai vettori. Restano gestiti i resti SIMD,
  i batch parziali e le discontinuità tra righe fisiche. La SIMD è applicata
  anche alla pressione. Non cambiano le formule delle condizioni al contorno.
- **Forzante:** `g_line` resta il percorso normale. Ogni worker ha un buffer
  privato lungo una linea; le coordinate delle tre componenti sono preparate
  all'avvio. Il tempo di g è ancora esposto nelle statistiche.
- **Pressione:** i coefficienti, invarianti nel tempo e tra linee dello stesso
  asse, sono fattorizzati una sola volta. All'avvio si propaga anche il
  coefficiente normalizzato all'interfaccia MPI. Durante la soluzione si
  scambia soltanto il termine noto trasformato: payload dell'andata dimezzato,
  payload complessivo andata/ritorno ridotto da tre a due Real per linea.
- **Dimensioni:** controlli di overflow sulle allocazioni, limite al batch
  compatibile con i conteggi MPI, nessuna conversione a int prima di aver
  limitato il numero di linee attive.

L'ottimizzazione locale del passo X si basa su un'invariante verificata nel
codice attuale: g legge eta lungo la stessa linea X e per la stessa componente.
La linea è preparata interamente prima di aggiornarla. Un futuro cambiamento
che introduca dipendenze trasversali in eta dovrà rispettare o rivedere questa
scelta.

Per un blocco MPI locale 128³, c' e d' di tre componenti occupano insieme
96 MiB in double prima del padding dei batch. Il nuovo percorso locale usa
invece due scratch da un batch per worker. La memoria dipende quindi da
geometria, batch e thread: non è una riduzione percentuale costante.

## Build e misure affidabili

Il Makefile identifica una configurazione tramite compilatore, versione e
flag effettivi. Gli oggetti e i binari stanno in `build/variants/<id>/`;
i percorsi tradizionali sono copie aggiornate a ogni richiesta. Cambiare
backend o batch e poi tornare alla prima configurazione non riutilizza più
l'alias dell'ultima build. Gli oggetti comuni vengono compilati una sola volta
per configurazione; i test con default di canale differenti hanno oggetti
separati. Pubblicazione con file temporanei e rename atomico.

Lo studio costruisce direttamente i percorsi di variante, eliminando la
corsa tra job che prima producevano e spostavano lo stesso `build/tests/bench`.
Anche risultati e chiavi di ripresa sono separati per contenuto delle sorgenti
e toolchain, in `build/study/<id>/`. Una nuova build non eredita automaticamente
il successo di casi misurati su codice precedente. Le vecchie campagne restano
conservate. Un `STUDY_BASE` esplicito con identità incompatibile viene rifiutato
quando si cerca di avviare una fase.

La fase 15 genera l'elenco dei casi attesi prima degli eventuali skip; verifica
presenza, esito, forma MPI richiesta, finitezza e accordo delle norme con un
riferimento Schur scalare seriale sempre presente. Fallisce con exit code non
zero quando il controllo fallisce. Se il budget lascia casi da completare,
rinvia il verdetto alla continuazione. Anche `run_equivalence.sh` restituisce
un errore per dati incompleti, non numerici o diversi oltre soglia.

Il tempo MPI mostrato viene limitato allo stesso intervallo del solve,
escludendo inizializzazione e scambi per l'output. La RSS su macOS viene
convertita da byte a KiB prima del confronto con Linux. Le norme sono stampate
con 17 cifre dopo la virgola in formato scientifico e bench fallisce su norme
non finite.

## Verifiche

`check_pipeline.sh` usa ora un confronto punto per punto di tutti gli undici
campi di stato con Schur seriale scalare. Le norme aggregate, da sole, possono
nascondere errori localizzati o permutazioni. Si ricostruiscono i blocchi MPI
controllando copertura, sovrapposizioni e valori non finiti.

I test coprono scenari con permeabilità costante e variabile nello spazio e
nel tempo, decomposizioni pure e miste, griglie rettangolari, batch parziali,
resti SIMD e blocchi sottili. Il test dei residui controlla direttamente le
equazioni di quantità di moto e pressione, incluse le giunzioni MPI, e ripete
la pressione con termini noti differenti per verificare il riuso dei fattori.

Sono inoltre presenti test automatici per il validatore dello studio, il
confronto dei campi, l'identità delle sorgenti e il cambio di configurazione
nel Makefile. Il workflow GitHub Actions aggiunto esegue matrici double/float,
SIMD attiva/disattiva, MPI/OpenMP e una verifica con AddressSanitizer/UBSan.
La sua esecuzione remota avverrà solo dopo il push dell'utente.

I risultati delle esecuzioni di questa revisione sono riportati in
`validation.md`, accanto a questo documento.

## Passo successivo sul cluster

Conservata l'euristica del batch introdotta da unified (potenza di due più
vicina a 256 × thread, limite 4096), insieme agli override runtime e compile
time. Era stata calibrata quando esistevano barriere per livello: il suo
ottimo va rimisurato. Dopo la fase 15, ripetere le fasi 10–14, soprattutto
13 per il batch e 14 per memoria e scaling, nella nuova cartella di campagna.
I risultati storici non sono misure della nuova implementazione.
