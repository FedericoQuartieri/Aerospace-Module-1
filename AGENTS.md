# Solver Refactoring Rules

- Leggi integralmente `docs/solver-design.md` prima di modificare il codice.
- L’implementazione identificata dal tag `solver-refactor-baseline` è il riferimento numerico.
  - Preserva formule, staggering temporale, boundary conditions e comportamento
    dei sistemi di Thomas dell’implementazione corrente.
  - Non introdurre correzioni numeriche ricavate da fonti esterne.
  - Implementa la soluzione più semplice che soddisfa il design.
  - Non aggiungere astrazioni speculative, framework generici o dispatch runtime.
  - Mantieni separate le componenti del VectorField in layout SoA.
  - Non eseguire allocazioni durante i timestep.
  - Non allentare tolleranze o rimuovere asserzioni per far passare i test.
  - I test devono avere output numerico disabilitato.
  - Non eseguire test con più di 64 punti per direzione.
  - Non modificare file estranei al refactoring.
  - Non conservare vecchio codice commentato: la versione precedente è disponibile
    nella cronologia Git.
  - Dopo ogni milestone esegui build e test di correttezza.
  - Dopo ogni milestone che modifica dati o percorso numerico esegui anche la
    convergenza sulle griglie 16, 32 e 64.
