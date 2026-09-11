#ifndef PHYSICS_H
#define PHYSICS_H
#include "decomp.h"
#include "types.h"

struct SolverMemState;

Real beta_from_k(Real k);
Real gamma_from_k(Real k);

Real time_physical_coord(Real t_step);
Real centered_physical_coord(int index, int component);
Real staggered_physical_coord(int index, int component);

/*
 * The boundary helpers describe positions in physical space only, so their
 * i, j, k arguments are *global* indices and the caller converts.  That also
 * turns their internal face tests (i == 0, i == WIDTH - 1, ...) into tests on
 * the global boundary, which is what they must be once the grid is split.
 */
Real bc_left(VectorFunction bc_velocity,
             int i, int j, int k, int t_step, int component);
Real bc_right(VectorFunction bc_velocity,
              int i, int j, int k, int t_step, int component);

/*
 * g_value addresses memory as well, so it takes local indices plus the
 * decomposition and derives the global ones itself.
 */
Real g_value(const Decomp *d,
             int i, int j, int k, int t_step, Real k_i,
             const struct SolverMemState *solver_mem_state,
             const Data *data, int component, Real forcing);

/*
 * Le ascisse delle celle di una linea lungo x, sfalsate per la componente
 * richiesta esattamente come le calcola g_value.  Non dipendono ne' da j e k
 * ne' dal passo temporale, quindi un riempimento solo serve tutte le linee
 * del blocco.
 */
void forcing_line_coords(const Decomp *d, int component, Real *restrict xs);

/*
 * Riempie `out` con il termine forzante delle celle della linea (j, k), che
 * poi g_value riceve gia' pronto.
 *
 * Usa forcing_line_fn se lo scenario ce l'ha, altrimenti chiama forcing_fn una
 * cella alla volta: il valore prodotto e' lo stesso, cambia solo quanto costa
 * produrlo.  Cosi' la scelta di ottimizzare o no resta dello scenario, e il
 * core non sa niente della forma della forzante.
 *
 * y, z e il tempo li ricava qui dentro con le stesse operazioni di g_value:
 * le regole sulle mezze celle restano scritte in un posto solo.
 */
void forcing_fill_line(const Decomp *d, const Data *data,
                       int j, int k, int t_step, int component,
                       const Real *restrict xs, Real *restrict out);

/*
 * Il termine forzante di UNA cella, con le stesse mezze celle e lo stesso
 * tempo di forcing_fill_line.
 *
 * E' quello che g_value calcolava al suo interno prima che la forzante gli
 * arrivasse gia' pronta.  Serve a chi non ha una linea lungo x da riempire --
 * il backend pipeline percorre i livelli, non le linee -- e paga come prima
 * una chiamata indiretta per cella.
 */
Real forcing_at_cell(const Decomp *d, const Data *data,
                     int i, int j, int k, int t_step, int component);

/*
 * g di tutta la linea (j, k) lungo x: out[i] e' quello che g_value
 * risponderebbe per la cella (i, j, k).
 *
 * La linea e' l'unita' che rende g economico.  Il test di supporto e la
 * scelta fra derivata seconda interna e nodo fantasma dipendono solo da j e
 * k, che sulla linea non cambiano: si decidono una volta invece che per
 * cella.  Quello che resta e' un conto dritto su celle contigue in memoria --
 * lungo x lo stride e' 1 -- quindi si vettorizza con letture normali, senza
 * gather e senza trasposizioni.
 *
 * Le celle che non hanno quella forma tornano a g_value: sono le due agli
 * estremi della linea, e tutte quelle delle linee che appoggiano su un nodo
 * fantasma, dove il valore al bordo dipende dall'ascissa.  Sono O(1/n) del
 * lavoro, e cosi' quell'algebra resta scritta una volta sola.
 */
void g_line(const Decomp *d, const Data *data,
            const struct SolverMemState *solver_mem_state,
            const Real *restrict k_porosity,
            int j, int k, int t_step, int component,
            const Real *restrict xs, Real *restrict out);

#endif
