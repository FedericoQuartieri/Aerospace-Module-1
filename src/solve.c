#include "solve.h"

void *io_thread_func(void *arg)
{
    while (1) {
        pthread_mutex_lock(&io_mutex);

        // Wait until solver thread signals new data
        while (io_pending == 0)
            pthread_cond_wait(&io_cond, &io_mutex);

        int buf_to_write = io_pending - 1; // convert 1/2 into 0/1
        io_pending = 0;

        pthread_mutex_unlock(&io_mutex);

        // ---- Write buffer buf_to_write to disk ----
        char filename[256];
        sprintf(filename, "solution_%06d.vti", current_time_step());

        write_vti_file(filename,
                       &U_buf[buf_to_write],
                       &P_buf[buf_to_write],
                       Nx, Ny, Nz, dx, dy, dz);

        // After writing, simply continue and wait for next job
    }
}

/* 
    Get the pointers to the Velocity struct and swap each pointers inside them
*/
static void swap_velocity(VelocityField *U, VelocityField *U_next) {
    
    DTYPE *tmp;
    /* v_x */
    tmp = U->v_x;
    U->v_x = U_next->v_x;
    U_next->v_x = tmp;
    /* v_y */
    tmp = U->v_y;
    U->v_y = U_next->v_y;
    U_next->v_y = tmp;
    /* v_z */
    tmp = U->v_z;
    U->v_z = U_next->v_z;
    U_next->v_z = tmp;
} 

/* 
    Get the pointers to the Pressure struct and swap the pointers inside
*/
static void swap_pressure(Pressure *pressure, Pressure *pressure_next) {
    
    DTYPE *tmp;
    tmp = pressure->p;
    pressure->p = pressure_next->p;
    pressure_next->p = tmp;
} 


void solve (GField g_field, ForceField f_field, Pressure pressure, Pressure pressure_next, DTYPE* K, VelocityField Eta, VelocityField Zeta, VelocityField U, DTYPE* Beta, DTYPE* Gamma, int write_frequency) {
    
    /* 
        For the seriel implementation, we will use a separate thread to write the .vtk, 
        using two buffer such that the solver_thread willd copy U and p to that buffer and the Output_thread
        will write back to disk using that thread.
    */
    /* Write buffers */
    VelocityField U_buf[2];
    Pressure P_buf[2];
    initialize_velocity_field(&U_buf[0]);
    initialize_velocity_field(&U_buf[1]);
    initialize_pressure(&P_buf[0]);
    initialize_pressure(&P_buf[1]);
    
    int current_buf = 0;
    pthread_t io_thread;
    pthread_mutex_t io_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t  io_cond  = PTHREAD_COND_INITIALIZER;

    int io_pending = 0; 
    /* Start the thread */
    pthread_create(&io_thread, NULL, io_thread_func, NULL);
    

    // For each time step:
    // declare U_next, Eta_next, Zeta_next, Xi
    VelocityField Eta_next;
    VelocityField Zeta_next;
    VelocityField U_next;
    VelocityField Xi;
    initialize_velocity_field(&Xi);
    initialize_velocity_field(&Eta_next);
    initialize_velocity_field(&Zeta_next);
    initialize_velocity_field(&U_next);

    for (int t = 0; t < STEPS; t++) {
        // Compute G at time step n
        solve_momentum_system(U, Eta, Zeta, Xi, g_field, U_next, Eta_next, Zeta_next, Beta, Gamma);

        solve_pressure_system(pressure, pressure_next, U_next);


        if (t % write_frequency == 0) {

            int buf = current_buf;

            // 1. Copy current solver fields into the write buffer
            memcpy(U_buf[buf].v_x, U_next.v_x, GRID_SIZE);
            memcpy(U_buf[buf].v_y, U_next.v_y, GRID_SIZE);
            memcpy(U_buf[buf].v_z, U_next.v_z, GRID_SIZE);
            memcpy(P_buf[buf].p, pressure_next.p, GRID_SIZE);

            // 2. Signal the IO thread that a write is pending
            pthread_mutex_lock(&io_mutex);
            io_pending = buf+1;  // e.g. 1 means buf 0, 2 means buf 1
            pthread_cond_signal(&io_cond);
            pthread_mutex_unlock(&io_mutex);

            // 3. Switch buffer (ping-pong)
            current_buf = 1 - current_buf;
        }

        swap_velocity(&U, &U_next);
        swap_velocity(&Eta, &Eta_next);
        swap_velocity(&Zeta, &Zeta_next);
        swap_pressure(&pressure, &pressure_next);
    }

    free_velocity_field(&Xi);
    free_velocity_field(&Eta_next);
    free_velocity_field(&Zeta_next);
    free_velocity_field(&U_next);

    pthread_cancel(io_thread);
    pthread_join(io_thread, NULL);
}