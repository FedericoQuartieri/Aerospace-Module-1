#include "solve.h"
#include "io_thread.h"


/* 
    Get the pointers to the Velocity struct and swap each pointers inside them
*/
static void swap_velocity(VelocityField *U, VelocityField *U_next) {
    
    DTYPE *tmp;
    /* v_x */
    tmp = U->v_x; U->v_x = U_next->v_x; U_next->v_x = tmp;
    /* v_y */
    tmp = U->v_y; U->v_y = U_next->v_y; U_next->v_y = tmp;
    /* v_z */
    tmp = U->v_z; U->v_z = U_next->v_z; U_next->v_z = tmp;
} 

/* 
    Get the pointers to the Pressure struct and swap the pointers inside
*/
static void swap_pressure(Pressure *pressure, Pressure *pressure_next) {
    
    DTYPE *tmp;
    tmp = pressure->p; pressure->p = pressure_next->p; pressure_next->p = tmp;
} 

void solve (GField g_field, forcing_function_t forcing, Pressure pressure, DTYPE* K, 
            VelocityField Eta, VelocityField Zeta, VelocityField U, 
            DTYPE* Beta, DTYPE* Gamma, 
            DTYPE *u_BC_current_direction, DTYPE *u_BC_derivative_second_direction, DTYPE *u_BC_derivative_third_direction, 
            int write_frequency, bool full_output, VelocityField** U_record, Pressure** P_record) {
    
    /* 
        For the seriel implementation, we will use a separate thread to write the .vtk, 
        The IO thread implements a ring queue wich (velocity_buffer, pressure_buffer) for each entries,
        this allows the solver to continue his execution without wasting time in writing into file,
        the memcpy is infact much faster than writing into a file.
        However, when the queue is full, the solver wait for the IO thread signal .not_full, slowing down performance
        The way to avoid that is to drop some frame and continue execution from the solver.
    */

    IOQueue io_queue;
    io_queue_init(&io_queue);
    pthread_t io_thread;

    /* Create the IO thread */
    if(pthread_create(&io_thread, NULL, io_thread_func, &io_queue) != 0) {
        printf("\nIO thread was not created correctly\n");
        io_queue_destroy(&io_queue);
        return;
    }
    mkdir("output", 0755);   /* create output/ directory if doesn't exists */

    VelocityField Eta_next;
    VelocityField Zeta_next;
    VelocityField U_next;
    VelocityField Xi;
    initialize_velocity_field(&Xi);
    initialize_velocity_field(&Eta_next);
    initialize_velocity_field(&Zeta_next);
    initialize_velocity_field(&U_next);

    Pressure pressure_next;
    initialize_pressure(&pressure_next);

    for (int t = 0; t < STEPS; t++) {

        // Update G field
        compute_g(&g_field, forcing, &pressure, K, &Eta, &Zeta, &U, t);        

        solve_momentum_system(U, Eta, Zeta, Xi, g_field, U_next, Eta_next, Zeta_next, Beta, Gamma, u_BC_current_direction, u_BC_derivative_second_direction, u_BC_derivative_third_direction);

        // not implemented yet
        // solve_pressure_system(pressure, pressure_next, U_next);

        if (t % write_frequency == 0) {

            pthread_mutex_lock(&io_queue.mutex);

            /* If the queue is full, wait... */
            while (io_queue.count == IO_QUEUE_SIZE) {
                pthread_cond_wait(&io_queue.not_full, &io_queue.mutex);
            }

            int idx = io_queue.head;
            /* Update the head of the queue, but wait to increment .count */
            io_queue.head = (io_queue.head + 1) % IO_QUEUE_SIZE;

            pthread_mutex_unlock(&io_queue.mutex);
            
            io_queue.timestep[idx] = t;
            /* Here IO thread can continue to work, doesn't wait for memcpy */
            /* Copy data into buffers at position [head] */
            memcpy(io_queue.U_buf[idx].v_x, U_next.v_x, GRID_SIZE);
            memcpy(io_queue.U_buf[idx].v_y, U_next.v_y, GRID_SIZE);
            memcpy(io_queue.U_buf[idx].v_z, U_next.v_z, GRID_SIZE);
            memcpy(io_queue.P_buf[idx].p,  pressure_next.p, GRID_SIZE);

            /* Signal that the next buffer is ready, so update .count */
            pthread_mutex_lock(&io_queue.mutex);
            io_queue.count++;
            pthread_cond_signal(&io_queue.not_empty);
            pthread_mutex_unlock(&io_queue.mutex);
        }

        if (full_output) {
            /* Store current solution in the record vectors */
            VelocityField U_copy;
            Pressure P_copy;
            initialize_velocity_field(&U_copy);
            initialize_pressure(&P_copy);

            memcpy(U_copy.v_x, U_next.v_x, GRID_SIZE);
            memcpy(U_copy.v_y, U_next.v_y, GRID_SIZE);
            memcpy(U_copy.v_z, U_next.v_z, GRID_SIZE);
            memcpy(P_copy.p,  pressure_next.p, GRID_SIZE);

            (*U_record)[t] = U_copy;
            (*P_record)[t] = P_copy;
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

    pthread_mutex_lock(&io_queue.mutex);
    io_queue.stop = 1;
    pthread_cond_signal(&io_queue.not_empty);
    pthread_mutex_unlock(&io_queue.mutex);

    pthread_join(io_thread, NULL);
    io_queue_destroy(&io_queue);


}