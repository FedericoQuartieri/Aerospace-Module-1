#include "solve.h"
#include "io_thread.h"


void solve (GField g_field, const Data *data, Pressure pressure, DTYPE* K, 
            VelocityField Eta, VelocityField Zeta, VelocityField U, 
            DTYPE* Beta, DTYPE* Gamma, 
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

    /* save first timestep, U(t=0) is given by the problem (still need to be added in main, now is done by the manufactured test)*/
    char filename[256];
    sprintf(filename, "output/solution_%06d.vti", 0);
    write_vti_file(filename, &U, &pressure);

    // Initialize the necessary velocity and force fields
    VelocityField Xi;
    VelocityField Delta;
    ForceField rhs;
    
    initialize_velocity_field(&Xi);
    initialize_velocity_field(&Delta);
    initialize_force_field(&rhs);

    /* TODO this should be done better */
    Pressure pressure_star;
    initialize_pressure(&pressure_star);
    memcpy(pressure_star.p, pressure.p, GRID_SIZE);
    

    /* 
        t=0 is set as the exact solution by definition, so we should 
        start to solve at t = 1 
    */
    for (int t = 1; t <= STEPS; t++) {
        //memset(pressure_star.p, 0, GRID_SIZE); // Test zero pressure

        /* g(t) is computed with forcing(t-1/2) and velocity(t-1) */
        compute_g(&g_field, &pressure_star, K, &Eta, &Zeta, &U, t, data);        

        /* here we set all the boundary as the delta of boundary(t) - boundary(t-1) */
        solve_momentum_system(U, Eta, Zeta, Xi, g_field, Delta, rhs, Beta, Gamma, data, t);
        
        solve_pressure_system(U, &pressure, &pressure_star);

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
            memcpy(io_queue.U_buf[idx].v_x, U.v_x, GRID_SIZE);
            memcpy(io_queue.U_buf[idx].v_y, U.v_y, GRID_SIZE);
            memcpy(io_queue.U_buf[idx].v_z, U.v_z, GRID_SIZE);
            memcpy(io_queue.P_buf[idx].p,  pressure.p, GRID_SIZE);

            /* Signal that the next buffer is ready, so update .count */
            pthread_mutex_lock(&io_queue.mutex);
            io_queue.count++;
            pthread_cond_signal(&io_queue.not_empty);
            pthread_mutex_unlock(&io_queue.mutex);
        }

        if (full_output) {
            // Store current solution in record vectors
            VelocityField U_copy;
            Pressure P_copy;
            initialize_velocity_field(&U_copy);
            initialize_pressure(&P_copy);

            memcpy(U_copy.v_x, U.v_x, GRID_SIZE);
            memcpy(U_copy.v_y, U.v_y, GRID_SIZE);
            memcpy(U_copy.v_z, U.v_z, GRID_SIZE);
            memcpy(P_copy.p, pressure.p, GRID_SIZE);

            (*U_record)[t] = U_copy;
            (*P_record)[t] = P_copy;
        }
    }

    free_velocity_field(&Xi);
    free_velocity_field(&Delta);
    free_force_field(&rhs);

    pthread_mutex_lock(&io_queue.mutex);
    io_queue.stop = 1;
    pthread_cond_signal(&io_queue.not_empty);
    pthread_mutex_unlock(&io_queue.mutex);

    pthread_join(io_thread, NULL);
    io_queue_destroy(&io_queue);

}