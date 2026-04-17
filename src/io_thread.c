#include "io_thread.h"

void io_queue_init(IOQueue *q)
{
    q->head  = 0;
    q->tail  = 0;
    q->count = 0;
    q->stop  = 0;
    for (int i = 0; i < IO_QUEUE_SIZE; i++)
        q->timestep[i] = -1;

    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);

    /* Initialize all buffers of the ring queue */
    for (int i = 0; i < IO_QUEUE_SIZE; ++i) {
        initialize_velocity_field(&q->U_buf[i]);
        initialize_pressure(&q->P_buf[i]);
    }
}

void io_queue_destroy(IOQueue *q)
{
    for (int i = 0; i < IO_QUEUE_SIZE; ++i) {
        free_velocity_field(&q->U_buf[i]);
        free_pressure(&q->P_buf[i]);
    }

    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
}

/* I/O Thread */
void *io_thread_func(void *arg)
{
    IOQueue *q = (IOQueue *)arg;

    while (1) {

        pthread_mutex_lock(&q->mutex);

        while (q->count == 0 && !q->stop) {
            pthread_cond_wait(&q->not_empty, &q->mutex);
        }

        /* If there are no buffers left and the solver thread has finished, stop */
        if (q->count == 0 && q->stop) {
            pthread_mutex_unlock(&q->mutex);
            break;  
        }

        /* Take the right buffers in position [tale] */
        int idx = q->tail;
        q->tail = (q->tail + 1) % IO_QUEUE_SIZE;
        q->count--;
        int ts = q->timestep[idx];

        pthread_cond_signal(&q->not_full);
        pthread_mutex_unlock(&q->mutex);

        /* Write in file */
        char filename[256];
        sprintf(filename, "output/solution_%06d.vti", ts);

        write_vti_file(filename, &q->U_buf[idx], &q->P_buf[idx]);
    }

    return NULL;
}
