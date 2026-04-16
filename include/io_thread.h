#ifndef IO_THREAD_H
#define IO_THREAD_H

#include <pthread.h>
#include <stdio.h>
#include "write_vti_file.h"
#include "solve.h"  

#define IO_QUEUE_SIZE 8  /* Ring buffer dimension */

/* IO queue */
typedef struct {
    VelocityField U_buf[IO_QUEUE_SIZE];
    Pressure      P_buf[IO_QUEUE_SIZE];

    int head;   /* Index where solver writes in the queue */
    int tail;   /* Index where the I/O thread reads from the queue */
    int count;  /* Elements in the queue */

    int stop;   /* Flag to indicate that solver has finished */
    int timestep[IO_QUEUE_SIZE];

    pthread_mutex_t mutex;
    pthread_cond_t  not_empty;  /* Signal that indicates if the IO thread should read the buffers */
    pthread_cond_t  not_full;   /* Signal that says to the solver thread if the queue is full */
} IOQueue;

void io_queue_init(IOQueue *q);
void io_queue_destroy(IOQueue *q);
void *io_thread_func(void *arg);

#endif
