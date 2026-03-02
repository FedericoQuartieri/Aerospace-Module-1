/* function.h */
#ifndef FUNCTION_H
#define FUNCTION_H

#include "function.h"

/* Opaque handle for a parsed function */
typedef struct FunctionContext *function_handle;

/* Parse function from file; returns NULL on error (prints to stderr). */
function_handle parse_function(const char *filename);

/* Evaluate the function for the given component (0 -> x, 1 -> y, 2 -> z) */
double eval_function(function_handle handle, double x, double y, double z, double t, int component);

/* Evaluate the function in the delta (t, t-1) */
double eval_delta_function(function_handle handle, double x, double y, double z, double t, int component);

/* Free resources associated with a function handle */
void destroy_function(function_handle handle);

#endif /* FUNCTION_H */
