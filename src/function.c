/* function.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyexpr.h"
#include "function.h"
#include "constants.h"

/* Context that holds compiled expressions and variable storage */
struct FunctionContext {
    te_expr *expr[3];   /* expr[0] -> fx, expr[1] -> fy, expr[2] -> fz */
    double x;
    double y;
    double z;
    double t;
};

/* Remove final newline/carriage return characters from a string, if present */
static void trim_newline(char *s)
{
    size_t len;

    if (!s)
        return;

    len = strlen(s);
    while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r')) {
        s[len - 1] = '\0';
        --len;
    }
}

/* Free all compiled expressions in the context */
static void free_context(function_handle ctx)
{
    int i;
    if (!ctx) return;
    for (i = 0; i < 3; ++i) {
        if (ctx->expr[i]) {
            te_free(ctx->expr[i]);
            ctx->expr[i] = NULL;
        }
    }
}

/*
    Evaluates the function delta_t for the given component as function(t) - function(t-1)
    If t==0 it returns the function(t==0) 
*/
double eval_delta_function(function_handle handle, double x, double y, double z, double t, int component)
{
    if (!handle || component < 0 || component > 2) {
        /* Invalid handle or out-of-range component: return 0.0 as a safe default */
        return 0.0;
    }

    if (!handle->expr[component]) {
        /* No expression compiled for this component: treat as zero */
        return 0.0;
    }

    /* Update variables in the context */
    handle->x = x;
    handle->y = y;
    handle->z = z;
    handle->t = t;

    /* Evaluate TinyExpr expression for the requested component at time t */
    double funct_t = te_eval(handle->expr[component]);
    double funct_t_prev;

    if(t == 0){
        return funct_t;
    } else {
        /* Update and evaluate at time t - DT !! since t is already in the physical domain */
        handle->t = t - DT;
        funct_t_prev = te_eval(handle->expr[component]);

        return (funct_t - funct_t_prev);
    }
}

/* Evaluate the function for the given component */
double eval_function(function_handle handle, double x, double y, double z, double t, int component)
{
    if (!handle || component < 0 || component > 2) {
        /* Invalid handle or out-of-range component: return 0.0 as a safe default */
        return 0.0;
    }

    if (!handle->expr[component]) {
        /* No expression compiled for this component: treat as zero */
        return 0.0;
    }

    /* Update variables in the context */
    handle->x = x;
    handle->y = y;
    handle->z = z;
    handle->t = t;

    /* Evaluate TinyExpr expression for the requested component */
    return te_eval(handle->expr[component]);
}

/* Parse function from file and return a handle.
 * File format:
 *   - Up to 3 non-empty, non-comment lines (in order): fx, fy, fz
 *   - Lines starting with '#' or empty lines are ignored.
 *   - Missing components are set to "0".
 */
function_handle parse_function(const char *filename)
{
    FILE *f;
    char expr_buf[3][512];
    int have_expr[3] = {0, 0, 0};
    int count = 0;
    char line[512];
    function_handle ctx;

    int c, k;
    int err;
    te_expr *expr;

    /* Allocate a new context */
    ctx = (function_handle)calloc(1, sizeof(struct FunctionContext));
    f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Cannot open function file: %s\n", filename);
        free(ctx);
        return NULL;
    }

    /* Read up to 3 non-empty, non-comment lines.
     * Each valid line corresponds to fx, fy, fz in order.
     */
    while (count < 3 && fgets(line, sizeof(line), f) != NULL) {
        char *p = line;

        /* Trim newline(s) */
        trim_newline(p);

        /* Skip leading spaces and tabs */
        while (*p == ' ' || *p == '\t')
            ++p;

        /* Skip empty lines or comment lines starting with '#' */
        if (*p == '\0' || *p == '#')
            continue;

        strncpy(expr_buf[count], p, sizeof(expr_buf[count]) - 1);
        expr_buf[count][sizeof(expr_buf[count]) - 1] = '\0';
        have_expr[count] = 1;
        ++count;
    }

    fclose(f);

    /* If fewer than 3 components are provided, set the remaining to "0" */
    for (c = 0; c < 3; ++c) {
        if (!have_expr[c]) {
            strcpy(expr_buf[c], "0");
            have_expr[c] = 1;
        }
    }

    /* Bind TinyExpr variables to the fields of the context.
     * Using an initializer ensures all other struct fields are zeroed
     * (important for newer versions of TinyExpr with extra fields).
     */
    te_variable vars[] = {
        { "x", &ctx->x },
        { "y", &ctx->y },
        { "z", &ctx->z },
        { "t", &ctx->t }
    };
    const int nvars = (int)(sizeof(vars) / sizeof(vars[0]));

    /* Compile each component expression */
    for (c = 0; c < 3; ++c) {
        err = 0;
        expr = te_compile(expr_buf[c], vars, nvars, &err);

        if (!expr) {
            /* Cleanup partially compiled expressions */
            for (k = 0; k < c; ++k) {
                if (ctx->expr[k]) {
                    te_free(ctx->expr[k]);
                    ctx->expr[k] = NULL;
                }
            }
            free(ctx);

            fprintf(stderr,
                    "Parse error in function component %d at position %d in expression: %s\n",
                    c, err, expr_buf[c]);
            return NULL;
        }

        ctx->expr[c] = expr;
    }

    /* Return the context handle */
    return ctx;
}

/* Free resources associated with a function handle */
void destroy_function(function_handle handle)
{
    if (!handle) return;
    free_context(handle);
    free(handle);
}
