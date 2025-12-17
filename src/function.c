/* forcing_parser.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyexpr.h"
#include "function.h"

/* Context that holds compiled expressions and variable storage */
typedef struct {
    te_expr *expr[3];   /* expr[0] -> fx, expr[1] -> fy, expr[2] -> fz */
    double x;
    double y;
    double z;
    double t;
} ForcingContext;

/* Single global context for simplicity */
static ForcingContext g_forcing_ctx = {
    { NULL, NULL, NULL },
    0.0, 0.0, 0.0, 0.0
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

/* Free all compiled expressions in the global context */
static void free_forcing_context(void)
{
    int i;
    for (i = 0; i < 3; ++i) {
        if (g_forcing_ctx.expr[i]) {
            te_free(g_forcing_ctx.expr[i]);
            g_forcing_ctx.expr[i] = NULL;
        }
    }
}

/* This is the actual function that will be returned as a function pointer.
 * It uses the global context and evaluates the correct component.
 */
static double forcing_impl(double x, double y, double z, double t, int component)
{
    if (component < 0 || component > 2) {
        /* Out-of-range component: return 0.0 as a safe default */
        return 0.0;
    }

    if (!g_forcing_ctx.expr[component]) {
        /* No expression compiled for this component: treat as zero */
        return 0.0;
    }

    /* Update variables in the context */
    g_forcing_ctx.x = x;
    g_forcing_ctx.y = y;
    g_forcing_ctx.z = z;
    g_forcing_ctx.t = t;

    /* Evaluate TinyExpr expression for the requested component */
    return te_eval(g_forcing_ctx.expr[component]);
}

/* Parse the forcing file and return a function pointer.
 * File format:
 *   - Up to 3 non-empty, non-comment lines (in order): fx, fy, fz
 *   - Lines starting with '#' or empty lines are ignored.
 *   - Missing components are set to "0".
 */
function parse_function(const char *filename)
{
    FILE *f;
    char expr_buf[3][256];
    int have_expr[3] = {0, 0, 0};
    int count = 0;
    char line[256];

    int c, k;
    int err;
    te_expr *expr;

    /* Free any previous expressions in the global context */
    free_forcing_context();

    f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Cannot open forcing file: %s\n", filename);
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

    /* Bind TinyExpr variables to the fields of the global context.
     * Using an initializer ensures all other struct fields are zeroed
     * (important for newer versions of TinyExpr with extra fields).
     */
    te_variable vars[] = {
        { "x", &g_forcing_ctx.x },
        { "y", &g_forcing_ctx.y },
        { "z", &g_forcing_ctx.z },
        { "t", &g_forcing_ctx.t }
    };
    const int nvars = (int)(sizeof(vars) / sizeof(vars[0]));

    /* Compile each component expression */
    for (c = 0; c < 3; ++c) {
        err = 0;
        expr = te_compile(expr_buf[c], vars, nvars, &err);

        if (!expr) {
            /* Cleanup partially compiled expressions */
            for (k = 0; k < c; ++k) {
                if (g_forcing_ctx.expr[k]) {
                    te_free(g_forcing_ctx.expr[k]);
                    g_forcing_ctx.expr[k] = NULL;
                }
            }

            fprintf(stderr,
                    "Parse error in forcing component %d at position %d in expression: %s\n",
                    c, err, expr_buf[c]);
            return NULL;
        }

        g_forcing_ctx.expr[c] = expr;
    }

    /* Return pointer to the implementation function */
    return &forcing_impl;
}

/* Free TinyExpr expressions when done */
void destroy_function(void)
{
    free_forcing_context();
}
