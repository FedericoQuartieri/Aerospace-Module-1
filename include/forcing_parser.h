/* forcing_parser.h */
#ifndef FORCING_PARSER_H
#define FORCING_PARSER_H

typedef double (*forcing_function_t)(double x,
                                     double y,
                                     double z,
                                     double t,
                                     int component); /* 0 -> fx, 1 -> fy, 2 -> fz */

/* Parse forcing from file; returns NULL on error (prints to stderr). */
forcing_function_t parse_forcing_function(const char *filename);

/* Free TinyExpr expressions when no longer needed. */
void destroy_forcing_function(void);


#endif /* FORCING_PARSER_H */
