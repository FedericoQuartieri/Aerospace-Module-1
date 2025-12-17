/* forcing_parser.h */
#ifndef FORCING_PARSER_H
#define FORCING_PARSER_H

typedef double (*function)(double x,
                           double y,
                           double z,
                           double t,
                           int component); /* 0 -> fx, 1 -> fy, 2 -> fz */


                           
/* Parse forcing from file; returns NULL on error (prints to stderr). */
function parse_function(const char *filename);

/* Free TinyExpr expressions when no longer needed. */
void destroy_function(void);


#endif /* FORCING_PARSER_H */
