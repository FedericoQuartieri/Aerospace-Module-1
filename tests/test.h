#ifndef TEST_H
#define TEST_H

#include <stdio.h>

enum TestStatus { SUCCESS, FAILURE };

#define DEF_TEST(name, ...)                                 \
    void _test_##name(enum TestStatus *_test_status,        \
                      ##__VA_ARGS__)                        \

#define RUN_TEST(name, ...)                                 \
do {                                                        \
    enum TestStatus result = SUCCESS;                       \
    _test_##name(&result, ##__VA_ARGS__);                   \
    if (result == SUCCESS) {                                \
        printf("[SUCCESS] " #name "(" #__VA_ARGS__ ")\n");  \
    } else {                                                \
        printf("[FAILURE] " #name "(" #__VA_ARGS__ ")\n");  \
    }                                                       \
} while (0)

#define EXPECT_SUCCESS(name, ...)                           \
do {                                                        \
    enum TestStatus result = SUCCESS;                       \
    _test_##name(&result, ##__VA_ARGS__);                   \
    if (result != SUCCESS) {                                \
        printf(" EXPECT_SUCCESS() FAILED at %s:%d\n",       \
               __FILE__, __LINE__);                         \
        *_test_status = FAILURE;                            \
    }                                                       \
} while (0)

#define EXPECT_TRUE(condition)                              \
do {                                                        \
    if (!(condition)) {                                     \
        printf(" EXPECT_TRUE() FAILED at %s:%d\n",          \
               __FILE__, __LINE__);                         \
        *_test_status = FAILURE;                            \
    }                                                       \
} while (0)

#define EXPECT_EQUAL(x, y)                                  \
do {                                                        \
    if ((x) != (y)) {                                       \
        printf(" EXPECT_EQUAL() FAILED at %s:%d\n",         \
               __FILE__, __LINE__);                         \
        *_test_status = FAILURE;                            \
    }                                                       \
} while (0)

#define _ABS(x) ((x) > 0 ? (x) : -(x))

#define EXPECT_EQUALF(x, y, tol)                            \
do {                                                        \
    if (_ABS((x) - (y)) > tol) {                            \
        /*printf(" EXPECT_EQUALF(%f %f) FAILED at %s:%d\n", */   \
        /*       (x), (y), __FILE__, __LINE__);             */   \
        *_test_status = FAILURE;                            \
    }                                                       \
} while (0)

#define ASSERT_EQUALF(x, y, tol)                            \
do {                                                        \
    if (_ABS((x) - (y)) > (tol)) {                          \
        printf(" ASSERT_EQUALF(%g %g) FAILED at %s:%d\n",        \
               (x), (y), __FILE__, __LINE__);                         \
        *_test_status = FAILURE;                            \
        return;                                             \
    }                                                       \
} while (0)

#define ASSERT_TRUE(condition)                              \
do {                                                        \
    if (!(condition)) {                                     \
        printf(" ASSERT_TRUE() FAILED at %s:%d\n",          \
               __FILE__, __LINE__);                         \
        *_test_status = FAILURE;                            \
        return;                                             \
    }                                                       \
} while (0)

#define SUCCEED_TEST() return;
#define FAIL_TEST() do { *_test_status = FAILURE; return; } while (0)

#endif
