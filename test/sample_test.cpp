#include <gtest/gtest.h>
#include "constants.h"
#include "g_field.h"
#include "utils.h"

TEST(SampleTest, BasicAssertions) {
    // Expect two strings to be equal.
    EXPECT_STREQ("hello", "hello");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
}