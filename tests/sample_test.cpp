#include <gtest/gtest.h>
#include "../defines.h"
#include "../g.h"
#include "../helpers.h"

TEST(SampleTest, BasicAssertions) {
    // Expect two strings to be equal.
    EXPECT_STREQ("hello", "hello");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
}