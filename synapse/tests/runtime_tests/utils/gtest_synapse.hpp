#pragma once
#include <gtest/gtest.h>
#undef GTEST_FATAL_FAILURE_
// in synapse tests all ASSSERT_... macros throw exceptions so there is no need to return
#define GTEST_FATAL_FAILURE_(message) GTEST_MESSAGE_(message, ::testing::TestPartResult::kFatalFailure)
