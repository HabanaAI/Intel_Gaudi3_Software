#pragma once

enum MemInitType
{
    MEM_INIT_RANDOM_WITH_NEGATIVE,
    MEM_INIT_RANDOM_POSITIVE,
    MEM_INIT_FROM_INITIALIZER,
    MEM_INIT_ALL_ONES,
    MEM_INIT_ALL_ZERO,
    MEM_INIT_FROM_INITIALIZER_NO_CAST,
    MEM_INIT_RANDOM_WITH_NEGATIVE_ONLY,
    MEM_INIT_NONE,
    MEM_INIT_COMPILATION_ONLY  // used for tests that only compile the graph
};