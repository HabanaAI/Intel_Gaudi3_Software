#pragma once

enum TestCompilationMode
{
    COMP_GRAPH_MODE_TEST,
    COMP_EAGER_MODE_TEST,
    COMP_BOTH_MODE_TESTS  // Use to determine run of both kind of tests
};

//  To add new groups, always move them to the end of the list. The order is significant
enum TestPackage
{
    TEST_PACKAGE_DEFAULT,
    TEST_PACKAGE_CONV_PACKING,
    TEST_PACKAGE_AUTOGEN,
    TEST_PACKAGE_NODE_OPERATIONS,
    TEST_PACKAGE_RT_API,
    TEST_PACKAGE_BROADCAST,
    TEST_PACKAGE_COMPARE_TEST,
    TEST_PACKAGE_GEMM,
    TEST_PACKAGE_DMA,
    TEST_PACKAGE_TRANSPOSE,
    TEST_PACKAGE_CONVOLUTION,
    TEST_PACKAGE_SRAM_SLICING,
    TEST_PACKAGE_DSD,
    TEST_PACKAGE_EAGER,
    LAST
};
