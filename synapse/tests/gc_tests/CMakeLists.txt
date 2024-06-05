include(ExternalProject)

ExternalProject_Get_Property(googletest source_dir)
ExternalProject_Get_Property(googletest binary_dir)

set(GTEST_LIBS
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest${libSuffix}${Suffix}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gmock${libSuffix}${Suffix}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${libSuffix}${Suffix}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gmock_main${libSuffix}${Suffix}
)

set(GC_TESTS_COMMON_LIBS
    mme_reference # for cpu calculator
    fma           # for cpu calculator
    ${GTEST_LIBS}
)

add_subdirectory(unit_tests)
add_subdirectory(platform_tests)
