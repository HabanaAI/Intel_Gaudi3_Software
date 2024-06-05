cmake_minimum_required(VERSION 3.5.1)
project(mme_test_runner LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GAUDI3_TARGET mme_test_runner_gaudi3)
set(GAUDI2_TARGET mme_test_runner_gaudi2)
set(GAUDI_TARGET mme_test_runner_gaudi)

set (TEST_RUNNER_COMMON_INCLUDES $ENV{MME_ROOT}/mme_verification/common
                                 $ENV{MME_ROOT}/src/sync)



file (GLOB UNIT_TEST_FILES ../unit_tests/*.cpp)

function(build_gtest target_name test_file target_define target_lib add_unit_tests)
    set(FILELIST test_main.cpp
                 mme_test_base.cpp
                 ${test_file})
    if(add_unit_tests)
        list(APPEND FILELIST ${UNIT_TEST_FILES})
    endif()
    add_executable(${target_name} ${FILELIST})
    target_link_libraries(${target_name}
                          ${binary_dir}/googlemock/gtest/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest${Suffix}
                          ${binary_dir}/googlemock/gtest/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${Suffix}
			  fma)
    target_compile_definitions(${target_name} PUBLIC ${target_define})
    target_include_directories(${target_name} PRIVATE ${TEST_RUNNER_COMMON_INCLUDES})
    target_link_libraries(${target_name} ${target_lib})
    add_dependencies(${target_name} googletest)

    if(CMAKE_COMPILER_IS_GNUC OR CMAKE_COMPILER_IS_GNUCXX)
        target_link_libraries(${target_name} pthread)
    endif()


    if(COVERAGE_ENABLED)
        find_package(gcovr REQUIRED)
        target_link_libraries(${target_name} -lgcov)
        add_custom_target(coverage WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND ${GCOVR_EXECUTABLE} -r ${CMAKE_SOURCE_PATH}/src --xml -o ${CMAKE_BINARY_DIR}/coverage.xml)
        add_dependencies(coverage check)
    endif()
endfunction()

if (GAUDI3_EN)
    file(GLOB SRC mme_verification_tests_gaudi3.cpp
                  mme_verification_tests_common.cpp)
    build_gtest("${GAUDI3_TARGET}" "${SRC}"
                "-DGAUDI3_EN" "mme_test_gaudi3_lib" ON)
    target_include_directories(${GAUDI3_TARGET} PRIVATE $ENV{MME_ROOT}/mme_verification/gaudi3)
endif()

if (GAUDI2_EN)
    file(GLOB SRC mme_verification_tests_gaudi2.cpp
                  mme_verification_tests_common.cpp
		  )
    build_gtest("${GAUDI2_TARGET}" "${SRC}"
                "-DGAUDI2_EN" "mme_test_gaudi2_lib" ON)
    target_include_directories(${GAUDI2_TARGET} PRIVATE $ENV{MME_ROOT}/mme_verification/gaudi2)
endif()

if (GAUDI_EN)
    build_gtest("${GAUDI_TARGET}" "mme_verification_tests_gaudi.cpp"
                "-DGAUDI_EN" "mme_test_gaudi_lib" ON)
endif()
