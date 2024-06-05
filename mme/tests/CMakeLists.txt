cmake_minimum_required(VERSION 3.5.1)
project(mme_unit_tests LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(ExternalProject)
#add gtest
ExternalProject_Add(
    googletest
    URL "$ENV{HABANA_SOFTWARE_STACK}/3rd-parties/googletest/"
    TIMEOUT 10
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -Dgtest_force_shared_crt=ON
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON)

ExternalProject_Get_Property(googletest source_dir)
ExternalProject_Get_Property(googletest binary_dir)
set_target_properties(googletest PROPERTIES EXCLUDE_FROM_ALL TRUE)

if(MSVC)
  set(Suffix ".lib")
  set (libPath ${CMAKE_BUILD_TYPE}"/")
else()
  set(Suffix ".a")
  set(Pthread "-pthread")
  set (libPath "")
endif()

include(compilation_flags)
# remove redundant warning from tests
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -Wno-sign-compare -Wno-strict-aliasing")

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}
                     $ENV{MME_ROOT}/src/
                     $ENV{MME_ROOT}/mme_reference
                     ${source_dir}/googletest/include)


add_subdirectory(mme_tests)

