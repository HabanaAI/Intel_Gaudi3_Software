cmake_minimum_required(VERSION 3.5.1)
project(dummyComplexGuid)

set(CMAKE_CXX_STANDARD 17)

include_directories($ENV{SPECS_EXT_ROOT})
add_library(dummyComplexGuid SHARED dummyComplexGuid.cpp dummyComplexGuid.h)
set(LIBRARY_OUTPUT_PATH $ENV{BUILD_ROOT_LATEST})
