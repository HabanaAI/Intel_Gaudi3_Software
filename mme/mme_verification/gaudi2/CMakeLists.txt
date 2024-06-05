cmake_minimum_required(VERSION 3.5.1)
project(mme_sim_gaudi2 LANGUAGES CXX)

set(TARGET mme_sim_gaudi2)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(compilation_flags)
include(findCoralSim)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories($ENV{SPDLOG_ROOT})
include_directories($ENV{SPECS_ROOT})
include_directories($ENV{SPECS_EXT_ROOT})

add_subdirectory(mme_test)
add_subdirectory(mme_user)
