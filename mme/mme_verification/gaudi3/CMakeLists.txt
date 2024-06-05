cmake_minimum_required(VERSION 3.5.1)
project(mme_sim_gaudi3 LANGUAGES CXX)

set(TARGET mme_sim_gaudi3)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(compilation_flags)
include(findCoralSim)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories($ENV{SPECS_ROOT_H9})

add_subdirectory(mme_test)
add_subdirectory(mme_user)
