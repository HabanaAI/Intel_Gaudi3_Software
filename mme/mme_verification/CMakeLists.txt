cmake_minimum_required(VERSION 3.5.1)

include(ExternalProject)

if (CMAKE_COMPILER_IS_GNUCC AND (NOT CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8.5) OR
    (CMAKE_SYSTEM_PROCESSOR MATCHES "^powerpc*" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc64*" ))
    message(WARNING "Not compiling mme test due to outdated GCC version")
    return()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(common)

include(sim_depend)
include(syn_depend)

add_subdirectory(common)

if(GAUDI_EN)
    add_subdirectory(gaudi) 
endif()
if(GAUDI2_EN)
    add_subdirectory(gaudi2)
endif()
if(GAUDI3_EN)
    add_subdirectory(gaudi3)
endif()
