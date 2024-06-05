cmake_minimum_required(VERSION 3.5.1)
project(mme_common LANGUAGES CXX)
set(TARGET mme_common)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(compilation_flags)
include(findCoralSim)

# compile all files with *.cpp suffix
file(GLOB SRC *.cpp)
#add headers
file(GLOB HDR *.h)
#group them nicely
source_group("Header files" FILES ${HDR})


# Build a library from all specified source files
add_library(${TARGET} STATIC ${SRC} ${HDR})
target_include_directories(${TARGET} PRIVATE ${CORAL_INCLUDES} $ENV{SPECS_ROOT_H9})
target_link_libraries(${TARGET} PUBLIC ${hl_gcfg})
# CORAL_GAUDI2_INCLUDES
