cmake_minimum_required(VERSION 3.5.1)
project(SynapseMme LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# add hl_thunk and func sim if needed.
include(sim_depend)

# add synapse sources if needed.
include(syn_depend)

# compile all files with *.cpp suffix
file(GLOB SRC  utils/*.cpp
               gaudi/*.cpp
               gaudi/new_descriptor_generator/*.cpp
               gaudi2/*.cpp
               gaudi3/*.cpp
               mme_common/*.cpp
               )
#add headers
file(GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h
              utils/*.h
              gaudi/*.h
              gaudi/new_descriptor_generator/*.h
              gaudi2/*.h
              gaudi3/*.h
              mme_common/*.h)

source_group("Header files" FILES ${HDR} ${SYNAPSE_HDR}) #group them nicely

# set compilation flags
include(compilation_flags)


add_library(mme_stack_obj OBJECT ${SRC} ${SYNAPSE_HDR} ${HDR})
target_include_directories(mme_stack_obj PRIVATE $ENV{SPECS_ROOT_H9})


set(LIBS ${hl_logger} ${hl_gcfg})

add_library(mme_stack STATIC $<TARGET_OBJECTS:mme_stack_obj>)
target_link_libraries(mme_stack PUBLIC ${LIBS})


add_library(SynapseMme SHARED $<TARGET_OBJECTS:mme_stack_obj>)
target_link_libraries(SynapseMme PUBLIC fma ${LIBS})
