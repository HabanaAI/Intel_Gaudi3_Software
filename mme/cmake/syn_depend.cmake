# headers required to use synapse mme serialization
if(SYN_DEPEND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYN_DEPEND=1")
    set(Synapse $ENV{BUILD_ROOT_LATEST}/libSynapse.so)

    include_directories($ENV{SYNAPSE_ROOT}/include/
                        $ENV{SYNAPSE_ROOT}/src
                        $ENV{SYNAPSE_ROOT}/src/graph_compiler/
                         )
endif()

