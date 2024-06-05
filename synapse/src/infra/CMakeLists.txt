cmake_minimum_required(VERSION 3.5)

project(SynapseInfra CXX)

set(CMAKE_CXX_STANDARD 17)


file(GLOB HDR
    containers/*.h
    containers/*.hpp
    debug_utils/*.h
    memory_management/*.h
    recipe/*.hpp
    threads/*.h
    threads/*.hpp
    *.h
    *.hpp
    *.inl
)

file(GLOB SRC
    containers/*.cpp
    debug_utils/*.cpp
    recipe/*.cpp
    threads/*.cpp
    *.cpp
)

add_library(SynapseInfra STATIC ${SRC} ${HDR})

target_include_directories(SynapseInfra
    PUBLIC
        #[[ TODO: move headers to infra/include/infra to reduce colision with other projects ]]
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        $ENV{MME_ROOT}
)

target_link_libraries(SynapseInfra PRIVATE fma)