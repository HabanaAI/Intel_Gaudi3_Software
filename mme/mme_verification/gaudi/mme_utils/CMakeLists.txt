cmake_minimum_required(VERSION 3.5.1)
project(mme_utils_gaudi LANGUAGES CXX)
set(TARGET mme_utils_gaudi)


# compile all files with *.cpp suffix
file(GLOB SRC *.cpp)
#add headers
file(GLOB HDR *.h ../headers/*.h)
#group them nicely
source_group("Header files" FILES ${HDR})

# Build a library from all specified source files
add_library(${TARGET} STATIC ${SRC} ${HDR})
