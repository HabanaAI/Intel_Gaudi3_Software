cmake_minimum_required(VERSION 3.5.1)
project(mme_user_gaudi2 LANGUAGES CXX)

set(TARGET mme_user_gaudi2)

# compile all files with *.cpp suffix
file(GLOB_RECURSE SRC *.cpp)
#add headers
file(GLOB_RECURSE HDR *.h ../headers/*.h)
#group them nicely
source_group("Header files" FILES ${HDR})

# Build a library from all specified source files
add_library(${TARGET} STATIC ${SRC} ${HDR})
target_include_directories(${TARGET} PUBLIC ${CORAL_INCLUDES} 
                                            ${CORAL_GAUDI2_INCLUDES})