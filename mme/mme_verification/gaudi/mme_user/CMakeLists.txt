cmake_minimum_required(VERSION 3.5.1)
project(mme_user_gaudi LANGUAGES CXX)
# Specify .cpp files
set(TARGET mme_user_gaudi)


# compile all files with *.cpp suffix
file(GLOB SRC *.cpp)
#add headers
file(GLOB HDR *.h ../headers/*.h)
#group them nicely
source_group("Header files" FILES ${HDR})


# Build a library from all specified source files
add_library(${TARGET} STATIC ${SRC} ${HDR})
target_include_directories(${TARGET} PUBLIC ${CORAL_INCLUDES} 
                                            ${CORAL_GAUDI_INCLUDES})
target_link_libraries(${TARGET} PRIVATE mme_common)
