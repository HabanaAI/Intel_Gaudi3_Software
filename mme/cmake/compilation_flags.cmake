set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Werror -Wno-sign-compare -Wno-array-bounds")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-overflow")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare -Wno-unused-variable -Wno-reorder -Wno-strict-aliasing")

    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable -Wno-invalid-offsetof")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces -Wno-overloaded-virtual")
    endif()
    # Debug\Release flags
    set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-ggdb -O3 -Wno-unused-result -D_FORTIFY_SOURCE=2 -DNDEBUG")

    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        if (NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-misleading-indentation")
        endif()
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument -Wno-missing-braces")
    endif()
else()
    # Visual Studio Compiler
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
endif()

# Sanitizer
if (SANITIZE_ON)
    set(SANITIZE "-fsanitize=undefined")
    set(SANITIZE "${SANITIZE} -fno-omit-frame-pointer -fsanitize=address")
endif()

# Valgrind
if (VALGRIND_ON)
    include(CheckIncludeFile)

    CHECK_INCLUDE_FILE("valgrind/memcheck.h" HAVE_VALGRIND_MEMCHECK)
    if (NOT HAVE_VALGRIND_MEMCHECK)
        message(FATAL_ERROR "valgrind/memcheck.h wasn't found")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_VALGRIND=1")
    message(STATUS "Enabling Valgrind")
endif()

# Enable/disable test coverage
if(COVERAGE_ENABLED)
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -fprofile-arcs -ftest-coverage")
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZE} ${NOAVX512}")

if (NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^powerpc*" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc64*" )) #PPC doesn't support march=native
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")

else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_POWER_PC_ -mcpu=powerpc64le -mpowerpc64")
endif()

