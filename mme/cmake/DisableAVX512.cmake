###############################################################################
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
################################################################################

macro (disable_avx512_if_needed)

    include(CheckCXXCompilerFlag)

    set(NOAVX512 "")
    
    if (VALGRIND_ON)
       # We only need to disable avx512f, then all the rest is disabled too
       message(STATUS "Disabling all avx512 extensions")
       check_cxx_compiler_flag("-mno-avx512f" HAS_FLAG_NOAVX512F)
       if (HAS_FLAG_NOAVX512F)
             set(NOAVX512 "-mno-avx512f")
       endif()
    endif()

endmacro()
