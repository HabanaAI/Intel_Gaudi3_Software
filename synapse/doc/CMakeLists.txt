# *****************************************************************************
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
# Author:
# tcohen@habana.ai
# ******************************************************************************
cmake_minimum_required(VERSION 3.5.1)

set(GEN_PDF 1)
if(DOC_MODE STREQUAL "ext")
    set(GEN_PDF 0)
endif()
set(PYTHON $ENV{SYNAPSE_ROOT}/doc/.doc_venv/bin/python3)
set(RST2PDF $ENV{SYNAPSE_ROOT}/doc/.doc_venv/bin/rst2pdf)
set(STYLESHEET $ENV{SYNAPSE_ROOT}/doc/common/style.style)

file(MAKE_DIRECTORY _auto_generated_docs)
set(GENERATED_DOC ${CMAKE_CURRENT_SOURCE_DIR}/_auto_generated_docs)

add_subdirectory(training)
add_subdirectory(inference)
