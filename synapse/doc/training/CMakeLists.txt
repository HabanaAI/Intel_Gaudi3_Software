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

set(asic training)

set(rst_out ${GENERATED_DOC}/synapse_doc_${asic}.rst)
set(pdf_out ${GENERATED_DOC}/synapse_doc_${asic}.pdf)
set(output ${rst_out})

if (RMAUTOGEN)
    execute_process(COMMAND rm -rf ${rst_out} ${pdf_out})
endif()

set(doc_gen_commands COMMAND protoc --python_out=$ENV{SYNAPSE_ROOT}/doc/common/protobuf --proto_path=$ENV{SYNAPSE_ROOT}/doc/common/protobuf $ENV{SYNAPSE_ROOT}/doc/common/protobuf/op_def.proto)
list(APPEND doc_gen_commands COMMAND ${PYTHON} $ENV{SYNAPSE_ROOT}/doc/common/opdef2rst.py ${asic} ${DOC_MODE})

if(GEN_PDF)
    set(output ${pdf_out})
    list(APPEND doc_gen_commands COMMAND ${RST2PDF} ${rst_out} ${pdf_out} -s ${STYLESHEET} -e dotted_toc)
endif()

# Generate a list of all operator proto texts for compilation dependencies.
file(GLOB_RECURSE protoList $ENV{SYNAPSE_ROOT}/doc/${asic}/op_def/*.pbtxt)

add_custom_command(OUTPUT ${output}
                   DEPENDS $ENV{SYNAPSE_ROOT}/doc/common/opdef2rst.py ${protoList} $ENV{SYNAPSE_ROOT}/doc/common/protobuf/op_def.proto
                   ${doc_gen_commands}
                   COMMENT "Generating ${output}")

add_custom_target( doc_${asic} ALL
                   DEPENDS ${output})
