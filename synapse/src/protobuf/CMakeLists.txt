
MESSAGE("Protobuf autogeneration STARTED")

file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")

set(ProtobufBuildDir ${CMAKE_BINARY_DIR}/../ThirdParties/protobuf/build)

# Ugly hack for deleting old proto-file's products (upon OLD folders' structure)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/recipe.pb.h)
    execute_process(COMMAND rm ${CMAKE_CURRENT_SOURCE_DIR}/recipe.pb.h)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/recipe.pb.cc)
    execute_process(COMMAND rm ${CMAKE_CURRENT_SOURCE_DIR}/recipe.pb.cc)
endif()

foreach(SINGLE_PROTO_FILE  ${ProtoFiles})
    message( "Proto file: " ${SINGLE_PROTO_FILE} )
    string(REGEX REPLACE "\\.[^.]*$" "" PROTO_FILE_WITHOUT_EXT ${SINGLE_PROTO_FILE})
    string(REGEX REPLACE "(protobuf).*$" "" PROTO_FILE_BASE_PATH ${SINGLE_PROTO_FILE})
    message( "Proto path: " ${PROTO_FILE_BASE_PATH} )

    if(NOT EXISTS ${PROTO_FILE_WITHOUT_EXT}.pb.h)
        execute_process(COMMAND ${ProtobufBuildDir}/bin/protoc -I=${PROTO_FILE_BASE_PATH} --cpp_out=${PROTO_FILE_BASE_PATH} ${SINGLE_PROTO_FILE})
        MESSAGE("${SINGLE_PROTO_FILE} was generated for the first time")
    elseif(${SINGLE_PROTO_FILE} IS_NEWER_THAN ${PROTO_FILE_WITHOUT_EXT}.pb.h)
        execute_process(COMMAND ${ProtobufBuildDir}/bin/protoc -I=${PROTO_FILE_BASE_PATH} --cpp_out=${PROTO_FILE_BASE_PATH} ${SINGLE_PROTO_FILE})
        MESSAGE("${SINGLE_PROTO_FILE} was generated due to update")
    endif()
endforeach()

MESSAGE("Protobuf autogeneration FINISHED")