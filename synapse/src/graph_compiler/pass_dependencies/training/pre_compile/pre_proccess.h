#pragma once

// clang-format off
// =========================== PRE_COMPILE_PRE_PROCCESS_GROUP =================================================== //
// group members
#define PRE_COMPILE_PRE_PROCCESS_GROUP_MEMBERS                  PASS_ID_GRAPH_VISUALIZATION_PRE,\
                                                                PASS_ID_CHECK_INPUT_PERSISTENCE,\
                                                                PASS_ID_VALIDATE_USER_MEMORY_SECTIONS,\
                                                                PASS_ID_REGISTER_MEM_COHERENCE

// group member dependencies
#define GRAPH_VISUALIZATION_PRE_DEPENDENCY_SET

#define CHECK_INPUT_PERSISTENCE_DEPENDENCY_SET                  PASS_ID_GRAPH_VISUALIZATION_PRE

#define VALIDATE_USER_MEMORY_SECTIONS_DEPENDENCY_SET            PASS_ID_GRAPH_VISUALIZATION_PRE,\
                                                                PASS_ID_CHECK_INPUT_PERSISTENCE

#define REGISTER_MEM_COHERENCE_DEPENDENCY_SET                   PASS_ID_GRAPH_VISUALIZATION_PRE,\
                                                                PASS_ID_VALIDATE_USER_MEMORY_SECTIONS

// clang-format on