#pragma once
// clang-format off
// ================================================================================================================== //
/*
    backend passes related to lowering and optimizing memcopies.
*/
// group members
#define BE_GC_BRAIN_MEMORY_GROUP_MEMBERS                        PASS_ID_SELECT_MEMCPY_ENGINE,\
                                                                PASS_ID_INSERT_NAN_INF_PROBE,\
                                                                PASS_ID_OPTIMIZE_MEMCPY_NODES

// group member dependencies

#define SELECT_MEMCOPY_ENGINE_DEPENDENCY_SET

#define INSERT_NAN_INF_PROBE_DEPENDENCY_SET                     PASS_ID_SELECT_MEMCPY_ENGINE

#define OPTIMIZE_MEMCPY_NODES_DEPENDENCY_SET                    PASS_ID_SELECT_MEMCPY_ENGINE

// ================================================================================================================== //
// ################################################################################################################## //
// clang-format on