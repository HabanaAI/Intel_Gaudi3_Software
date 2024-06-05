#pragma once

// clang-format off
// ================================================================================================================== //
/*
    backend mme optimization passes sub-group.

*/
// group members
#define BE_MME_OPTIMIZATION_GROUP_MEMBERS                                      PASS_ID_INIT_MME_BRAIN_IFC,\
                                                                               PASS_ID_MME_CONCURRENCY_IDENTIFIER,\
                                                                               PASS_ID_LOWER_DEDX,\
                                                                               PASS_ID_PACKING_MME_NODES

// group member dependencies
#define INIT_MME_BRAIN_IFC_DEPENDENCY_SET

#define MME_CONCURRENCY_IDENTIFIER_DEPENDENCY_SET                              PASS_ID_INIT_MME_BRAIN_IFC

#define PACKING_MME_NODES_DEPENDENCY_SET                                       PASS_ID_MME_CONCURRENCY_IDENTIFIER

#define LOWER_DEDX_DEPENDENCY_SET                                              PASS_ID_PACKING_MME_NODES

// ================================================================================================================== //
// ################################################################################################################## //

// clang-format on