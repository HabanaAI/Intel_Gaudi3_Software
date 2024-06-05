#pragma once

// clang-format off
// ================================================================================================================== //
/*
    backend brain passes post-slicing sub-group.
    handles nodes that should be handled in the post-slicing stage.
*/
// group members
#define BE_GC_BRAIN_POST_SLICING_GROUP_MEMBERS                  PASS_ID_MME_CONCURRENCY,\
                                                                PASS_ID_MME_CONCURRENCY_MEMSET,\
                                                                PASS_ID_FUSE_SPILL_FILL,\
                                                                PASS_ID_CONVERT_1X1BATCH_GEMM_TO_GEMM,\
                                                                PASS_ID_MEMSET_NODE_OUTPUT,\
                                                                PASS_ID_FUSE_BATCH_NORM_MEMCPY, \
                                                                PASS_ID_VALIDATE_PRE_SLICING_SIZES, \
                                                                PASS_ID_ALIGN_BPT_FCD_STRIDE_TO_CACHELINE

// group member dependencies
#define MME_CONCURRENCY_MEMSET_DEPENDENCY_SET                   PASS_ID_VALIDATE_PRE_SLICING_SIZES

// gaudi1 only pass
#define MME_CONCURRENCY_DEPENDENCY_SET                          PASS_ID_VALIDATE_PRE_SLICING_SIZES

#define FUSE_SPILL_FILL_DEPENDENCY_SET                          PASS_ID_VALIDATE_PRE_SLICING_SIZES

#define MEMSET_NODE_OUTPUT_DEPENDENCY_SET                       PASS_ID_CONVERT_1X1BATCH_GEMM_TO_GEMM

#define CONVERT_1X1BATCH_GEMM_TO_GEMM_DEPENDENCY_SET            PASS_ID_MME_CONCURRENCY_MEMSET

#define FUSE_BATCH_NORM_MEMCPY_DEPENDENCY_SET                   PASS_ID_CONVERT_1X1BATCH_GEMM_TO_GEMM

#define VALIDATE_PRE_SLICING_SIZES_DEPENDENCY_SET

#define ALIGN_BPT_FCD_STRIDE_TO_CACHELINE_DEPENDENCY_SET

// ================================================================================================================== //
// ################################################################################################################## //
// clang-format on
