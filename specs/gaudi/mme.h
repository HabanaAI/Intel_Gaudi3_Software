#pragma once

#include <stdint.h>

#pragma pack(push, 4)

namespace Mme
{
    enum EMmeCore
    {
        MME_CORE_NW = 0,
        MME_CORE_SW = 1,
        MME_CORE_NE = 2,
        MME_CORE_SE = 3,

        MME_CORES_NR,
        MME_MASTERS_NR = MME_CORES_NR / 2,

        MME_CORE_NORTH_MASTER = MME_CORE_NW,
        MME_CORE_SOUTH_MASTER = MME_CORE_SW,
        MME_CORE_NORTH_SLAVE = MME_CORE_NE,
        MME_CORE_SOUTH_SLAVE = MME_CORE_SE,
    };

    typedef int16_t bf16_t;
    typedef int32_t f32_t;

    static const unsigned c_mme_max_conv_dims = 4;
    static const unsigned c_mme_max_tensor_dims = 5;
    static const unsigned c_mme_accums_nr = 16;
    static const unsigned c_mme_lfsr_seeds_nr = 32;
    static const unsigned c_mme_matrix_size = 64;
    static const unsigned c_mme_sb_size = 512;
    static const unsigned c_mme_max_sb_reuse = 240;


    typedef enum
    {
        e_mme_gemm_loop = (1 << 0) - 1,  // Single gemmm
        e_mme_conv_loop_0 = (1 << 1) - 1,  // Single loop of gemms
        e_mme_conv_loop_1 = (1 << 2) - 1,  // Two loops of gemms
        e_mme_conv_loop_2 = (1 << 3) - 1,  // Three loops of gemms.
        e_mme_conv_loop_3 = (1 << 4) - 1,  // four loops of gemms - Every Tetris.
        e_mme_tetris_loop = (1 << 5) - 1,  // Loop of Tetrises
        e_mme_outer_loop = (1 << 6) - 1,  // Two loops of Tetrises
    } EMmeLoopMask;

    typedef enum
    {
        e_mme_dt_bf = 0x0,
        e_mme_dt_sp = 0x1,
    } EMmeDataType;

    typedef enum
    {
        e_mme_rm_rn = 0x0, // round half to nearest even
        e_mme_rm_rz = 0x1, // round to zero
        e_mme_rm_ru = 0x2, // round up
        e_mme_rm_rd = 0x3, // round down
        e_mme_rm_rs = 0x4, // stochastic rounding 

        e_mme_rm_rsn = e_mme_rm_rn | e_mme_rm_rs,
        e_mme_rm_rsz = e_mme_rm_rz | e_mme_rm_rs,
        e_mme_rm_rsu = e_mme_rm_ru | e_mme_rm_rs,
        e_mme_rm_rsd = e_mme_rm_rd | e_mme_rm_rs,
    } EMmeRoundingMode;

    typedef enum
    {
        e_mme_local = 0x0,
        e_mme_remote = 0x1,
        e_mme_local_and_remote = 0x2,
    } MmeHalf;

    typedef union _MmeHeader
    {
        struct {
            // byte 0
            uint32_t transS : 1;               // When set, the shared operand is transposed.
            uint32_t transL : 1;               // When set, the local operand is transposed.
            uint32_t transO : 1;               // When set, the output operand is transposed.
            uint32_t advanceS : 1;             // Advance the shared operand in the outer conv loop.
            uint32_t advanceL : 1;             // Advance the local operand in the outer conv loop.
            uint32_t advanceO : 1;             // Advance the output operand in the outer conv loop.
            uint32_t lowerL : 1;               // lower the local operand.
            uint32_t lowerS : 1;               // lower the shared operand.

                                               // byte 1
            uint32_t accumMask : 4;            // EMmeLoopMask value. Bit mask of loops to accumulate in the GRF.
                                               // "1" means the loop is accumulated.
                                               // The field is 4 bits wide (and not 5) since the tetris loop
                                               // is always outer to the conv and is never accumulated.
            uint32_t accStoreIncDisable : 1;   // Avoid incrementing the store acc counter.
            uint32_t roundingMode : 3;         // Rounding mode. (Mme::EMmeRoundingMode)

                                               // byte 2
            uint32_t dataTypeIn : 1;           // The data type of the input operands. (EMmeDataType)
            uint32_t dataTypeOut : 1;          // The data type of the output operands. (EMmeDataType)
            uint32_t accum : 1;                // Accumulate output in the accumulator.
            uint32_t storeEn : 1;              // Store the output.
            uint32_t rollAccums : 4;           // The number of accumolator inc to do after the last rollup.

                                               // byte 3
            uint32_t signalMask : 6;           // EMmeLoopMask value bit mask that specifies when to signal the
                                               // sync object. signalMask must never be less than accumMask.
            uint32_t signalEn : 1;             // Enable signaling.
            uint32_t reluEn : 1;               // Enable RELU.

                                               // byte 4-7
            uint32_t partialHeightLoopS : 6;   // enable bit per loop. 
                                               // in transpose the loops that use partial height in their last iteration.
                                               // in non-transpose - the loops that use partial dense FCD.

            uint32_t partialHeightLoopLLocal : 6;   // enable bit per loop. 
                                                    // in transpose the loops that use partial height in their last iteration.
                                                    // in non-transpose - the loops that use partial dense FCD.

            uint32_t partialHeightLoopLRemote : 6;   // enable bit per loop. 

                                                     // in transpose the loops that use partial height in their last iteration.
                                                     // in non-transpose - the loops that use partial dense FCD.

            uint32_t partialHeightLoopOLocal : 6;    // enable bit per loop. the loop that use partaial dense FCD.
            uint32_t partialHeightLoopORemote : 6;   // enable bit per loop. the loop that use partaial dense FCD.
            uint32_t fpEn : 1;                       // reserved. 
            uint32_t euBEn : 1;                      // enable the eu brain.
        };

        uint8_t bytes[8];
        uint32_t dw[2];
        uint64_t ddw;
    } MmeHeader;

    typedef struct _MmeTensorDesc
    {                                                                     // Index '0' is the fastest changing dim.
        uint32_t  validElements[c_mme_max_tensor_dims];                   // The number of valid elements.
        int32_t   loopStride[c_mme_max_tensor_dims];                      // The offset in which the ROI base should be shifted between loop iterations.
        int32_t   roiSize[c_mme_max_tensor_dims - 1];                     // The size of the ROI.
        uint32_t  spatialStrides[c_mme_max_tensor_dims - 1];              // The strides of the spatial dimensions.
        uint32_t  spatialSizeMinus1;                                      // Out - the number of rows in the last tetris.
                                                                          // In transposed - the number of rows in the last loop. (partialHeightLoopS/L)
                                                                          // In not transposed - The total number of spatial rows. 
    } MmeTensorDesc;


    typedef struct _MmeAguCoreDesc
    {
        int32_t roiBaseOffset[c_mme_max_tensor_dims];   // The walk ROI base offset.
        uint32_t startOffset[c_mme_max_tensor_dims - 1]; // The dimension's start offset.
    } MmeAguCoreDesc;

    typedef union _MmeAssociatedDims
    {
        struct
        {
            uint16_t dimS : 3;
            uint16_t dimL : 3;
            uint16_t dimO : 3;
            uint16_t reserved : 7;
        };
        uint16_t w;
    } MmeAssociatedDims;

    typedef union _MmeKernelSize
    {
        uint8_t dim[c_mme_max_conv_dims];
        uint32_t dw;
    } MmeKernelSize;

    typedef struct _MmeConvDesc
    {
        MmeKernelSize kernelSizeMinus1;                         // The kernel size.
        MmeAssociatedDims associatedDims[c_mme_max_conv_dims]; // The dims associated with each loop in the conv loop.
    } MmeConvDesc;

    typedef union _MmeOuterLoop
    {
        struct
        {
            MmeAssociatedDims associatedDims;
            uint8_t sizeMinus1;
            uint8_t reserved;
        };
        uint32_t dw;
    } MmeOuterLoop;

    typedef union _MmeSyncObject
    {
        struct {
            // dw 0
            uint32_t addrLow[e_mme_local_and_remote];   // Lower 32 bits of the sync object address (for local and remote).
            uint32_t addrHigh;

            // dw 1       
            union
            {
                struct
                {
                    uint32_t value : 15;
                    uint32_t reserved : 15;
                    uint32_t perfEn : 1;
                    uint32_t operation : 1;
                };
                uint32_t data;
            };
        };

        uint32_t dw[2];
        uint64_t ddw;
    } MmeSyncObject;

    typedef union MmeUserData
    {
        struct
        {
            uint32_t first : 9;
            uint32_t steady : 9;
            uint32_t mask : 6;
            uint32_t reserved : 8;
        };
        uint32_t dw;
    } MmeUserData;

    typedef struct _MmeMetaData
    {
        uint32_t aguS;
        uint32_t aguL[e_mme_local_and_remote];
        uint32_t aguO[e_mme_local_and_remote];
    } MmeMetaData;

    typedef union _MmeRateLimeter
    {
        struct
        {
            uint32_t aguS : 8;
            uint32_t aguL : 8;
            uint32_t aguO : 8;
            uint32_t reserved : 8;
        };
        uint32_t dw;
    } MmeRateLimeter;

    typedef union MmePerfEvt
    {
        struct
        {
            uint32_t value : 16;
            uint32_t rst : 1;
            uint32_t incMask : 1;
            uint32_t startEndMask : 2;    // start bit 0, end bit 1
            uint32_t loopMask : 6;
            uint32_t reserved : 4;
        };
        uint32_t dw;
    } MmePerfEvt;

    typedef union _MmeSBRepeat
    {
        struct
        {
            uint32_t repeatSMinus1 : 8;
            uint32_t aguSLoopMask : 6;  // mask of loops that are disabled in AGU S. 
            uint32_t loadS : 1;         // enable AGU-S.
            uint32_t teEnS : 1;         // enable TE-S brain.

            uint32_t repeatLMinus1 : 8;
            uint32_t aguLLoopMask : 6;  // mask of loops that are disabled in AGU L. 
            uint32_t loadL : 1;         // enable AGU-L.
            uint32_t teEnL : 1;         // enable TE-L brain.
        };
        uint32_t dw;
        uint16_t w[2];
    } MmeSBRepeat;

    typedef union _MmePCU
    {
        struct
        {
            uint32_t rlSaturation : 24;
            uint32_t reserved0 : 8;
        };
        uint32_t dw;
    } MmePCU;

    typedef union _MmeSw
    {
        struct
        {
            uint32_t swMemsetFwd : 1;    // Memset fwd operation - Ignored by the hw
            uint32_t swMemsetDedx : 1;   // Memset dedx operation - Ignored by the hw
            uint32_t swMemsetDedw : 1;   // Memset dedw operation - Ignored by the hw
            uint32_t reserved0 : 29;
        };
        uint32_t dw;
    } MmeSw;
    inline namespace gaudi
    {
        typedef union _Desc
        {
            struct
            {
                uint32_t baseAddrHighS;                            // The higher part of of the base address of the shared operand.
                uint32_t baseAddrHighL;                            // The higher part of of the base address of the local operand.
                uint32_t baseAddrHighO;                            // The higher part of of the base address of the output operand.
                uint32_t baseAddrLowS;                             // The lower part of of the base address of the shared operand.
                uint32_t baseAddrLowL;                             // The lower part of of the base address of the local operand.
                uint32_t baseAddrLowO;                             // The lower part of of the base address of the output operand.
                MmeHeader header;                                  // The operation header.
                MmeConvDesc conv;                                  // The convolution descriptor.
                uint32_t numIterationsMinus1;                      // The number of consecutive activations (number of tetrises).
                MmeOuterLoop outerLoop;                            // Number of tetrises loops.
                MmeTensorDesc tensorS;                             // The tensor of the shared operand.
                MmeAguCoreDesc aguS;                               // The AGU info for the shared operand.
                MmeTensorDesc tensorL;                             // The tensor of the local operand.
                MmeAguCoreDesc aguL[e_mme_local_and_remote];       // The AGU info for the local input operand - (for local and remote AGU).
                MmeTensorDesc tensorO;                             // The tensor of the local operand.
                MmeAguCoreDesc aguO[e_mme_local_and_remote];       // The AGU info for the output operand - (for local and remote).
                MmeSBRepeat sbRepeat;                              // SB rewind info.
                MmeRateLimeter rateLimiter;                        // RL info.
                MmeSyncObject syncObject;                          // The sync object value and address.
                MmeUserData axiUserData;                           // AXI user data.
                MmePerfEvt perfEvtS;                               // Performance event info for SB-S.
                MmePerfEvt perfEvtL[e_mme_local_and_remote];       // Performance event info for SB-L.
                MmePerfEvt perfEvtO[e_mme_local_and_remote];       // Performance event info for SB-O.
                uint32_t paddingValueS;                            // Padding value for the local tensor.
                uint32_t paddingValueL;                            // Padding value for sheared tensor.
                MmeMetaData metaData;                              // Metadata.
                MmePCU pcu;                                        // PCU RL info.
                MmeSw sw;                                         // MD for the SW. Ignored by the HW.
            };
            uint32_t dw[0x88];
        } Desc;
    }

    typedef union _RegBlock
    {
        struct
        {
            union
            {
                struct
                {
                    uint32_t status;            // The status register.
                    uint32_t pad0;
                    Desc desc;                // The SW copy of the descriptor.
                };
                uint8_t pad1[0x280];
            };
            uint32_t cmd;                        // Write to bit 0 in this register starts the execution.
            uint32_t status1;
            uint32_t reset;
            uint32_t stall;
            uint32_t dummy0;
            uint32_t dummy1;
        };
        uint32_t dw[1];
        uint8_t b[1];
    } RegBlock;
}

#pragma pack(pop)
