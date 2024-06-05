#pragma once
#include <stdint.h>

#pragma pack(push, 4)
namespace Gaudi2
{
    namespace Mme
    {
        enum EMmeCore
        {
            MME_CORE_SW = 0,
            MME_CORE_SE = 1,
            MME_CORE_NW = 2,
            MME_CORE_NE = 3,

            MME_CORES_NR = 4,
            MME_CORE_MASTERS_NR = 2,

            MME_CORE_MASTER0 = MME_CORE_SW,
            MME_CORE_MASTER1 = MME_CORE_NW,
            MME_CORE_SLAVE0 = MME_CORE_SE,
            MME_CORE_SLAVE1 = MME_CORE_NE,

            MME_CORE_MASTER = 0,
            MME_CORE_SLAVE = 1,
            MME_CORE_PAIR_SIZE = 2,
        };

        static const unsigned c_cl_size = 128;
        static const unsigned c_mme_max_conv_dims = 4;
        static const unsigned c_mme_max_tensor_dims = 5;
        static const unsigned c_mme_accums_nr = 4;
        static const unsigned c_mme_2x_accums_nr = 2*c_mme_accums_nr;
        static const unsigned c_mme_lfsr_seeds_nr = 256;

        static const unsigned c_mme_dcore_matrix_width_in_bytes = c_cl_size * 4;
        static const unsigned c_mme_dcore_matrix_height_in_bytes = c_cl_size * 2;
        static const unsigned c_mme_sb_size = 2768;
        static const unsigned c_mme_max_sb_reuse = 240;
        static const unsigned c_mme_sb_nr = 5;
        static const unsigned c_mme_wb_nr = 2;

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
            e_mme_dt_fp16 = 0x08,
            e_mme_dt_bf16 = 0x09,
            e_mme_dt_fp32 = 0x0a,
            e_mme_dt_tf32 = 0x0b,
            e_mme_dt_fp8_143 = 0x0c,
            e_mme_dt_fp8_152 = 0x0d,
            e_mme_dt_fp32ieee = 0x0e,
        } EMmeDataType;


        typedef enum
        {
            e_mme_rm_rn   = 0x0, // round half to nearest even
            e_mme_rm_rz   = 0x1, // round to zero
            e_mme_rm_ru   = 0x2, // round up
            e_mme_rm_rd   = 0x3, // round down
            e_mme_rm_rs   = 0x4, // stochastic rounding
            e_mme_rm_rhaz = 0x6, // round half away zero rounding
            e_mme_rm_rsn  = 0x7, // stochastic rounding + rne
        } EMmeRoundingMode;

        typedef enum
        {
            e_mme_pwr_loop_start = (1 << 0),    // fire when first trans leaves AGU-B
            e_mme_pwr_loop_end = (1 << 1),      // fire when last fence request returns
        } EMmePwrLoopCtrl;

        typedef enum
        {
            e_mme_shuffle_none = 0,           // no shuffle
            e_mme_shuffle_2ports = 1,           // shuffle 2 ports (odd/even de-interleaving)
            e_mme_shuffle_4ports = 3,           // shuffle 4 ports (mod4 de-interleaving)
        } EMmeShuffleAMode;

        typedef union _MmeHeader
        {
            struct
            {
                // byte 0
                uint32_t transA : 1;                // transpose A.
                uint32_t transB : 1;                // transpose B.
                uint32_t advanceA : 1;              // Advance operand A in the outer conv loop.
                uint32_t advanceB : 1;              // Advance operand B in the outer conv loop.
                uint32_t advanceC : 1;              // Advance operand C in the outer conv loop.
                uint32_t lowerA : 1;                // lower operand A.
                uint32_t lowerB : 1;                // lower operand B.
                uint32_t accumEn : 1;               // Accumulate output in the accumulator.

                // byte 1
                uint32_t rollAccums : 3;            // The number of accumulator inc to do after the last rollup.
                uint32_t aguReadsA : 5;             // AGU reads operand A - bit per AGU.

                // byte 2
                uint32_t aguReadsB : 5;             // AGU reads operand B - bit per AGU.
                uint32_t doubleAccums : 1;          // 0 - 8x64x256 Accums; 1 - 4x128x256 Accums
                uint32_t storeEn0 : 1;              // Store the output to Cout 0 (primary address).
                uint32_t storeEn1 : 1;              // Store the output to Cout 1 (secondary address).

                // byte 3
                uint32_t dataTypeIn  : 4;          // The data type of the input operand. (EMmeDataType)
                uint32_t dataTypeOut : 4;          // The data type of the C-output operand. (EMmeDataType)

                // byte 4
                uint32_t swapBaseAndOffsetA : 1;    // Swap RoiBase[4:1] with StartOffset[3:0] for TensorA
                uint32_t swapBaseAndOffsetB : 1;    // Swap RoiBase[4:1] with StartOffset[3:0] for TensorB
                uint32_t swapBaseAndOffsetOut : 1;  // Swap RoiBase[4:1] with StartOffset[3:0] for TensorCOut
                uint32_t reserved4 : 5;

                // byte 5
                uint32_t storeColorSet0 : 1;        // Cout-0 (primary address) color set
                uint32_t storeColorSet1 : 1;        // Cout-1 (secondary address) color set
                uint32_t hx2 : 1;                   // Enable 2 X H mode.
                uint32_t reserved5 : 5;

                // byte 6
                uint32_t partialHeightLoopA : 6;    // enable bit per loop.
                                                    // in transpose the loops that use partial height in their last iteration.
                                                    // in non-transpose - the loops that use partial dense FCD.
                uint32_t reserved6 : 2;

                // byte 7
                uint32_t partialHeightLoopB : 6;    // enable bit per loop.
                                                    // in transpose the loops that use partial height in their last iteration.
                                                    // in non-transpose - the loops that use partial dense FCD.
                uint32_t teBypassA : 1;             // bypass TE (power feature)
                uint32_t teBypassB : 1;             // bypass TE (power feature)
            };

            uint8_t bytes[8];
            uint32_t dw[2];
            uint64_t ddw;
        } MmeHeader;

        typedef struct _MmeEnableAndMask
        {
            uint8_t loopMask : 6;  // bit mask of the MME loops that should be masked. (EMmeLoopMask)
            uint8_t masterEn : 1;  // enable bit for the master MME. (should also be used in 'two_masters_mode')
            uint8_t slaveEn : 1;  // enable bit for the slave MME.
        } MmeEnableAndMask;

        typedef union _MmeBrainsCtrl
        {
            struct
            {
                MmeEnableAndMask aguA;
                MmeEnableAndMask aguB;
                MmeEnableAndMask aguOut0;
                MmeEnableAndMask aguOut1;
                MmeEnableAndMask eu;
                MmeEnableAndMask ap;
                struct
                {
                    uint16_t decEn : 1;                 // Operand B - enable the decompressor
                    uint16_t shuffleA : 2;              // EMmeShuffleAMode
                    uint16_t bgemm : 1;                 // Batch-gemm mode
                    uint16_t clipFpEu : 1;              // Clip inf to max value when down-converting (FP32->TF32)
                    uint16_t clipFpAp : 1;              // Clip inf to max value when down-converting
                    uint16_t sbACacheEn : 1;            // Enable SB A cache.
                    uint16_t sbBCacheEn : 1;            // Enable SB A cache.
                    uint16_t roundingMode : 3;          // Rounding mode
                    uint16_t reluEn : 1;                // Enable RELU.
                    uint16_t noRollup : 1;              // mask rollup indication. (EU brain)
                    uint16_t nullDesc : 1;              // MME only signal w/o any operation (this bit hsould be set in CMD)
                    uint16_t reserved : 2;
                };
            };
            uint32_t dw[2];
            uint8_t byte[8];
        } MmeBrainsCtrl;

        typedef struct _MmeTensorDesc
        {                                                                     // Index '0' is the fastest changing dim.
            uint32_t validElements[c_mme_max_tensor_dims];                   // The number of valid elements.
            int32_t loopStride[c_mme_max_tensor_dims];                      // The offset in which the ROI base should be shifted between loop iterations.
            int32_t roiSize[c_mme_max_tensor_dims - 1];                     // The size of the ROI.
            uint32_t spatialStrides[c_mme_max_tensor_dims - 1];              // The strides of the spatial dimensions.
            int32_t startOffset[c_mme_max_tensor_dims - 1];                  // The dimension's start offset.
        } MmeTensorDesc;

        typedef struct _MmeAguCoreDesc
        {
            int32_t roiBaseOffset[c_mme_max_tensor_dims];    // The walk ROI base offset.
        } MmeAguCoreDesc;

        typedef union _MmeAssociatedDims
        {
            struct
            {
                uint16_t dimA : 3;
                uint16_t dimB : 3;
                uint16_t dimOut : 3;
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

        typedef union _MmeSyncObjectVal
        {
            struct
            {
                uint32_t soValue : 15;
                uint32_t soReserved : 15;
                uint32_t soPerfEn : 1;
                uint32_t soOp : 1;
            };
            uint32_t dw;
        } MmeSyncObjectVal;


        typedef union _MmeSyncObject
        {
            struct
            {
                // dw 0
                struct
                {
                    uint32_t signalMask0 : 6;                  // EMmeLoopMask value bit mask that specifies when to signal the
                    // sync object of color 0. signalMask must never be less than accumMask.
                    uint32_t signalEn0 : 1;                    // Enable signaling of color 0.
                    uint32_t reserved0 : 1;
                    uint32_t signalMask1 : 6;    // EMmeLoopMask value bit mask that specifies when to signal the
                                                 // sync object of color 1. signalMask must never be less than accumMask.
                    uint32_t signalEn1 : 1;                // Enable signaling of color 1.
                    uint32_t masterWaitForSlaveFence : 1;  // When set, the master must wait for 'fence done' from the slave.
                    uint32_t slaveSendFence2Master  : 1;   // When set, the slave sends its fence completion to the master.
                    uint32_t slaveSignalEn : 1;          // When set, the slave signals through the LBW port.
                    uint32_t slave0UseSlaveSOAddr : 1;    // When set, slave 0 signals to the address specified in the slave SOaddress field
                    uint32_t slave1UseSlaveSOAddr : 1;    // When set, slave 1 signals to the address specified in the slave SOaddress field
                    uint32_t slave0UseMasterSOAddrPlus4 : 1;  // When set, slave 0 signals to the SO that is after the master's .
                    uint32_t slave1UseMasterSOAddrPlus4 : 1;  // When set, slave 1 signals to the SO that is after the master's .
                    uint32_t reserved1 : 10;
                };

                // dw 1
                uint32_t so0Addr;                    // Lower 32 bits of the address of sync object 0.

                // dw 2
                MmeSyncObjectVal so0Val;             // value of SO 0

                // dw 3
                uint32_t so1Addr;                    // Lower 32 bits of the address of sync object 1 for the master (0) and slave (1).

                // dw 4
                MmeSyncObjectVal so1Val;             // value of SO 1
            };

            uint32_t dw[5];
        } MmeSyncObject;

        typedef union MmeUserData
        {
            struct
            {
                uint32_t first : 10;
                uint32_t steady : 10;
                uint32_t mask : 6;
                uint32_t reserved : 6;
            };
            uint32_t dw;
        } MmeUserData;

        typedef struct _MmeSpare
        {
            uint32_t sb0 : 16;
            uint32_t sb1 : 16;
            uint32_t sb2 : 16;
            uint32_t sb3 : 16;
            uint32_t sb4 : 16;
            uint32_t Out : 16;
            uint32_t eu : 16;
            uint32_t ap : 16;
        } MmeSpare;

        typedef union _MmeRateLimiter
        {
            struct
            {
                uint32_t aguA : 8;
                uint32_t aguB : 8;
                uint32_t aguOut : 8;
                uint32_t eu : 8;
            };
            uint32_t dw;
        } MmeRateLimiter;

        typedef union _MmePerfEvt
        {
            struct
            {
                uint32_t value : 16;                // trace payload
                uint32_t rst : 1;                   // use value as trace payload
                uint32_t incMask : 1;               // increment value each iteration
                uint32_t startEndMask : 2;          // start bit 0, end bit 1 (active high)
                uint32_t loopMask : 6;              // loop-mask for trigger (active high)
                uint32_t operand : 5;               // operand enable mask:
                                                    //  for input - aguIn index
                                                    //  for output - aguOut index (only 2-bits)
                uint32_t slaveSendsPerfEvent : 1;   // active low (H6-3187):
                                                    //  0: only slave issue events (regardless to startEndMask)
                                                    //  1: both master+slave issue events according to startEndMask

            };
            uint32_t dw;
        } MmePerfEvt;

        typedef union _MmeRouting
        {
            struct
            {
                uint32_t sb0En    : 1;
                uint32_t sb1En    : 1;
                uint32_t sb2En    : 1;
                uint32_t sb3En    : 1;
                uint32_t sb4En    : 1;
                uint32_t in0En    : 1;
                uint32_t in1En    : 1;
                uint32_t sb0Sel   : 3;
                uint32_t sb1Sel   : 3;
                uint32_t sb2Sel   : 3;
                uint32_t sb3Sel   : 3;
                uint32_t sb4Sel   : 3;
                uint32_t in0Sel   : 3;
                uint32_t in1Sel   : 3;
                uint32_t sb0OutEn : 1;
                uint32_t sb2OutEn : 1;
                uint32_t sb3OutEn : 1;
                uint32_t reserved : 1;
            };
            uint32_t dw;
            uint8_t b[4];
        } MmeRouting;

        typedef union _MmeCtrl
        {
            MmeRouting eus[MME_CORE_PAIR_SIZE];   // eu sync info per core.
            uint32_t dw[2];
            uint64_t ddw;
        } MmeCtrl;

        typedef union _MmeSBRepeat
        {
            struct
            {
                uint32_t repeatAMinus1 : 8;
                uint32_t repeatBMinus1 : 8;
                uint32_t repeatAMask : 6;
                uint32_t reserved2 : 2;
                uint32_t repeatBMask : 6;
                uint32_t reserved3 : 2;
            };
            uint32_t dw;
            uint8_t b[4];
        } MmeSBRepeat;

        typedef union _MmePCU
        {
            struct
            {
                uint32_t rlSaturation : 24;
                uint32_t reserved : 8;
            };
            uint32_t dw;
        } MmePCU;

        typedef union _MmePowerLoop
        {
            struct
            {
                uint32_t ctrlMstr : 2;              // 2-color fence for master
                uint32_t ctrlSlv  : 2;              // 2-color fence for slave
                uint32_t powerLoopMd : 8;           // Meta-data for Power loop (MME->ARC)
                uint32_t reserved : 20;
            };
            uint32_t dw;
        } MmePowerLoop;

        typedef union _MmeAddress
        {
            struct
            {
                uint32_t low;
                uint32_t high;
            };
            uint64_t addr;
            uint32_t dw[2];
        } MmeAddress;

        typedef union _MmeFP8Bias
        {
            struct
            {
                uint32_t a : 4;
                uint32_t b : 4;
                uint32_t out : 5;
                uint32_t reserved : 19;
            };
        } MmeFP8Bias;

        typedef struct _Desc
        {
            MmeAddress baseAddrCOut1;                         // address of the second COut operand
            MmeAddress baseAddrCOut0;                         // address of the first COut operand
            MmeAddress baseAddrA;                             // address of operand A
            MmeAddress baseAddrB;                             // address of operand B
            MmeBrainsCtrl brains;                              // Brain CTRL info.
            MmeHeader header;                                  // The operation header.
            MmeCtrl ctrl;                                      // Routing and EU sync info.
            MmeTensorDesc tensorA;                             // The tensor of operand A.
            MmeTensorDesc tensorB;                             // The tensor of operand B.
            MmeTensorDesc tensorCOut;                          // The tensor of operand COut.
            MmeSyncObject syncObject;                          // The sync object value and address.
            MmeAguCoreDesc aguIn[c_mme_sb_nr][MME_CORE_PAIR_SIZE];           // The Input AGUs info
            uint32_t spatialSizeMinus1A;                       // spatial size for A
            uint32_t spatialSizeMinus1B;                       // spatial size for B slave and master.
            MmeAguCoreDesc aguOut[c_mme_wb_nr][MME_CORE_PAIR_SIZE];        // The Output AGUs info
            uint32_t spatialSizeMinus1Cout;                    // spatial size for C out.
            MmeConvDesc conv;                                  // The convolution descriptor.
            MmeOuterLoop outerLoop;                            // Number of tetrises loops.
            uint32_t numIterationsMinus1;                      // The number of consecutive activations (number of tetrises).
            MmeSBRepeat sbRepeat;                              // SB rewind info.
            MmeFP8Bias fp8Bias;                                // bias values for fp8 1-5-2
            MmeRateLimiter rateLimiter;                        // RL info.
            MmeUserData axiUserData;                           // AXI user data.
            MmePerfEvt perfEvtIn;                              // Performance event for input operands.
            MmePerfEvt perfEvtOut;                             // Performance event for output operands.
            MmePCU pcu;                                        // PCU RL info.
            uint32_t slaveSyncObject0Addr;                     // Slave SO0 address
            uint32_t slaveSyncObject1Addr;                     // Slave SO1 address
            MmePowerLoop powerLoop;                            // Power Loop info
            MmeSpare spare[MME_CORE_PAIR_SIZE];                // Spare bits.
            uint32_t wkldID;                                   // workload ID
        } Desc;

        typedef union _MmeCmd
        {
            struct
            {
                // status bits: '1' sets the status bit. '0' is ignored.
                uint32_t aguIn : 5;
                uint32_t eu : 1;
                uint32_t ap : 1;
                uint32_t aguOut : 2;

                // Increment the current desc idx, block until it's ready, copy its content.
                // When the write returns: currIdx = (prevIdx + 1) % 4.
                // When 'copyAndInc' is set, the status bits will be set only after the
                // desctiptor that is pointed by the next index is freed.
                uint32_t copyAndInc : 1;

                // Selects the desc to set the status bits for.
                // the status bits will be set in: idx = (currIdx + copyAndInc + sel) % 4.
                uint32_t descSel : 2;

                // Mask the idle indication. When set the idle indication to the QMAN is masked.
                uint32_t maskIdleIndication : 1;

                // Copy aguOut1 from agu0
                uint32_t aguOut1FromAguOut0_DW0 : 1;
                uint32_t aguOut1FromAguOut0_DW1_4 : 1;

                // Null-Descriptor:
                // when this bit is set MME will only send syncObject to signal
                uint32_t nullDesc : 1;

                uint32_t reserved : 16;
            };
            uint32_t dw;
        } MmeCmd;

        typedef union _MmeDescWrapper
        {
            struct
            {
                uint32_t status;   // The status register.
                MmeCmd cmd;        // Write to this register starts the execution.
                Desc desc;         // The SW copy of the descriptor.
            };
            uint32_t pad[0x400];
        } MmeDescWrapper;


        typedef union _RegBlock
        {
            struct
            {
                MmeDescWrapper descWrap;
                uint32_t stall;
            };
            uint32_t dw[1];
            uint8_t b[1];
        } RegBlock;
    }
}
#pragma pack(pop)



