#pragma once
#include <stdint.h>

#pragma pack(push, 4)
namespace gaudi3
{
    namespace Mme
    {

        enum EMmeCore
        {
		    DCORE0_MME = 0,
		    DCORE1_MME = 1,
		    DCORE2_MME = 2,
		    DCORE3_MME = 3,

		    MME_CORES_NR = 16,
		    MME_CORE_MASTERS_NR = 8,

		    HDCORE_MME0 = 0,
		    HDCORE_MME1 = 1,
		    HDCORE_SIZE = 2,

		    MME_MASTER = 0,
		    MME_SLAVE = 1,
		    MME_PAIR_SIZE = 2,
        };

        static const unsigned c_cl_size = 128;
        static const unsigned c_mme_max_conv_dims = 4;
        static const unsigned c_mme_max_tensor_dims = 5;
        static const unsigned c_mme_accums_nr = 4;
        static const unsigned c_mme_2x_accums_nr = 2*c_mme_accums_nr;
        static const unsigned c_mme_lfsr_seeds_nr = 128;

        static const unsigned c_mme_dcore_matrix_width_in_bytes = c_cl_size * 4;
        static const unsigned c_mme_dcore_matrix_height_in_bytes = c_cl_size * 2;
        static const unsigned c_mme_sb_size = 3200;
        static const unsigned c_mme_max_sb_reuse = 240;
        static const unsigned c_mme_sb_nr = 4;
        static const unsigned c_mme_wb_nr = 1;

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
            e_mme_dt_fp8 = 0x0c,
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
        } EMmeShuffleAMode;


        typedef enum
        {
            e_mme_te_in_x = (0 << 2),
            e_mme_te_out_x = (1 << 2),

            e_mme_te_x1 = 0,
            e_mme_te_x2 = 1,
            e_mme_te_x4 = 2,
            e_mme_te_x8 = 3,

            e_mme_te_in_x2 = (e_mme_te_in_x | e_mme_te_x2),
            e_mme_te_in_x4 = (e_mme_te_in_x | e_mme_te_x4),
            e_mme_te_in_x8 = (e_mme_te_in_x | e_mme_te_x8),
            e_mme_te_out_x2 = (e_mme_te_out_x | e_mme_te_x2),
            e_mme_te_out_x4 = (e_mme_te_out_x | e_mme_te_x4),
            e_mme_te_out_x8 = (e_mme_te_out_x | e_mme_te_x8),

        } EMmeTeAcceleration;

        typedef union _MmeHeader
        {
            struct
            {
                // byte 0
                uint32_t transA : 1;                // trans operand A
                uint32_t transB : 1;                // trans operand B
                uint32_t sbTransA : 1;              // SB trans mode operand A
                uint32_t sbTransB : 1;              // SB trans mode operand B
                uint32_t advanceA : 1;              // Advance operand A in the outer conv loop.
                uint32_t advanceB : 1;              // Advance operand B in the outer conv loop.
                uint32_t advanceC : 1;              // Advance operand C in the outer conv loop.
                uint32_t accumEn : 1;               // Accumulate output in the accumulator.

                // byte 1
                uint32_t lowerA : 1;                // lower operand A.
                uint32_t lowerB : 1;                // lower operand B.
                uint32_t rollAccums : 3;            // The number of accumulator inc to do after the last rollup.
                uint32_t storeEn0 : 1;              // Store the output to Cout 0 (primary address).
                uint32_t storeEn1 : 1;              // Store the output to Cout 1 (secondary address).
                uint32_t reluEn : 1;                // Enable RELU.

                // byte 2
                uint32_t doubleAccums : 1;          // 0 - 8x64x256 Accums; 1 - 4x128x256 Accums
                uint32_t bgemm : 1;		            // 0 - Symmetric (128x256); 1 - Bgemm (2x128x128)
                uint32_t shuffleA : 1;              // EMmeShuffleAMode
                uint32_t roundingMode : 3;          // Rounding mode
	            uint32_t noRollup : 1;              // mask rollup indication. (EU brain)
	            uint32_t nullDesc : 1;              // MME only signal w/o any operation (this bit hsould be set in CMD)

                // byte 3
                uint32_t dataTypeIn  : 4;           // The data type of the input operand. (EMmeDataType)
                uint32_t dataTypeOut : 4;           // The data type of the C-output operand. (EMmeDataType)

                // byte 4
                uint32_t swapBaseAndOffsetA : 1;    // Swap RoiBase[4:1] with StartOffset[3:0] for TensorA
                uint32_t swapBaseAndOffsetB : 1;    // Swap RoiBase[4:1] with StartOffset[3:0] for TensorB
                uint32_t swapBaseAndOffsetOut : 1;  // Swap RoiBase[4:1] with StartOffset[3:0] for TensorCOut
                uint32_t opANonShared : 1;          // when set operand A is not shared between EUs
                uint32_t clipFpEu : 1;              // Clip inf to max value when down-converting (FP32->TF32)
                uint32_t clipFpAp : 1;              // Clip inf to max value when down-converting
                uint32_t sbACacheEn : 1;            // Enable SB A cache.
                uint32_t sbBCacheEn : 1;            // Enable SB A cache.

                // byte 5
                uint32_t partialHeightLoopA : 6;    // enable bit per loop.
                                                    // in transpose the loops that use partial height in their last iteration.
                                                    // in non-transpose - the loops that use partial dense FCD.
                uint32_t storeColorSet0 : 1;        // Cout-0 (primary address) color set
                uint32_t storeColorSet1 : 1;        // Cout-1 (secondary address) color set

                // byte 6
                uint32_t partialHeightLoopB : 6;    // enable bit per loop.
                                                    // in transpose the loops that use partial height in their last iteration.
                                                    // in non-transpose - the loops that use partial dense FCD.
                uint32_t teBypassA : 1;             // bypass TE (power feature)
                uint32_t teBypassB : 1;             // bypass TE (power feature)

                // byte 7
                uint32_t teAccelA : 3;              // EMmeTeAcceleration
                uint32_t sftzFp32ToFp8 : 1;         // Stochastic FTZ for FP32->FP8 conversions
                uint32_t wbCacheEn : 1;		        // enable aggregation in wr-port
                uint32_t dualGemm : 1;              // dualGemm mode enable (must be set together with bgemm)
                uint32_t dmaMode : 1;               // DMA-mode
                uint32_t ftz : 1;                   // Flush denormals to zero in AP

            };

            uint8_t bytes[8];
            uint32_t dw[2];
            uint64_t ddw;
        } MmeHeader;

        typedef struct _MmeEnableAndMask
        {
            uint8_t loopMask : 6;   // bit mask of the MME loops that should be masked. (EMmeLoopMask)
            uint8_t masterEn : 1;   // enable bit for the master MME. (should also be used in 'two_masters_mode')
            uint8_t slaveEn : 1;    // enable bit for the slave MME.
        } MmeEnableAndMask;

        typedef union _MmeBrainsCtrl
        {
            struct
            {
                MmeEnableAndMask aguA;
                MmeEnableAndMask aguB;
                MmeEnableAndMask aguOut;
                MmeEnableAndMask eu;
                MmeEnableAndMask ap;
                MmeEnableAndMask aguOutDma;
                uint8_t reserved[2];
            };
            uint32_t dw[2];
            uint8_t byte[8];
        } MmeBrainsCtrl;

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

        typedef struct _MmeDualGemmTensorDesc //dualGemm tensorDesc structure
        {
            uint32_t validElements[MME_PAIR_SIZE][2];       // The number of valid elements.
            uint32_t spatialSizeMinus1Gemm1;                // spatial size for Gemm1
            int32_t roiSize[MME_PAIR_SIZE][2];              // The size of the ROI.
            uint32_t spatialStrides[MME_PAIR_SIZE];         // The strides of the spatial dimensions.
            MmeAddress baseAddrGemm1;                       // Gemm1 base base Address
            int32_t startOffset[MME_PAIR_SIZE][2];          // The dimension's start offset.
            MmeAddress baseAddrGemm1Dup;                    // Only for Cout1 (duplicate output)
            uint32_t reserved[3];
        } MmeDualGemmTensorDesc;

        typedef union _MmeTensorDesc
        {
            struct
            {                                                                   // Index '0' is the fastest changing dim
                uint32_t validElements[c_mme_max_tensor_dims];                  // The number of valid elements.
                int32_t roiSize[c_mme_max_tensor_dims - 1];                     // The size of the ROI.
                uint32_t spatialStrides[c_mme_max_tensor_dims - 1];             // The strides of the spatial dimensions.
                int32_t startOffset[c_mme_max_tensor_dims - 1];                 // The dimension's start offset.
                int32_t loopStride[c_mme_max_tensor_dims];                      // The offset in which the ROI base should be shifted between loop iterations.
            };
            MmeDualGemmTensorDesc dualGemm;
            uint32_t dw[22];

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
                    uint32_t signalMask0 : 6;                   // EMmeLoopMask value bit mask that specifies when to signal the
                                                                // sync object of color 0. signalMask must never be less than accumMask.
                    uint32_t signalEn0 : 1;                     // Enable signaling of color 0.
                    uint32_t reserved0 : 1;
                    uint32_t signalMask1 : 6;                   // EMmeLoopMask value bit mask that specifies when to signal the
                                                                // sync object of color 1. signalMask must never be less than accumMask.
                    uint32_t signalEn1 : 1;                     // Enable signaling of color 1.
                    uint32_t masterWaitForSlaveFence : 1;       // When set, the master must wait for 'fence done' from the slave.
                    uint32_t slaveSendFence2Master  : 1;        // When set, the slave sends its fence completion to the master.
                    uint32_t slaveSignalEn : 1;                 // When set, the slave signals through the LBW port.
                    uint32_t slave0UseSlaveSOAddr : 1;          // When set, slave 0 signals to the address specified in the slave SOaddress field
                    uint32_t slave1UseSlaveSOAddr : 1;          // When set, slave 1 signals to the address specified in the slave SOaddress field
                    uint32_t slave0UseMasterSOAddrPlus4 : 1;    // When set, slave 0 signals to the SO that is after the master's .
                    uint32_t slave1UseMasterSOAddrPlus4 : 1;    // When set, slave 1 signals to the SO that is after the master's .
                    uint32_t reserved1 : 10;
                };

                // dw 1
                uint32_t so0Addr;                    // Lower 32 bits of the address of sync object 0.

                // dw 2
                MmeSyncObjectVal so0Val;             // value of SO 0

                // dw 3
                uint32_t so1Addr;                    // Lower 32 bits of the address of sync object 1.

                // dw 4
                MmeSyncObjectVal so1Val;             // value of SO 1

		        // dw 5
                uint32_t slaveSo0Addr;               // Lower 32 bits of the address of sync object 0 for the slave.

		        // dw 6
                uint32_t slaveSo1Addr;               // Lower 32 bits of the address of sync object 1 for the slave.
            };

            uint32_t dw[7];
        } MmeSyncObject;

        typedef union _MmeCacheData	// AxCache bits (see H9 NoC spec)
        {
            struct
            {
                uint32_t aguA : 4;          // ARCACHE
                uint32_t aguB : 4;          // ARCACHE
                uint32_t aguOut : 4;        // AWCACHE
                uint32_t reserved : 20;
            };
            uint32_t dw;
        } MmeCacheData;

        typedef union _MmeUserData	// AxUser bits (see H9 NoC spec)
        {
            struct
            {
                uint32_t qosFirst : 4;
                uint32_t qosSteady : 4;
                uint32_t qosMask : 6;
                uint32_t mcid : 16;
                uint32_t clss : 2;
            };
            uint32_t dw;
        } MmeUserData;

        typedef union _MmeAwUserData	// AwUser Reduction bits (see H9 NoC spec)
        {
            struct
            {
                uint32_t first : 11;
                uint32_t steady : 11;
                uint32_t mask : 6;
                uint32_t reserved : 4;
            };
            uint32_t dw;
        } MmeAwUserData;

        typedef struct _MmeSpare
        {
            uint32_t agu0 : 16;
            uint32_t agu1 : 16;
            uint32_t agu2 : 16;
            uint32_t agu3 : 16;
            uint32_t aguOut : 16;
            uint32_t eu : 16;
            uint32_t ap : 16;
            uint32_t reserved : 16;
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
                uint32_t incEn : 1;                 // increment value each iteration
                uint32_t startEndEn : 2;            // start bit 0, end bit 1 (active high)
                uint32_t loopMask : 6;              // loop-mask for trigger (active high)
                uint32_t operand : 4;               // operand enable mask:
                                                    //  for input - aguIn index
                                                    //  for EU/Output - NA
                uint32_t slaveSendsPerfEvent : 1;   // enable slave setting event
                uint32_t reserved : 1;

            };
            uint32_t dw;
        } MmePerfEvt;

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

        typedef union _MmePower
        {
            struct
            {
                uint32_t loopCtrl : 2;                  // bit0 - enable start, bit1 - enable end
                uint32_t loopMd : 8;           // Meta-data for Power loop (MME->ARC)
                uint32_t pmuRlSaturation : 20;	    // PMU rate-limiter staturation
                uint32_t sbOppDisA : 1;		// Disable SB opp mode operand A
                uint32_t sbOppDisB : 1;		// Disable SB opp mode operand B
            };
            uint32_t dw;
        } MmePower;

        typedef union _MmeNumericFlavor
        {
            struct
            {
                uint32_t biasA : 6;                     // operandA FP8/FP16_SHP bias
                uint32_t biasB : 6;                     // operandB FP8/FP16_SHP bias
                uint32_t biasOut : 6;                   // operandOut FP8/FP16_SHP bias
                uint32_t accRoundingMode : 2;           // ACC Rounding mode only support RNE/RZ
                uint32_t fp8FlavorA : 1;                // operandA FP8 flavor: 0 - fp8_143; 1 - fp8_152
                uint32_t fp8FlavorB : 1;                // operandB FP8 flavor: 0 - fp8_143; 1 - fp8_152
                uint32_t fp8FlavorOut : 1;              // operandOut FP8 flavor: 0 - fp8_143; 1 - fp8_152
                uint32_t fp16FlavorA : 1;               // operandA FP16 flavor: 0 - fp16_shp; 1 - fp16_uhp
                uint32_t fp16FlavorB : 1;               // operandB FP16 flavor: 0 - fp16_shp; 1 - fp16_uhp
                uint32_t fp16FlavorOut : 1;             // operandOut FP16 flavor: 0 - fp16_shp; 1 - fp16_uhp
                uint32_t infNanModeA : 2;               // operandA/B/Out  inf/nan representation for FP8/FP16_SHP:
                uint32_t infNanModeB : 2;               //  0 - legacy      (fp8/fp16)
                uint32_t infNanModeOut : 2;             //  1 - no inf/nan  (fp8/fp16)
                                                        //  2 - min inf/nan (fp8 only)
                                                        //  3 - reserved
            };
	    uint32_t dw;
        } MmeNumericFlavors;

        typedef struct _Desc
        {
              MmeAddress baseAddrCOut1;                          // address of the secondary COut operand
              MmeAddress baseAddrCOut0;                          // address of the primary COut operand
              MmeAddress baseAddrA;                              // address of operand A
              MmeAddress baseAddrB;                              // address of operand B
              MmeBrainsCtrl brains;                              // Brain CTRL info.
              MmeHeader header;                                  // The operation header.
              MmeTensorDesc tensorA;                             // The tensor of operand A.
              MmeTensorDesc tensorB;                             // The tensor of operand B.
              MmeTensorDesc tensorCOut;                          // The tensor of operand COut.
              MmeAguCoreDesc aguIn[MME_PAIR_SIZE][c_mme_sb_nr];  // The Input AGUs info
              uint32_t spatialSizeMinus1A;                       // spatial size for A
              uint32_t spatialSizeMinus1B;                       // spatial size for B slave and master.
              MmeAguCoreDesc aguOut[MME_PAIR_SIZE];              // The Output AGUs info
              uint32_t spatialSizeMinus1Cout;                    // spatial size for C out.
              MmeConvDesc conv;                                  // The convolution descriptor.
              MmeOuterLoop outerLoop;                            // Number of tetrises loops.
              uint32_t numIterationsMinus1;                      // The number of consecutive activations (number of tetrises).
              MmeSBRepeat sbRepeat;                              // SB rewind info.
              MmeSyncObject syncObject;                          // The sync object value and address.
              MmeNumericFlavors numerics;                        // Numeric flavors for fp8/fp16
              MmeAwUserData axiAwUserData;                       // AXI write port only user data.
              MmeUserData axiUserDataA;                          // AXI user data operand A.
              MmeUserData axiUserDataB;                          // AXI user data operand B.
              MmeUserData axiUserDataCout;                       // AXI user data operand Cout.
              MmeCacheData axiCacheData;                         // AXI cache data.
              MmePerfEvt perfEvtIn;                              // Performance event for input operands.
              MmePerfEvt perfEvtOut;                             // Performance event for output operands.
              MmePerfEvt perfEvtEU;                              // Performance event for EU.
              MmeRateLimiter rateLimiter;                        // RL info.
              MmePower powerLoop;                                // Power Loop and PMU info
              MmeSpare spare[MME_PAIR_SIZE];                     // Spare bits.
              uint32_t wkldID;                                   // workload ID
        } Desc;

        typedef union _MmeCmd
        {
            struct
            {
                // status bits: '1' sets the status bit. '0' is ignored.
                uint32_t aguIn : 4;
                uint32_t eu : 1;
                uint32_t ap : 1;
                uint32_t aguOut : 1;
                uint32_t aguOutDma : 1;

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

                // Compute Null-Descriptor:
                // when this bit is set MME will only send syncObject to signal (on compute channel)
                uint32_t nullDesc : 1;

                // Commit of DMA-descriptor - Copy DMA-Arch desc to shadow
                uint32_t dmaDesc : 1;

                // Dma Null-Descriptor:
                // when this bit is set MME will only send syncObject to signal (on dma channel)
                uint32_t nullDmaDesc : 1;

                uint32_t reserved : 17;
            };
            uint32_t dw;
        } MmeCmd;

        typedef union _MmeDescWrapper
        {
            struct
            {
                uint32_t status[MME_PAIR_SIZE];   // The status register.
                Desc desc;         // The SW copy of the descriptor.
                uint32_t reserved1[26];		// reserved
                MmeCmd cmd;        // Write to this register starts the execution.
                uint32_t reserved2[63];		// reserved
                uint32_t stall;
                uint32_t reserved3[257];		// reserved
                Desc dmaDesc;         // The SW copy of the descriptor.
		        uint32_t reserved4[314];		// reserved

            };
            uint32_t pad[0xf80];
        } MmeDescWrapper;


        typedef union _RegBlock
        {
            struct
            {
                MmeDescWrapper descWrap;
            };
            uint32_t dw[0xf80];
        } RegBlock;
    }
}
#pragma pack(pop)



