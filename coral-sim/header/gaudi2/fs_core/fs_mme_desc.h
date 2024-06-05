#pragma once
#include <stdint.h>

namespace Gaudi2
{
namespace Mme
{
struct ACCAPDescriptor
{
    // Rollup
    uint32_t     accIdx; // Accumulator index
    uint32_t     widthMinus1; // Effective symmetric matrix width in elements
    uint32_t     heightMinus1; // Effective symmetric matrix height in elements.
    uint32_t     apRoundingMode; // AP (down-conversion) rounding-mode
    uint32_t     high2xEn; // 2xHigh mode
    EMmeDataType coutDataType; // Cout data-type
    uint32_t     signalOnly; // Signal only mode - WBC should be dummy-activated in order to produce SyncMessage
    uint32_t     doubleAccums; // Double accums: '0' - 4 acc, '1' - 8 acc
    uint32_t     spare; // spare bits
    unsigned     descDbgCnt; // debug info

    // AP modes
    uint32_t ap1En; // AP1 is active
    uint32_t rmwEn; // Enable read-modify-write in accMem
    uint32_t reluEn; // Enable Relu
    uint32_t storeEn; // Write results to WBC (otherwise AP not active)
    uint32_t clipFP; // clip inf->max value
    uint32_t fp8Bias; // fp8_143 bias

    // Read accMem descriptor
    uint32_t apChunkSize; // num of elements to read from ACC ( 0 = 32, 1 = 64, 2 = 128 )
    uint32_t ap0Dim0Size; // AP0 Last chunk size (up to 128 elements)
    uint32_t ap1Dim0Size; // AP1 Last chunk size (up to 128 elements)
    uint32_t ap0Dim1Size; // AP0 number of chunks (in last 512B line)
    uint32_t ap1Dim1Size; // AP1 number of chunks (in last 512B line)
    uint32_t apDim1Stride; // stride to next chunk ( 0 = 128, 1 = 256, 2 = 512 )
};

struct EUSDescriptor
{
    bool             storeEn; // Store enable
    MmeRouting       routing; // Mux Routing
    bool             high2xEn; // 2xHigh mode
    uint32_t         accIdx; // Accumulator index
    uint32_t         widthMinus1; // effective width in elements
    uint32_t         heightMinus1; // effective height in elements.
    EMmeDataType     dataType; // EMmeDataType
    bool             rollupEn; // enable rollup
    uint32_t         vecNumMinus1; // gemm common dimension (vec_num)
    uint32_t         pcuRlSaturation; // PCU RL stauration
    uint32_t         rateLimiterRstToken; // PCU RL stauration
    bool             transA; // A transposed
    bool             transB; // B transposed
    bool             bgemm; // Batch Gemm
    EMmeShuffleAMode shuffleA; // de-interleave A
    bool             doubleAccums; // 4/8 accums
    uint8_t          fp8BiasOpA; // opA fp8 bias
    uint8_t          fp8BiasOpB; // opB fp8 bias
    bool             clipFP; // clip float to max
    uint32_t         spare;
    uint32_t         wkldID; // PCU RL stauration
    unsigned         descDbgCnt; // debug info
};
} // namespace Mme
} // namespace Gaudi2
