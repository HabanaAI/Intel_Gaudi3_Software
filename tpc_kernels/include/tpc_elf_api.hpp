/*****************************************************************************
 * Copyright (C) 2019 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 * Authors:
 * Tzachi Cohen <tcohen@habana.ai>
 ******************************************************************************
 */
#ifndef TPC_ELF_API_HPP
#define TPC_ELF_API_HPP

#include <cstdint>
#include <gc_interface.h>
#include <tpc_kernel_lib_interface.h>
#include <memory>

extern "C"{
namespace TpcElfTools
{

enum TpcElfStatus
{
    TPC_ELF_SUCCESS                            = 0,
    TPC_ELF_INVALID_ELF_BUFFER                 = 1,
    TPC_ELF_SECTION_NOT_FOUND                  = 2,
    TPC_ELF_UNSUPPORTED                        = 3,
    TPC_ELF_INDEXSPACE_ERROR                   = 4,
    TPC_ELF_INVALID_BUFFER                     = 5,
    TPC_ELF_COMPILER_UNAVAILABLE               = 6,
    TPC_ELF_COMPILATION_FAILURE                = 7,
    TPC_ELF_UNEXPECTED_OPTIONS                 = 8,
    TPC_ELF_CANNOT_WRITE_FILE                  = 9,
    TPC_ELF_KERNEL_USING_MMIO                  = 10,
    TPC_ELF_KERNEL_WITHOUT_LL                  = 11,
    TPC_ELF_KERNEL_TARGET_SPECIFIC             = 12,
    TPC_ELF_KERNEL_USE_REG_ID                  = 13,
    TPC_ELF_JSON_NOT_READABLE                  = 14,
    TPC_ELF_EVALUATOR_ERROR                    = 15,
    TPC_ELF_KERNEL_USING_RMW                   = 16,
    TPC_ELF_RECOMPILATION_FAILURE              = 17,
    TPC_ELF_BB_OVERFLOW                        = 18,
    TPC_ELF_DS_BAILOUT_INVALID_ARCH            = 19,
    TPC_ELF_DS_BAILOUT_OPTIONAL_TENSOR         = 20,
    TPC_ELF_DS_BAILOUT_PRINTF_TENSOR           = 21,
    TPC_ELF_DS_BAILOUT_STORE_RMW               = 22,
    TPC_ELF_DS_BAILOUT_NO_TENSOR_REF           = 23,
    TPC_ELF_DS_BAILOUT_INVALID_TENSORS_IDS     = 24,
    TPC_ELF_DS_BAILOUT_MAX_TENSOR_LIMIT        = 25,
    TPC_ELF_DS_BAILOUT_MAX_ARGS_LIMIT          = 26,
    TPC_ELF_DS_BAILOUT_LD_G_ST_G               = 27,
    TPC_ELF_CANNOT_CREATE_TEMP_FILE            = 28,
    TPC_ELF_DS_BAILOUT_VECT_PRED_LDTNSR        = 29
} ;

enum TPCArch {
  Unknown,
  Goya   = 1,
  Gaudi  = 2,
  Goya2  = 3,
  Gaudi2 = 4,
  Gaudi3 = 5
};

enum SCEVStatusMassage{
    SCEVSuccess =               0,
    GC_CUSTOM_MODE =            1,         // SCEV is in GC_CUSTOM mode.
    IF_BRANCH =                 2,         // SCEV Return it sees a if.
    TOKEN_NOT_REC =             3,         // SCEVParser not recognize token
    TOKEN_IS_GLOBAL =           4,         // SCEVParser detect ld_g instruction
    SCEVEmpty =                 6,         // SCEV is empty
    IdGreaterThenNumTensor =    7,         // The indexSpace tensor ID is gather then available tensor in target
    DimSmallerThenZero  =       8,         // At lest one of the indexes is negative number
    //AllZero = 9,                         // One of the dimension is set to all to zero
    TrySetAuxTensor =          10,         // Setting an auxiliary tensor
    NumberOfDimIsGreaterThen5= 11,         // Glue code error: params tensor.geometry.dims is higher then 5
    TensorReg =                12,         // Tensor register is not support by Evaluator
    ExceedParms =              13,         // Exceeded number of params given by the user
    UnvalidLoopSCEV =          14          // SCEV is not valid loop
};

#define TPC_LLVM_DS_BAILOUT_MASK                0x00ffffffffffffff
#define TPC_LLVM_DS_BAILOUT_INVALID_ARCH        0x00ffffffffffffff
#define TPC_LLVM_DS_BAILOUT_OPTIONAL_TENSOR     0x01ffffffffffffff
#define TPC_LLVM_DS_BAILOUT_PRINTF_TENSOR       0x02ffffffffffffff
#define TPC_LLVM_DS_BAILOUT_STORE_RMW           0x03ffffffffffffff
#define TPC_LLVM_DS_BAILOUT_NO_TENSOR_REF       0x04ffffffffffffff
#define TPC_LLVM_DS_BAILOUT_INVALID_TENSORS_IDS 0x05ffffffffffffff
#define TPC_LLVM_DS_BAILOUT_MAX_TENSOR_LIMIT    0x06ffffffffffffff
#define TPC_LLVM_DS_BAILOUT_MAX_ARGS_LIMIT      0x07ffffffffffffff
#define TPC_LLVM_DS_BAILOUT_LD_G_ST_G           0x08ffffffffffffff
#define TPC_LLVM_DS_BAILOUT_VECT_PRED_LDTNSR    0x09ffffffffffffff
#define TPC_LLVM_DS_BAILOUT_FULL_MASK           0xffffffffffffffff

#if defined(__clang__) || defined(__GNUC__)
#pragma pack(push,1)
#endif
struct TPCProgramHeader
{
    uint32_t    version;              // bytes 0-3 (version of header)
    bool        specialFunctionUsed;  // byte 4
    bool        printfUsed;           // byte 5
    bool        lockUnlock;           // byte 6
    bool        mmioUse;              // byte 7
    TPCArch     march;                // bytes 8-11 : target architecture
    uint8_t     paramsNum;            // byte 12
    uint8_t     printTensorNum;       // byte 13 : The printf tensor number
                                      //           allocated by the compiler
    uint8_t     numberOfThreads;      // byte 14 : The number of threads
                                      //           activate in kernel
    uint8_t     directMMIOAccess;     // byte 15 : Indication of hardcoded
                                      //           MMIO address.
    uint8_t     targetSpecific;       // byte 16 : Target Specific Kernel
    uint8_t     dnorm;                // byte 17
    uint32_t    hashedISA;            // bytes 18-21
    uint8_t     reserved_temp[4];     // bytes 22-25
    uint16_t    scalarLoad;           // bytes 26-27 :
    uint16_t    rmwStore;             // bytes 28-29
                                      //  RMW store is a bitmask that indicates
                                      //  whether the tensor has RMW store
                                      //  operation or not. The MSB indicates
                                      //  TensorID 15 (MAX_TENSOR_ID). Example:
                                      //  0b0100 --> tensor id 2 has RMW store
    uint64_t    duplicateTensors;     // bytes 30-37 :
                                      //  Represents the original->duplicate
                                      //  TensorID mapping or indicates Bailout
    uint16_t    genAddrTensorMask;    // bytes 38-39 :
                                      //  Each one of the 16 bits represent
                                      //  whether the corresponding TensorID
                                      //  (bit position) is referenced by a
                                      //  'gen_addr' instruction
    bool        regIDUsed;            // byte 40 :
                                      //  Indicates if Tensor ID is captured
                                      //  in a Register
    bool        irf32Mode;            // byte 41 :
                                      // Indicates if the kernel is executed in
                                      // irf32 mode or irf44 mode
    uint16_t    partialStore;         // bytes 42-43
                                      //  Partial store is a bitmask that indicates
                                      //  whether the tensor has Partial store
                                      //  operation or not. The MSB indicates
                                      //  TensorID 15 (MAX_TENSOR_ID). Example:
                                      //  0b0100 --> tensor id 2 has Partial store
                                      //  Following stoes are considered partial :
                                      //  a) st_tnsr_partial
                                      //  b) st_tnsr_sqz
                                      //  c) st_g
    uint16_t    exclusiveSwitch;       // bytes 44-45
                                      //  Exclusive switch is a bitmask that indicates
                                      //  whether the tensor has exclusive access
                                      //  or not. The MSB indicates
                                      //  TensorID 15 (MAX_TENSOR_ID). Example:
                                      //  0b0100 --> tensor id 2 has exclusive access
                                      //  Only the Following Instructions can use
                                      //  SW_EXC switch :
                                      //  a) ld_g
                                      //  b) st_g
    bool        unsetSmallVLM;        // byte 46
    uint16_t    scalarStore;          // bytes 47-48 :
    uint8_t     temp_padding;         // byte 49
    uint32_t    reserved[53];         // byte 50+ : unused
};
#if defined(__clang__) || defined(__GNUC__)
#pragma pack(pop)
#endif

struct CostModelResult {
    uint64_t cycles;                     // Estimated execution time in cycles
    uint64_t cycles_limited;             // Estimated execution time in cycles (threshold in use)
    uint64_t bw;                         // = LS# * VecSize * TPC_Freq / cycles
    uint64_t loadBW;                     // = VectorLoad# * VecSize * TPC_Freq / cycles
    uint64_t storeBW;                    // = VectorStore# * VecSize * TPC_Freq / cycles
    uint64_t vectorLoad;
    uint64_t vectorStore;
    uint64_t totalLoad;
    uint64_t totalStore;
    uint64_t totalVPUNops;
    uint64_t threshold;
    uint64_t finalDecision;
    uint64_t asicCycles;
    double asicTime;
    SCEVStatusMassage status;
};

/*!
 ***************************************************************************************************
 *   @brief Returns pointer and size of TPC binary from elf buffer
 *
 *   @param pElf            [in]    pointer to elf buffer
 *   @param size            [in]    size of elf buffer
 *   @param pTpcBinary      [out]   Returned pointer to TPC binary on elf buffer
 *   @param tpcBinarySize   [out]   Returned size to TPC binary
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
TpcElfStatus ExtractTpcBinaryFromElf(   const void*  pElf,
                                        uint32_t     elfSize,
                                        void*&       pTpcBinary,
                                        uint32_t&    tpcBinarySize);



/*!
 ***************************************************************************************************
 *   @brief Returns TPC program header from elf buffer
 *
 *   @param pElf            [in]    Pointer to elf buffer
 *   @param size            [in]    Size of elf buffer
 *   @param programHeader   [out]   program header structure.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
TpcElfStatus ExtractTpcProgramHeaderFromElf(    const void*     pElf,
                                                uint32_t        elfSize,
                                                TPCProgramHeader&  programHeader);


TpcElfStatus ExtractTpcBinaryAndHeaderFromElf(const void*  pElf,
                                              uint32_t     elfSize,
                                              void*&       pTpcBinary,
                                              uint32_t&    tpcBinarySize,
                                              TPCProgramHeader&  programHeader);

/*!
 ***************************************************************************************************
 *   @brief Returns estimated cycle count of program execution and BW
 *
 *   @param pHabanaKernelParam [in]    pointer to HabanaKernelParam
 *   @param HabanaKernelInstantiation_t [in]    pointer to HabanaKernelInstantiation
 *   @param CostModelResult    [out]   BW and estimated execution time in cycles
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
TpcElfStatus GetTpcProgramCostModelValuesV2(const gcapi::HabanaKernelParamsV2_t *pHabanaKernelParam,
                                         const gcapi::HabanaKernelInstantiation_t *pHabanaKernelInstantiation,
                                         uint32_t tpcNr, CostModelResult &result);

/* This Version will be deprecated soon, start using GetTpcProgramCostModelValuesV2 */
TpcElfStatus GetTpcProgramCostModelValues(const gcapi::HabanaKernelParamsV2_t *pHabanaKernelParam,
                                         const gcapi::HabanaKernelInstantiation_t *pHabanaKernelInstantiation,
                                         CostModelResult &result);

/* This Version will be deprecated soon, start using GetTpcProgramCostModelValuesV4 */
TpcElfStatus GetTpcProgramCostModelValuesV3(const tpc_lib_api::HabanaKernelParams *pHabanaKernelParam,
                                         const tpc_lib_api::HabanaKernelInstantiation *pHabanaKernelInstantiation,
                                         CostModelResult &result);
/*!
 ***************************************************************************************************
 *   @brief Computes estimated cycle count of program execution and BW
 *
 *   @param pHabanaKernelParam [in]    pointer to HabanaKernelParam
 *   @param HabanaKernelInstantiation_t [in]    pointer to HabanaKernelInstantiation
 *   @param tpcNr [in]    number of tpc cores being used for kernel execution
 *   @param Result [out]    BW and estimated execution time in cycles
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */

TpcElfStatus GetTpcProgramCostModelValuesImpl(const tpc_lib_api::HabanaKernelParams *pHabanaKernelParam,
                                         const tpc_lib_api::HabanaKernelInstantiation *pHabanaKernelInstantiation,
                                         uint32_t tpcNr, CostModelResult &result);

TpcElfStatus GetTpcProgramCostModelValuesV4(const tpc_lib_api::HabanaKernelParams *pHabanaKernelParam,
                                         const tpc_lib_api::HabanaKernelInstantiation *pHabanaKernelInstantiation,
                                         uint32_t tpcNr, CostModelResult &result);


/*!
 ***************************************************************************************************
 *   @brief Fills index space mapping in kernel instatiation struct based on SCEV analysis
 *
 *   @param params        [in]    kernel instantiation struct which holds the elf buffer.
 *   @param instance      [out]   program instance struct.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
TpcElfStatus GetTpcProgramIndexSpaceMapping(   tpc_lib_api::HabanaKernelParams *          params,
                                               tpc_lib_api::HabanaKernelInstantiation*    instance,
                                               SCEVStatusMassage &Status);

/*!
 ***************************************************************************************************
 *   @brief retrieve kernel name
 *
 *   @param params        [in]    kernel instantiation struct which holds the elf buffer.
 *   @param kernelISA     [out]   kernel name
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
TpcElfStatus GetTpcKernelISAName(  tpc_lib_api::HabanaKernelInstantiation *instance,
                                   _OUT_   char*    kernelISA,  unsigned* nameSize);

/*!
 ***************************************************************************************************
 *   @brief retrieve kernel name
 *
 *   @param kernelElf        [in]    The elf buffer.
 *   @param elfSize          [in]    The elf size.
 *   @param kernelISA        [out]   kernel name
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
TpcElfStatus GetTpcKernelISANameByElf(const void *kernelElf, uint32_t elfSize,
                                      char *kernelISA, unsigned *nameSize);

/*!
 ***************************************************************************************************
 *   @brief compile kernel from the IR bitcode in the specified ELF buffer.
 *
 *   @param pSourceElf    [in]    Pointer to source ELF buffer.
 *   @param sourceElfSize [in]    Size of source ELF.
 *   @param options       [in]    Additional compilation options.
 *   @param pResultElf    [out]   Returned pointer to recompiled ELF buffer.
 *   @param resultElfSize [out]   Returned size of recompiled ELF.
 *   @param tempDir       [in]    Directory for temporary files. If nullptr, temporary files
 *                                are not kept.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
TpcElfStatus RecompileKernel(  const void*  pSourceElf,
                               uint32_t     sourceElfSize,
                               const char  *options,
                               void       *&pResultElf,
                               uint32_t    &resultElfSize,
                               bool        for_device = true,
                               const char  *tempDir = nullptr);

/*!
 *   @brief loads a content of the kernel binary object in the specified ELF buffer.
 *
 *   @param name            [in]    Full path to the kernel binary file.
 *   @param pTpcBinary      [in]    Pointer to TPC binaries.
 *   @param tpcBinarySize   [in]    Size of the content buffer.
 *
 *   @return                  The status of the operation
 */

TpcElfStatus LoadTpcBinary(const char *name, void *&pTpcBinary, uint32_t &tpcBinarySize);

/*!
 ***************************************************************************************************
 *   @brief compile kernel IR bitcode
 *
 *   @param pSourceElf    [in]    Pointer to elf buffer.
 *   @param sourceElfSize [in]    Size of elf buffer.
 *   @param targetDevice  [in]    Architecture to compile for.
 *   @param pTpcBinary    [out]   Returned pointer to TPC binary on elf buffer
 *   @param tpcBinarySize [out]   Returned size to TPC binary
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
TpcElfStatus RecompileTpcKernelFromIR(  const void*  pElf,
                                        uint32_t     elfSize,
                                        const char  *compilerOptions,
                                        void       *&pTpcBinary,
                                        uint32_t    &tpcBinarySize);

/*!
 ***************************************************************************************************
 *   @brief compile kernel IR bitcode using knowledge about kernel parameters and index space.
 *
 *   @param pSourceElf    [in]    Pointer to elf buffer.
 *   @param sourceElfSize [in]    Size of elf buffer.
 *   @param targetDevice  [in]    Architecture to compile for.
 *   @param pTpcBinary    [out]   Returned pointer to TPC binary on elf buffer
 *   @param tpcBinarySize [out]   Returned size to TPC binary
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
TpcElfStatus RecompileTpcKernelFromIRTune(const void*  pElf,
                                          uint32_t     elfSize,
                                          const char  *targetDevice,
                                          const tpc_lib_api::HabanaKernelParams *pHabanaKernelParams,
                                          const tpc_lib_api::_HabanaKernelInstantiation *pHabanaKernelInstantiation,
                                          void       *&pTpcBinary,
                                          uint32_t    &tpcBinarySize,
                                          const char  *tempDir = nullptr
                                          );

// New class to gracefully free up ElfBinaryPtr allocated when recompiling any kernel
class ElfData{
public:
    ElfData(void *ptr) : elfptr(ptr) {}
    /**  Uncomment this once all the user have adapted to use RecompileKernelWithMemLeakHandled() API
      ~ElfData() {
        if (elfptr) {
          printf("destroying ElfData ...\n");
          delete[] static_cast<char *>(elfptr);
        }
      }
    **/
    void *get() { return elfptr; }
    void  set(void *ptr) { elfptr = ptr; };
private:
    void *elfptr;
};

/*!
 ***************************************************************************************************
 *   @brief compile kernel from the IR bitcode in the specified ELF buffer.
 *
 *   @param pSourceElf    [in]    Pointer to source ELF buffer.
 *   @param sourceElfSize [in]    Size of source ELF.
 *   @param options       [in]    Additional compilation options.
 *   @param pResultElf    [out]   Returned shared pointer to recompiled ELF buffer.
 *   @param resultElfSize [out]   Returned size of recompiled ELF.
 *   @param tempDir       [in]    Directory for temporary files. If nullptr, temporary files
 *                                are not kept.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
TpcElfStatus RecompileKernelWithMemLeakHandled(const void *pSourceElf,
                                uint32_t sourceElfSize,
                                const char *options,
                                std::unique_ptr<ElfData> &pResultElf,
                                uint32_t &resultElfSize,
                                bool for_device = true,
                                const char *tempDir = nullptr);
} // end of TpcElfTools
}
#endif /* TPC_ELF_API_HPP */
