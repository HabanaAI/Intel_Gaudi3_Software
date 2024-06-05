#pragma once

#include "convolution_params.h"
#include "coral_user_program_base.h"
#include "mme_common/mme_descriptor_generator_base.h"
#include "mme_mem_access_checker.h"
#include "mme_verification/common/mme_test_data_gen.h"
#include "mme_verification/common/mme_reg_write_cmd.h"
#include "mme_verification/common/sync_object_manager.h"  // TODO split file and move to common

namespace MmeCommon
{
using SubParamsVec = std::vector<MmeLayerParams>;
using OperandRolesVec = std::vector<OperandRoles>;

typedef struct
{
    uint64_t hbmBase;
    uint64_t sramBase;
    uint64_t hbmSize;
    uint64_t sramSize;
    //    uint32_t hostMemSize;
} MmeTensorMemoryAttrib;

struct MmeTestParams
{
    bool randomMD;
    bool wkldIdMD;
    unsigned repeats;
    bool incDec;
    bool maskSignals;
    bool prefetchA;
    bool prefetchB;
    bool fullDesc;
    EMmeTraceMode traceMode;
    bool recipeTest;
    bool optimizationTest;
    bool testOutputTensor;

    // Power management parameters:
    bool powerTest;
    unsigned powerLoops;
    unsigned powerIdleCycles;
    unsigned powerIdleLoops;

    // Unit testing
    bool useBrain;  // run the MME brain to select geometry and pattern
};

typedef struct
{
    char* hostAddr;
    uint64_t deviceAddr;
    uint64_t size;
    bool host2device;
    bool isSram;
} MmePciDma;

typedef struct
{
    uint64_t hbmUsage = 0;
    uint64_t sramUsage = 0;
    uint32_t hostMemUsage = 0;
    std::vector<MmePciDma> dmaList;
} MmeMemoryUsage;

struct OperandsInfo
{
    const MmeSimTensor* a = nullptr;
    const MmeSimTensor* b = nullptr;
    const MmeSimTensor* c = nullptr;
    bool aInSram = false;
    bool bInSram = false;
    bool cInSram = false;
};

typedef struct lfsrData_t
{
    std::vector<uint32_t> lfsrPolynomial;
    std::vector<std::vector<uint32_t>> lfsrRegs;
    bool duplicateLfsrValuesForAllCores;
} LfsrData;

class MmeUser
{
public:
    MmeUser(const ChipType chipType, unsigned mmeNr);
    virtual ~MmeUser() = default;
    void setSyncObjectManager(SyncObjectManager* soManager) { m_syncObjectManager = soManager; }
    void setDmaMode(bool dmaDesc) { m_dmaDesc = dmaDesc; }
    const MmeRecipe& getRecipe() const;
    const CommonGeoAttr& getGeoAttr() const;
    const MmeLayerParams& getParams(unsigned idx) { return m_subParams[idx]; }

    //  descriptor generation
    void doTensorAllocation(const EMmeOpType op,
                      const MmeTensorMemoryAttrib& memAttrib,
                      const MmeStrategy& strategy,
                      MmeDataParams& dataParams,
                      MmeMemoryUsage& memUsage);

    bool createActivations(const EMmeOpType op,
                           const ConvolutionParams& conv,
                           const MmeTensorMemoryAttrib& memAttrib,
                           const MmeStrategy& strategy,
                           const MmeControls& controls,
                           const MmeMemoryConfig& memoryConfig,
                           MmeDataParams& dataParams,
                           MmeMemoryUsage& memUsage,
                           unsigned& actNum,
                           const MmeTestParams& testParams,
                           unsigned testId);

    bool patchActivations(const EMmeOpType op,
                          const MmeTensorMemoryAttrib& memAttrib,
                          const MmeStrategy& strategy,
                          MmeDataParams& dataParams,
                          MmeMemoryUsage& memUsage,
                          unsigned testId);

    void patchTensors(MmeDataParams& dataParams, const uint64_t sramUsage, const uint64_t hbmUsage);
    void createSubNodeParams(const MmeLayerParams& params, const std::vector<MmeTensorParams>& tensorParams,
                          SubParamsVec& subParamsVec, std::vector<OperandRoles>& tensorRoleVec);

    //  commands generation
    virtual void buildCmds(const MmeTestParams& testParams,
                           std::vector<MmeQmanCmd> cmds[],
                           MmeDataParams& dataParams,
                           bool& firstValidTest,
                           MmeMemoryUsage testMemUsage) = 0;

    // code generation
    void initProgram(std::list<CPProgram>& progs, std::list<CPProgram>& powerProgs, unsigned stream);

    void generateSingleTestProgram(std::list<CPProgram>& progs,
                                   std::list<CPProgram>& powerProgs,
                                   std::vector<MmeQmanCmd> cmds[],
                                   SyncObjectManager& soMgr,
                                   MmeDataParams& dataParams,
                                   const bool firstValidTest,
                                   const bool clipInfIn,
                                   const bool configLfsr,
                                   const LfsrData& lfsrData,
                                   const unsigned seed,
                                   const unsigned stream,
                                   const unsigned testGroupSize,
                                   const unsigned testIdInGroup,
                                   PmuConfig pmuCfgMode);

    virtual void createNullDescCmds(MmeDataParams& dataParams, std::vector<MmeQmanCmd> cmds[]) = 0;

    virtual std::list<CPProgram> createSimulatorProgram(const std::list<CPProgram>& progs, unsigned seed) = 0;
    virtual void setSoForPowerTest(CPProgram& prog, bool isPowerProg) = 0;
    void setDoStaticConfig(bool val) { m_canDoStaticConfig = val; }
    bool canDoStaticConfig() const { return m_canDoStaticConfig; }
    void setscalFw(bool scalFw) { m_scalFw = scalFw; };
    void setPowerTest(bool powerTest) { m_powerTest = powerTest; };

protected:
    ChipType m_chipType;
    bool m_powerTest;
    bool m_scalFw;
    bool m_dmaDesc = false;
    const unsigned m_mmeNr;
    const MmeCommon::MmeHalReader& m_mmeHal;
    SubParamsVec m_subParams;
    pMmeDescriptorGeneratorBase m_descGenerator;
    // non-owning ptr to TestManager sync object manager.
    SyncObjectManager* m_syncObjectManager = nullptr;

    // code generation
    virtual void addMmeLfsrInitSequence(CPProgram& prog, unsigned seed, const LfsrData& lfsrData, bool configLfsr) = 0;
    virtual void addClipInfInputConfig(CPProgram& prog, const bool clipInfIn) = 0;
    virtual void addMessageBarrier(CPProgram& prog) = 0;

    void addCmdsToProg(std::vector<MmeQmanCmd>& cmds, CPProgram& prog, CPProgram& powerProg);
    std::array<std::pair<uint32_t, uint32_t>, 4> getRandomRedundancyFmaWithBitMask(unsigned seed);

    virtual uint64_t getMmeCtrlBase() = 0;
    virtual unsigned getMmeQueueId(unsigned mmeIdx, unsigned stream) = 0;
    virtual void pushFenceCmd(CPProgram& prog, unsigned fenceIdx, unsigned incWaitVal) = 0;
    virtual void pushWaitCmd(CPProgram& prog, unsigned waitCycles, unsigned waitValue, unsigned waitIdx) = 0;

private:
    //  descriptor generation
    void createLayerParams(const EMmeOpType opType,
                           const unsigned wkldId,
                           const ConvolutionParams& conv,
                           const std::vector<MmeTensorParams>& tensorParams,
                           const MmeStrategy& strategy,
                           const MmeControls& controls,
                           const MmeMemoryConfig& memoryConfig,
                           const MmeTestParams& testParams,
                           SubParamsVec& subParamsVec,
                           OperandRolesVec& operandRolesVec);

    void initConv(const ConvolutionParams& in, MmeConv* out);
    void initMmeTensorView(const MmeSimTensor& tensor, MmeTensorView* view);
    void makeOperandsInfo(const EMmeOpType op,
                          const MmeDataParams& dataParams,
                          const MmeTensorParams& tensorParams,
                          OperandsInfo* info);
    void setInputs(EMmeOpType op, bool& isInputX, bool& isInputW, bool& isInputY);
    void getMemAttribPtrs(const bool isSram,
                          const MmeTensorMemoryAttrib& attrs,
                          MmeMemoryUsage& usage,
                          uint64_t** offsetPtr,
                          const uint64_t** sizePtr,
                          const uint64_t** basePtr);
    void allocTensorDeviceMem(const bool isSram,
                              const bool isInput,
                              const MmeSimTensor& t,
                              const MmeTensorMemoryAttrib& memAttrib,
                              MmeMemoryUsage& memUsage,
                              uint64_t& addr,
                              bool isAligned);
    void allocTensorsDeviceMem(const EMmeOpType op,
                               const MmeDataParams& dataParams,
                               const MmeTensorParams& tensorParams,
                               const MmeTensorMemoryAttrib& memAttrib,
                               MmeMemoryUsage& memUsage,
                               uint64_t& addrA,
                               uint64_t& addrB,
                               uint64_t& addrC,
                               uint64_t& addrO,
                               bool alignedAddresses);
    void allocPrimaryTensors(const EMmeOpType op,
                         MmeDataParams& dataParams,
                         MmeTensorParams& tensorParams,
                         const MmeTensorMemoryAttrib& memAttrib,
                         MmeMemoryUsage& memUsage,
                         bool alignedAddresses);
    void allocAuxTensors(const EMmeOpType op,
                         MmeDataParams& dataParams,
                         MmeTensorParams& tensorParams,
                         const MmeTensorMemoryAttrib& memAttrib,
                         MmeMemoryUsage& memUsage,
                         bool alignedAddresses);
    void getTensorAddresses(const EMmeOpType op,
                            const MmeTensorParams& tensorParams,
                            uint64_t& addrA,
                            uint64_t& addrB,
                            uint64_t& addrC,
                            uint64_t& addrO);

    //  commands generation
    static void setPMUConfiguration(CPProgram& prog, PmuConfig pmuCfgMode);
    bool m_canDoStaticConfig = true;
};

}  // namespace MmeCommon
