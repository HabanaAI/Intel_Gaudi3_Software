#pragma once
#include <vector>
#include "json.hpp"
#include "mme_verification/common/mme_reg_write_cmd.h"
#include "mme_user.h"
#include "mme_verification/common/sync_object_manager.h"
#include "mme_user.h"
#include "device_handler.h"
#include "config_parser.h"

namespace MmeCommon
{
// Test specific resources
typedef struct testResources_t
{
    std::vector<std::vector<MmeQmanCmd>> cmds;
    unsigned numOfSoIdx = 0;
    MmeDataParams testDataParams;
    MmeMemoryUsage testMemUsage;
    nlohmann::json* testJson = nullptr;
    unsigned testId = 0;
    unsigned mmeLimit = 0;
} TestResources;

typedef std::vector<std::unique_ptr<TestResources>> GroupType;

struct MmeTestAllocator : public Allocator
{
    MmeTestAllocator() = default;

    ~MmeTestAllocator()
    {
        for (void* p : m_ptrs)
        {
            if (p)
            {
                free(p);
            }
        }
    }

    void* alloc(size_t size)
    {
        void* ret = malloc(size);
        m_ptrs.push_back(ret);
        return ret;
    }

    std::vector<void*> m_ptrs;
};

class MmeTestManager
{
public:
    MmeTestManager(const ChipType chipType);
    virtual ~MmeTestManager() = default;
    virtual bool runTests(std::vector<nlohmann::json>& testsParams,
                          const std::string& dumpDir,
                          const std::string& dumpUnit,
                          const EMmeDump dumpMmes,
                          const unsigned mmeDumpIdx,
                          const std::string& lfsrDir,
                          const DeviceType devTypeA,
                          const DeviceType devTypeB,
                          std::vector<unsigned>& deviceIdxs,
                          const unsigned seed,
                          const unsigned numOfThreads,
                          unsigned mmeLimit = 0,
                          const bool checkRoi = false,
                          const bool chipAlternatives = false);

    void createLfsrValues(uint32_t seed,
                          LfsrData& lfsrData,
                          uint32_t euNr,
                          uint32_t lfsrNumRegs,
                          bool duplicateForAllCores,
                          bool configLfsr);

    void generateProgram(MmeUser& mmeUser,
                         const GroupType& group,
                         SyncObjectManager& soMgr,
                         std::list<CPProgram>& progs,
                         std::list<CPProgram>& powerProgs,
                         bool& firstValidTest,
                         const LfsrData& lfsrData,
                         unsigned seed,
                         unsigned stream,
                         PmuConfig pmuCfgMode);

    void setscalFw(bool scalFw) { m_scalFw = scalFw; };
    void setPowerTest(bool powerTest) { m_powerTest = powerTest; };

private:
    bool runRecipeTest(const json& testJson, std::string& errorStr);
    bool runOptimizationTest(const json& testJson, std::string& errorStr);
    bool unitTestValue(const json& testJson, const std::string& field, unsigned actualVal, std::string& errorStr);
    bool unitTestArray(const json& testJson,
                       const std::string& field,
                       std::vector<unsigned> actualVal,
                       std::string& errorStr);
    void initStrategyFromJson(const nlohmann::json& testJson, unsigned mmeLimit, MmeStrategy& strategy);
    void initControlsFromJson(const nlohmann::json& testJson, MmeControls& controls);
    void initTestParamsFromJson(const nlohmann::json& testJson, MmeTestParams* testParams);
    void initMemoryConfigFromJson(const nlohmann::json& testJson, MmeMemoryConfig& memoryConfig);
    EMmePattern getMmeStackPattern(const nlohmann::json& testJson, const EMmeOpType op);

protected:
    ChipType m_chipType;
    bool m_scalFw;
    bool m_powerTest;
    const MmeCommon::MmeHalReader& m_mmeHal;
    std::unique_ptr<MmeUser> m_mmeUser = nullptr;
    std::unique_ptr<DeviceHandler> m_devHandler = nullptr;
    std::unique_ptr<SyncObjectManager> m_syncObjectManager = nullptr;
    virtual void makeMmeUser(unsigned mmeLimit) = 0;
    virtual void
    makeDeviceHandler(MmeCommon::DeviceType devA, MmeCommon::DeviceType devB, std::vector<unsigned>& deviceIdxs) = 0;
    virtual void makeSyncObjectManager(const uint64_t smBase, unsigned mmeLimit) = 0;
    virtual std::shared_ptr<MmeCommon::MmeMemAccessChecker> createAccessChecker(unsigned euNr) = 0;
    bool verifyTestParams(nlohmann::json& testJson);
    virtual bool verifyChipSpecificTestParams(nlohmann::json& testJson) = 0;
    static bool verifyTensorStrides(const std::vector<int>& sizes, const std::vector<int>& strides, std::string& errorMessage);
    virtual void fixCacheModeAlloc(std::vector<nlohmann::json>& tests);
    virtual void initMemAttrib(const nlohmann::json& testJson, MmeTensorMemoryAttrib& memAttrib);
    void initConvParamsFromJson(const nlohmann::json& testJson, ConvolutionParams* convParams);
    void allocateSyncObjects(TestResources& testResources, unsigned mmeNr);
    bool createTestActivations(TestResources& testResources,
                               ConvolutionParams& convParams,
                               const MmeTensorMemoryAttrib& memoryAttrib,
                               unsigned& actNum);
    bool canAddTestToGroup(const GroupType& group,
                           const unsigned currentGroupId,
                           MmeMemoryUsage& groupMemUsage,
                           const TestResources& curTest,
                           const unsigned testLimit,
                           const MmeTensorMemoryAttrib& memAttrib,
                           const bool programInSram,
                           const bool groupClipInfIn,
                           const bool testClipInfIn);
    void patchTestCmds(TestResources& testResources, MmeMemoryUsage& groupMemUsage, bool& firstValidTest);
    void runReference(const ChipType chipType,
                      const nlohmann::json& testJson,
                      const ConvolutionParams& convParams,
                      MmeDataParams& testDataParams,
                      MmeMemoryConfig memConfig,
                      unsigned seed,
                      unsigned numOfThreads,
                      uint32_t lfsrNumRegs,
                      uint32_t* lfsr,
                      uint32_t polynomial);
    bool usesLfsr(const nlohmann::json& testJson);
    void updateGroupMemUsage(MmeMemoryUsage& groupMemUsage, MmeMemoryUsage& testMemUsage);
    bool shouldRunGroup(const GroupType& group, unsigned testLimit, bool canAddCurTest, bool currTestForceCloseGroup);
    uint64_t calcProgAddr(const std::list<CPProgram>& progs,
                          MmeMemoryUsage& groupMemUsage,
                          const bool programInSram,
                          const MmeTensorMemoryAttrib& memAttrib);
    void gatherDmaBuffers(const std::vector<MmePciDma>& dmaVec,
                          std::list<Buffer>* inputBuffers,
                          std::list<Buffer>* outputBuffers);
    void copyOutputTensors(const GroupType& group, bool useSimulatorDev, bool hostToDevice = true, unsigned devIdx = 0);
    bool doTensorComparison(const MmeDataParams& testDataParams,
                            const unsigned testCounter,
                            bool runOnSim,
                            bool runOnChip,
                            bool runOnRef,
                            nlohmann::json& testJson,
                            const std::string& testInfoStr);
    bool shouldAddNullDescToGroup(const bool allowedToAddNullDescriptors,
                                  const bool addNullDescToTest,
                                  const bool nullDescInGroup);
    void dumpTensors(const MmeDataParams& testDataParams, EMmeOpType op, bool runOnSim, bool runOnChip);
    bool compareDCDCResults(nlohmann::json& testJson,
         EMmeOpType op,
         const MmeDataParams& testDataParams,
         bool runOnSim,
         bool runOnChip);
         
    virtual bool runAndCompareGroup(const GroupType& group,
                                    SyncInfo& groupSI,
                                    unsigned groupId,
                                    uint64_t programAddr,
                                    std::list<CPProgram>& progs,
                                    std::list<CPProgram>& powerProgs,
                                    unsigned stream,
                                    CoralMmeHBWSniffer& hbwSniffer,
                                    CoralMmeLBWSniffer& lbwSniffer,
                                    MMEConfigPrinter& printer,
                                    const std::string& dumpDir,
                                    const unsigned seed,
                                    const unsigned mmeLimit);
    virtual uint64_t getHbmBaseAddress(uint64_t hbmBase);

    unsigned getMmePerDie() const;
    unsigned getDieNr() const;
    unsigned getMmeNr() const;
    unsigned getEuNr() const;
    unsigned getLFSRSeedsNr() const;
    uint64_t getSramStart(unsigned dieNr) const;
    uint64_t getSramSize(unsigned dieNr) const;
};
}  // namespace MmeCommon
