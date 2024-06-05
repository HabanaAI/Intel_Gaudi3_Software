#ifndef MME__GAUDI3_MME_DESCRIPTOR_GENERATOR_H
#define MME__GAUDI3_MME_DESCRIPTOR_GENERATOR_H

#include <list>
#include <memory>
#include <vector>
#include "include/gaudi3/gaudi3_mme_signal_info.h"
#include "include/mme_common/mme_descriptor_generator_base.h"
#include "include/utils/gaudi3_desc_dumper.h"

namespace gaudi3
{
using MmeActivation = MmeCommon::MmeActivation<Mme::Desc>;
using ActivationVec = MmeCommon::ActivationVec<Mme::Desc>;

class MmeDescriptorGenerator : public MmeCommon::MmeDescriptorGenerator<Mme::Desc>
{
public:
    virtual ~MmeDescriptorGenerator() = default;
    void mmeGenerateActivations() override;

    static unsigned countSignals(const Mme::Desc* desc);

    /* LayerParams getter - original params are the original input from the user (GC) - not changed at all,
     * whereas params is the sub-problem params - same as originalParams but
     * they could be altered in case we have several sub-problems (see dedx with strides or dedw with unroll).
     * original params should be used in general , but when looking for tensor sizes\strides\bases - better to take
     * params.
     * In BGemm - original params and params are the same - no division to sub problems and no alteration of the
     * user layer params.
     */
    virtual const MmeCommon::MmeLayerParams& getParams() const
    {
        return (getCurrentSubProblem() == nullptr) ? getOriginalParams() : getCurrentSubProblem()->params;
    };
    static std::unique_ptr<MmeDescriptorGenerator>
        createMmeDescGenerator(bool isDmaOperation = false, unsigned totalNumOfMmes = Mme::MME_CORE_MASTERS_NR);
    void mmeGenerateNullDescs();
    void setZeroActivationsForDcore();
    virtual ActivationVec& getMmeActivations() override { return isPerforated() ? m_dcoreActivations : m_activations; }
    virtual const ActivationVec& getMmeActivations() const override
    {
        return isPerforated() ? m_dcoreActivations : m_activations;
    }
    virtual const unsigned getDcoreActivationSize() const  // for debug purpose
    {
        return m_dcoreActivations.size();
    }
    void reorderDcoreActivations(const MmeCommon::MmeLayerParams& fullNodeParams);
    virtual bool setParams(const MmeCommon::MmeLayerParams& newParams) override;

    //  patching function for the actual dualGemm sizes
    virtual void patchDualGemm(MmeCommon::MmeTensorView& x,
                               MmeCommon::MmeTensorView& w,
                               MmeCommon::MmeTensorView& y,
                               const uint64_t addrX,
                               const uint64_t addrW,
                               const uint64_t addrY,
                               const uint64_t addrO,
                               const bool YisSram,
                               const bool OisSram,
                               unsigned gemmIdx) override;

    //  this function is used to set the same SO to each MME across all activations, used by mme user.
    virtual void mmePatchSyncObjects(const uint32_t mmeIdx,
                                     const uint32_t addr0,
                                     const uint32_t addr1,
                                     const uint32_t slaveAddr0 = 0,
                                     const uint32_t slaveAddr1 = 0) override;

    //  this function is used when each descriptors gets a unique SO.
    static void mmePatchSyncObject(Mme::Desc& desc,
                                   const uint32_t addr0,
                                   const uint32_t addr1,
                                   const uint32_t slaveAddr0 = 0,
                                   const uint32_t slaveAddr1 = 0);

    virtual void patchSignalColoring(MmeActivation& activation, const bool addr0isSram, const bool addr1isSram) override;
    void mmeGetDescValidMask(const Mme::Desc& desc,
                             bool* mask,
                             bool* aguOut1FromAguOut0_DW0,
                             bool* aguOut1FromAguOut0_DW1_4) override;

    virtual std::vector<std::string> getRecipeDebugInfo(bool verbose = true) const override;
    virtual std::string getRecurringMisalignmentDebugInfo() const override;
    virtual void patchMcids(uint16_t mcidA, uint16_t mcidB, uint16_t mcidC) override;
    virtual void patchMcids(uint16_t mcidA,
                            uint16_t mcidB,
                            uint16_t mcidC,
                            std::optional<uint16_t> mcidAuxScratchpad,
                            std::optional<uint16_t> mcidAuxReductionAdd) override;

protected:
    MmeDescriptorGenerator(unsigned totalNumOfMmeUnits) { m_totalNumOfMmeUnits = totalNumOfMmeUnits; }
    virtual bool validateParams(const MmeCommon::MmeLayerParams& params, std::string& errorMsg) override;

    void buildDesc(unsigned mmeIDx, Mme::Desc* desc);
    void buildEmptyJobDesc(Mme::Desc* desc, bool isLast, unsigned signalsToAdd);
    std::string getVerboseRecipeSummaryStr() const;
    virtual void setSBCacheDisable(const MmeCommon::EMmeOperand operand,
                                   const bool isSram,
                                   const bool addrIsAligned,
                                   Mme::Desc& desc) override {};  // TODO: implement add Ticket!!
    virtual void setSinglePerfEvent(MmeCommon::EMmeTraceEngine engine, unsigned startEndMask, Mme::Desc& desc) override;
    unsigned calcDcoreNumSignals(const unsigned targetDcoreIdx);
    unsigned getMaxNumSignals();

protected:
    virtual void setSimpleFields(Mme::Desc* desc) = 0;  //  will be deleted later
    virtual std::shared_ptr<MmeDescriptorDumper<gaudi3::Mme::Desc>> getDescriptorDumper() const override
    {
        return std::make_shared<Gaudi3DescriptorDumper>();
    }
    unsigned getTotalNumOfMmeUnits() { return m_totalNumOfMmeUnits; }
    Gaudi3SignalingInfo m_signalingInfo;
    ActivationVec m_dcoreActivations;
    ActivationVec::iterator m_dcoreStartActivation;
    std::vector<unsigned> m_numOfActivationsPerDcore;
    unsigned m_totalNumOfMmeUnits;
};

class MmeConvDescriptorGenerator : public MmeDescriptorGenerator
{
public:
    explicit MmeConvDescriptorGenerator(unsigned totalNumOfMmeUnits);
    virtual ~MmeConvDescriptorGenerator() = default;

protected:
    virtual void setSimpleFields(Mme::Desc* desc) override;  //  will be deleted later
private:
    void createHardcodedABDesc(Mme::Desc* desc, unsigned mmeIdx);
    void createHardcodedAtBDesc(Mme::Desc* desc, unsigned mmeIdx);
};

class MmeDmaDescriptorGenerator : public MmeDescriptorGenerator
{
public:
    explicit MmeDmaDescriptorGenerator(unsigned totalNumOfMmeUnits);
    virtual ~MmeDmaDescriptorGenerator() = default;

protected:
    virtual void setSimpleFields(Mme::Desc* desc) override;  //  will be deleted later
};

using pMmeDescriptorGenerator = std::unique_ptr<MmeDescriptorGenerator>;
}  // namespace gaudi3

#endif //MME__GAUDI3_MME_DESCRIPTOR_GENERATOR_H
