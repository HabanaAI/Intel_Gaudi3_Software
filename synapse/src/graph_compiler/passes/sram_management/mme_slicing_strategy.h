#pragma once

#include "types.h"
#include "bundle.h"
#include "sliced_operand_traversal.h"
#include "slice_mapping.h"
#include "slicing_strategy.h"
#include "strategy_slicing_data.h"
#include "mme_dim_controller.h"
#include "mme_geometry.h"

class MmeSlicingStrategy;
using pMmeSlicingStrategy = std::shared_ptr<MmeSlicingStrategy>;

// Interface class for SlicingStrategy - define the API for different strategies.
class MmeSlicingStrategy : public SlicingStrategy
{
public:
    class MmeSlicingData;

    virtual void accept(StrategyVisitor& visitor) override {visitor.visit(*this);}
    // create a concrete object of SlicingStrategy
    static pMmeSlicingStrategy createStrategyForMMENode(const HalReader& halReader, const pNode& mmeNode);

    std::string getSlicingDataString(bool exactMatch = true) const override;

    virtual MmeSlicingData& getMmeSlicingData() = 0;
    virtual const MmeSlicingData& getMmeSlicingData() const = 0;
    virtual unsigned getMMENarrowGeometryInElements() const = 0;
    virtual unsigned getMMEWideGeometryInElements() const = 0;
    virtual unsigned alignToMMEWide(unsigned size, bool floor) = 0;
    virtual unsigned alignToMMENarrow(unsigned size, bool floor) = 0;

    // builder methods
    MmeSlicingStrategy& setOutputTraversalPattern(DimVector);
    MmeSlicingStrategy& setGeometry(MmeGeometry geometry);

    void tryAlignToCacheLine() override;

protected:
    MmeSlicingStrategy(const HalReader& halReader, const StrategySlicingDataPtr& slicingData)
    : SlicingStrategy(halReader, slicingData)
    {
    }
    MmeSlicingStrategy(const MmeSlicingStrategy& rhs, bool resetAlignment) : SlicingStrategy(rhs, resetAlignment) {}
    virtual ~MmeSlicingStrategy() = default;
    void tryAlignOperandToCacheLine(pSlicedOperand& operand, bool adjustSramCapacityForSlave);
    bool isValidForCacheLineAlignment(const pSlicedOperand& operand);
};

class MmeSlicingStrategy::MmeSlicingData : public StrategySlicingData
{
public:
    MmeSlicingData(const HalReader& halReader, const TensorVector& inputTensors, const pTensor& outputTensor);
    MmeSlicingData(const MmeSlicingData& other);

    StrategySlicingDataPtr clone() const override;

    virtual bool compareInitialSlicing(const StrategySlicingData& other, bool exactMatch = true) const override;

    SlicedOperandTraversalPattern getOutputSlices() const override;

    void setDimControllerNode(const pNode& mmeNode);
    // Narrow == the dimension in which coordinate change in every iteration (almost)
    pSlicedOperand& getWide();
    const pSlicedOperand& getWide() const;
    const DimVector&      getWideNonCommonSlicingDims() const;
    const DimVector&      getWideOutputSlicingDims() const;
    const DimVector&      getWideCommonSlicingDims() const;
    // Wide   == the dimension in which coordinate change only after a complete pass on the narrow dimension.
    pSlicedOperand& getNarrow();
    const pSlicedOperand& getNarrow() const;
    const DimVector&      getNarrowNonCommonSlicingDims() const;
    const DimVector&      getNarrowOutputSlicingDims() const;

    pSlicedOperand getSlaveOutputOperand() const;

    StrategySlicingData::WalkingDir getWalkingDir() const override;
    uint64_t getCommonDimSize() const;
    uint64_t getQRSSize() const;

    NodeSet getStrategyNodes(const pBundle& bundle) const override;

    NodeSet getStrategyProducers() const;

    using RoleCandidatesArray = std::array<pBundleExpansion, BundleExpansion::NumOfRoles>;
    RoleCandidatesArray& getRoleCandidates();
    const std::list<pBundleExpansion>& getInvalidCandidates() const;
    const RoleCandidatesArray& getRoleCandidates() const;
    void addValidCandidate(const pBundleExpansion& candidate, bool needToAdjust = true);
    void addInvalidCandidate(const pBundleExpansion& candidate, bool needToAdjust = true);
    bool blockExpansionForRole(BundleExpansion::Role role);

    // MME Geometry
    MmeGeometry MMEGeometryUsed = gaudi_geometry_2wx2h;

    // When a candidate is added, any slicing data it contain (chunk sizes, etc.) may not have come
    // from the current strategy, so it may differ. This method creates a new candidated from the
    // original one with the slicing data replaced to be the slicing data of the current strategy.
    pBundleExpansion getAdjustedCandidate(const pBundleExpansion& origCandidate);
    void addSlaveTraversalPattern(const pBundleExpansion& candidate);
    SlicedOperandTraversalPattern getSlaveTraversalPattern(const pNode& slaveNode,
                                                           const pTensor& sharedInput,
                                                           const pSlicedOperand& slaveOutputOperand);
    void setCandidateAsBundled(pBundleExpansion& candidate);
    bool isCandidateInBundle(const pBundleExpansion& candidate);

    bool hasRole(BundleExpansion::Role role) const;

    bool isNodeStitched(const pNode& node) const;

    // Sets the additional SRAM capacity that this candidate would require if stitched
    void adjustSRAMCapacity(pBundleExpansion& candidate);

private:
    const HalReader&                           m_halReader;
    std::unordered_map<pBundleExpansion, bool> m_candidateInBundle;       /* For candidate c if  m_candidateInBundle[c] is true
                                                                           * --> c is stitched to bundle */
    std::shared_ptr<MmeDimController>          m_dimController = nullptr;
    RoleCandidatesArray                        m_roleValidCandidates {};  /* Candidates that will be stitched to the bundle       */
    std::list<pBundleExpansion>                m_invalidCandidates;       /* Candidates that will be used for strategy comparison */
    bool                                       m_sharedMMEInputStitchedToNarrow = false;

    pSlicedOperand findSlaveOutput();
    uint32_t computeSRAMCapacity(const pSlicedOperand& operand);

    /* Looking for slave stitched operand - slave producer and consumer
     * The salve operand may be valid or invalid candidate, so we have to look in both lists.
     * Pre-condition - given candidate must be SlaveWide/NarrowInputProducer or SlaveOutputConsumer*/
    pSlicedOperand findStitchedOperandForCandidatesDependsOnSlave(const pBundleExpansion& origCandidate);
};
