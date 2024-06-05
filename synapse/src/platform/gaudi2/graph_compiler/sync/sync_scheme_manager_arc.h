#pragma once

#include "include/sync/overlap.h"
#include "platform/gaudi2/graph_compiler/hal_conventions.h"
#include "graph_compiler/sync/sync_scheme_manager_arc.h"

class Gaudi2Graph;

using OverlapArcGaudi2 = Overlap<gaudi2::LOGICAL_QUEUE_MAX_ID>;

//----------------------------------------------------------------------------
//  Overlap signal to ARC signal converter
//----------------------------------------------------------------------------
class OverlapSigToArcSigGaudi2
{
public:
    static const unsigned SOB_VALUE_MASK_TPC    = 0x7FE0;  // TPC's sob-value bits
    static const unsigned SOB_VALUE_MASK_MME    = 0x1FFF;  // MME's sob-value bits
    static const unsigned SOB_VALUE_MASK_DMA    = 0x7FFF;  // DMA's sob-value bits
    static const unsigned SOB_VALUE_MASK_ROT    = 0x1FFF;  // ROT's sob-value bits
    static const unsigned c_tpcSharedSobSetSize = 32;      // FW uses two sets of 32 SOBs (each) in TPC Shared SOB mode

    OverlapSigToArcSigGaudi2();

    // Convert overlap dependencies to ARC dependencies
    void convert(const OverlapDescriptor&               overlapDesc,    // input overlap descriptor
                 const OverlapArcGaudi2::DependencyCtx& overlapDep,     // input overlap dependencies
                 OverlapArcGaudi2::DependencyCtx&       arcDep,         // output arc dependencies
                 unsigned&                              emittedSignal); // output emitted signal

    // Convert ARC signal value to overlap signal index
    unsigned reverseConvert(unsigned logicalId, unsigned sigVal) const;

    static unsigned safeIncrement(unsigned logicalId, unsigned sigVal, unsigned incAmount);

private:
    unsigned m_physicalSigVal[gaudi2::LOGICAL_QUEUE_MAX_ID] = {0};  // tracking on the physical signal for each engine
    unsigned m_overlapSigIdx[gaudi2::LOGICAL_QUEUE_MAX_ID]  = {0};  // tracking on the overlap index for each engine

    struct SigOffset
    {
        SigOffset(unsigned s, unsigned o) : start(s), offset(o) {}
        unsigned start;   // starting from this overlap signal
        unsigned offset;  // add this offset
    };
    std::vector<SigOffset> m_sigOffsets[gaudi2::LOGICAL_QUEUE_MAX_ID];
};

//----------------------------------------------------------------------------
//  Sync scheme for Gaudi 2 with ARCs
//----------------------------------------------------------------------------
class SyncSchemeManagerArcGaudi2 : public SyncSchemeManagerArc
{
public:
    SyncSchemeManagerArcGaudi2(Gaudi2Graph* graph);
    virtual ~SyncSchemeManagerArcGaudi2() = default;

protected:
    virtual void createRoiPipelineSyncs(const NodePtr&             node,
                                        const std::list<NodeROI*>& rois,
                                        const DependencyMap&       inputDependencies,
                                        DependencyMap&             outputDependencies,
                                        unsigned&                  emittedSigVal) override;

    virtual void              resetOverlap() override;
    virtual void              archiveOverlap() override { /* not implemented for gaudi2 */ }
    virtual unsigned          getLogicalId(const NodePtr& node) const override;
    virtual const NumEngsMap& getNumEngsPerLogicalId() const override;
    virtual unsigned          convertSigValToOverlapIdx(unsigned logicalId, unsigned sigVal) const override;
    virtual std::string       logicalIdToStr(unsigned logicalId) const override;
    virtual std::string       sigValToStr(unsigned logicalId, Settable<unsigned> sigVal) const override;
    void                      validateSigVal(unsigned logicalId, unsigned sigVal) const;

private:
    std::shared_ptr<OverlapArcGaudi2>         m_pOverlap;
    std::shared_ptr<OverlapSigToArcSigGaudi2> m_pConverter;
};
