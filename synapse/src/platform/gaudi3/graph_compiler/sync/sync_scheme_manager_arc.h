
#pragma once

#include "platform/gaudi3/graph_compiler/gaudi3_types.h"
#include "include/sync/overlap.h"
#include "platform/gaudi3/graph_compiler/hal_conventions.h"
#include "graph_compiler/sync/sync_scheme_manager_arc.h"

class Gaudi3Graph;

//----------------------------------------------------------------------------
//  Sync scheme for Gaudi 3 with ARCs
//----------------------------------------------------------------------------
class SyncSchemeManagerArcGaudi3 : public SyncSchemeManagerArc
{
public:
    SyncSchemeManagerArcGaudi3(Gaudi3Graph* graph);
    virtual ~SyncSchemeManagerArcGaudi3() = default;

protected:
    virtual DependencyMap getDependenciesOnControlEdges(const NodePtr& node) const override;

    virtual void createRoiPipelineSyncs(const NodePtr&             node,
                                        const std::list<NodeROI*>& rois,
                                        const DependencyMap&       inputDependencies,
                                        DependencyMap&             outputDependencies,
                                        unsigned&                  emittedSigVal) override;

    virtual void              resetOverlap() override;
    virtual void              archiveOverlap() override;
    virtual unsigned          getLogicalId(const NodePtr& node) const override;
    virtual const NumEngsMap& getNumEngsPerLogicalId() const override;
    virtual unsigned          convertSigValToOverlapIdx(unsigned logicalId, unsigned sigVal) const override;
    virtual std::string       logicalIdToStr(unsigned logicalId) const override;
    virtual std::string       sigValToStr(unsigned logicalId, Settable<unsigned> sigVal) const override;
    void                      validateSigVal(unsigned logicalId, unsigned sigVal) const;

private:
    std::unique_ptr<gaudi3::Overlap> m_pOverlap;
    unsigned m_emittedSigVal[gaudi3::LOGICAL_QUEUE_MAX_ID] = {0};  // tracking the emitted signal for each engine
};
