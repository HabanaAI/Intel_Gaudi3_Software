#include "sync_scheme_manager_arc.h"

#include "platform/gaudi3/graph_compiler/gaudi3_code_generator.h"
#include "platform/gaudi3/graph_compiler/gaudi3_graph.h"
#include "sync_utils.h"
#include "types_exception.h"

#include <sstream>

static unsigned trivialSafeIncrement(unsigned logicalId, unsigned sigVal, unsigned incAmount)
{
    // In contrast to gaudi2, in gaudi3 we don't need special handling for signal increment
    return sigVal + incAmount;
}

SyncSchemeManagerArcGaudi3::SyncSchemeManagerArcGaudi3(Gaudi3Graph* graph)
: SyncSchemeManagerArc(graph), m_pOverlap(std::make_unique<gaudi3::Overlap>())
{
    m_sobResetMngr.bindSafeIncrement(&trivialSafeIncrement);
    m_sobResetMngr.setCmeExist();
}

void SyncSchemeManagerArcGaudi3::archiveOverlap()
{
    Gaudi3CodeGenerator& codeGen = dynamic_cast<Gaudi3CodeGenerator&>(*(m_graph->getCodeGenerator()));
    codeGen.archiveOverlap(std::move(m_pOverlap));
}

void SyncSchemeManagerArcGaudi3::resetOverlap()
{
    m_pOverlap.reset(new gaudi3::Overlap());
    memset(m_emittedSigVal, 0, sizeof(m_emittedSigVal));
}

unsigned SyncSchemeManagerArcGaudi3::getLogicalId(const NodePtr& node) const
{
    return gaudi3::deviceTypeToLogicalQueue(m_graph->getNodeUtility().getNodeDeviceType(node), *node);
}

const SyncSchemeManagerArcGaudi3::NumEngsMap& SyncSchemeManagerArcGaudi3::getNumEngsPerLogicalId() const
{
    static SyncSchemeManagerArcGaudi3::NumEngsMap ret;
    if (ret.empty())
    {
        ret[gaudi3::DEVICE_MME_LOGICAL_QUEUE] = m_graph->getHALReader()->getNumMmeEngines();
        ret[gaudi3::DEVICE_TPC_LOGICAL_QUEUE] = m_graph->getHALReader()->getNumTpcEngines();
        ret[gaudi3::DEVICE_ROT_LOGICAL_QUEUE] = m_graph->getHALReader()->getNumRotatorEngines();
        ret[gaudi3::DEVICE_XPS_LOGICAL_QUEUE] = m_graph->getHALReader()->getNumMmeEngines();  // transpose is inside mme
    }
    return ret;
}

DependencyMap SyncSchemeManagerArcGaudi3::getDependenciesOnControlEdges(const NodePtr& node) const
{
    if (shouldBlockOnControlEdges(node, *m_graph))
    {
        return nodeSetToDepMap(m_graph->getBlockingNodes(node)); // all dependencies
    }
    else
    {
        // As a minimum we have to add SYNC dependencies to avoid cache trashing
        return nodeSetToDepMap(m_graph->getBlockingNodes(node, Tensor::ControlEdgeType::SYNC));
    }
}

void SyncSchemeManagerArcGaudi3::createRoiPipelineSyncs(const NodePtr&             node,
                                                        const std::list<NodeROI*>& rois,
                                                        const DependencyMap&       inputDependencies,
                                                        DependencyMap&             outputDependencies,
                                                        unsigned&                  emittedSigVal)
{
    OverlapDescriptor desc;

    desc.numSignals = 0;
    desc.engineID   = getLogicalId(node);

    // Transpose and GEMM share the same input port of the MME engine; thus, the overlap shall use
    // the same dependency context for both logical IDs, so in both cases we use the MME's context.
    desc.engineIDForDepCtx =
        desc.engineID == gaudi3::DEVICE_XPS_LOGICAL_QUEUE ? gaudi3::DEVICE_MME_LOGICAL_QUEUE : desc.engineID;

    for (auto roi : rois)
    {
        desc.numSignals += roi->numSignals;
        generateOverlapRois(roi->inputRois, desc.inputRois);
        generateOverlapRois(roi->outputRois, desc.outputRois);
    }

    gaudi3::Overlap::DependencyCtx overlapDep = {0};

    // Fill in input dependencies
    for (const auto& inDep : inputDependencies) // inDep.first is logical ID, and inDep.second is signal value
    {
        overlapDep.valid[inDep.first] = true;
        overlapDep.signalIdx[inDep.first] = convertSigValToOverlapIdx(inDep.first, inDep.second);
    }

    // Run the overlap
    m_pOverlap->addDescriptor(desc, overlapDep, getMaxOverlapSigIdxForNodeToDependOn(node));

    m_emittedSigVal[desc.engineID] += desc.numSignals;  // accumulate emitted signal
    emittedSigVal = m_emittedSigVal[desc.engineID];     // set the output
    validateSigVal(desc.engineID, emittedSigVal);

    for (unsigned logicalId = 0; logicalId < gaudi3::LOGICAL_QUEUE_MAX_ID; logicalId++)
    {
        if (overlapDep.valid[logicalId])
        {
            // convert dependencies from overlap's 0-based index-realm to 1-based signal-realm and set the output
            outputDependencies[logicalId] = overlapDep.signalIdx[logicalId] + 1;
            validateSigVal(logicalId, outputDependencies[logicalId]);
        }
    }
}

void SyncSchemeManagerArcGaudi3::validateSigVal(unsigned logicalId, unsigned sigVal) const
{
    if (sigVal > GCFG_ARC_SYNC_SCHEME_SIGNAL_LIMIT.value())
    {
        LOG_ERR(SYNC_SCHEME, "Signal value {} exceeds the maximum (logical engine={})", sigVal, logicalIdToStr(logicalId));
        throw PassFailedException();
    }
}

unsigned SyncSchemeManagerArcGaudi3::convertSigValToOverlapIdx(unsigned logicalId, unsigned sigVal) const
{
    HB_ASSERT(sigVal > 0, "unexpected signal value 0");
    return sigVal - 1;  // convert from 1-based signal-realm to overlap's 0-based index-realm
}

std::string SyncSchemeManagerArcGaudi3::logicalIdToStr(unsigned logicalId) const
{
    if (logicalId == gaudi3::DEVICE_TPC_LOGICAL_QUEUE)
    {
        return std::string("TPC");
    }
    else if (logicalId == gaudi3::DEVICE_MME_LOGICAL_QUEUE)
    {
        return std::string("MME");
    }
    else if (logicalId == gaudi3::DEVICE_ROT_LOGICAL_QUEUE)
    {
        return std::string("ROT");
    }
    else if (logicalId == gaudi3::DEVICE_XPS_LOGICAL_QUEUE)
    {
        return std::string("TRANSPOSE");
    }
    else
    {
        HB_ASSERT(0, "Unsupported logical ID");
        return std::string("Unknown");
    }
}

std::string SyncSchemeManagerArcGaudi3::sigValToStr(unsigned logicalId, Settable<unsigned> sigVal) const
{
    return sigVal.is_set() ? std::to_string(sigVal.value()) : std::string("none");
}
