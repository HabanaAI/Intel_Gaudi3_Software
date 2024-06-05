
#include "bundle_memory_directive_executor.h"

using namespace gc::layered_brain;

POCBundleMemoryDirectiveExecutor::POCBundleMemoryDirectiveExecutor(HabanaGraph& graph, const MemoryUsageDB& db)
: m_graph(graph), m_db(db)
{
}

bool POCBundleMemoryDirectiveExecutor::executeDirectivesFor(TensorPtr slice)
{
    if (m_db.slices.at(slice).directives.placement == Placement::SRAM)
    {
        if (m_db.slices.at(slice).properties.joinedBy)
        {
            // slice is an input to concat/tensor-view so have to insert memcpy
            addSpillFor(slice);
        }
        if (m_db.slices.at(slice).properties.forkedBy)
        {
            // slice is an output of a split/tensor-view so have to insert memcpy
            addFillFor(slice);
        }
        LOG_DEBUG(LB_CACHE_MNGR, "Executing SRAM placement for slice: {}", slice->getName());
        slice->setTensorInSram();
    }
    else if (m_db.slices.at(slice).directives.placement == Placement::HBM)
    {
        slice->setTensorInDram();
    }
    return true;
}

void POCBundleMemoryDirectiveExecutor::addSpillFor(TensorPtr slice)
{
    LOG_DEBUG(LB_CACHE_MNGR, "Adding spill node for slice {}", slice->getName());
    NodePtr spillNode = insertSpillNode(slice);
    scheduleSpill(spillNode, slice);
    moveAliasing(slice, spillNode->getOutput(0));
}

void POCBundleMemoryDirectiveExecutor::addFillFor(TensorPtr slice)
{
    LOG_DEBUG(LB_CACHE_MNGR, "Adding fill node for slice {}", slice->getName());
    NodePtr fillNode = insertFillNode(slice);
    scheduleFill(fillNode, slice);
    moveAliasing(slice, fillNode->getInput(0));
}

NodePtr POCBundleMemoryDirectiveExecutor::insertSpillNode(TensorPtr slice)
{
    NodePtr spill = GraphEditor::insertMemcpyForInput(m_graph, m_db.slices.at(slice).properties.joinedBy, slice);
    HB_ASSERT(spill, "Unable to insert memcpy for slice {}", slice->getName());
    spill->setName(fmt::format("{}/spill_{:x}/bundle_{}",
                               slice->getName(),
                               spill->getId(),
                               spill->getNodeAnnotation().bundleInfo->bundleIndex));
    return spill;
}

NodePtr POCBundleMemoryDirectiveExecutor::insertFillNode(TensorPtr slice)
{
    NodePtr fill = GraphEditor::insertMemcpyForOutput(m_graph, m_db.slices.at(slice).properties.forkedBy, slice);
    HB_ASSERT(fill, "Unable to insert memcpy for slice {}", slice->getName());
    fill->setName(fmt::format("{}/fill_{:x}/bundle_{}",
                              slice->getName(),
                              fill->getId(),
                              fill->getNodeAnnotation().bundleInfo->bundleIndex));
    return fill;
}

void POCBundleMemoryDirectiveExecutor::scheduleSpill(NodePtr spillNode, TensorPtr slice)
{
    int sliceProducerStep = m_db.slices.at(slice).properties.producingStep.value();
    scheduleWith(spillNode, sliceProducerStep);  // spill needs to be scheduled right after the slice producer
}

void POCBundleMemoryDirectiveExecutor::scheduleFill(NodePtr fillNode, TensorPtr slice)
{
    int sliceFirstConsumerStep = *std::min_element(m_db.slices.at(slice).properties.consumingSteps.begin(),
                                                   m_db.slices.at(slice).properties.consumingSteps.end());
    scheduleWith(fillNode, sliceFirstConsumerStep);  // Fill needs to be scheduled right before the first consumer
}

// Make sure 'n' will be scheduled right before/after the bundle node in step 'step', by giving them the same opIdx.
// when 2 operations in the bundle have the same index, they are scheduled together and the order between them is
// determined by their data dependency, if there is such. In the case of spill/fill, there is always data dependency, so
// this is well defined.
void POCBundleMemoryDirectiveExecutor::scheduleWith(NodePtr n, int step)
{
    const auto& stepNode              = m_db.steps[step].sliceNode;
    n->getNodeAnnotation().bundleInfo = stepNode->getNodeAnnotation().bundleInfo;
}

void POCBundleMemoryDirectiveExecutor::moveAliasing(TensorPtr from, TensorPtr to)
{
    to->cloneAliasInfo(from);
    from->resetAliasing();
}
