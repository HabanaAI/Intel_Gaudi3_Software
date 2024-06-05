#include "node2desc.h"

// eager includes (relative to src/eager/lib/)
#include "chip_info.h"
#include "desc_gen/desc_base.h"
#include "eager_graph.h"
#include "node_info/eager_node.h"
#include "utils/general_defs.h"

namespace eager_mode
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// SingleNode2Desc
///////////////////////////////////////////////////////////////////////////////////////////////////

bool SingleNode2Desc::init(const EagerNode& node, const EagerNode* latestPhysicalProducer)
{
    EAGER_ASSERT_PTR(node);
    EAGER_ASSERT(m_descGenPtr == nullptr, "Descriptors were generated before");
    m_latestPhysicalProducer = latestPhysicalProducer;

    for (const TensorVector* tensors : {&node->getInputs(), &node->getOutputs()})
    {
        for (const auto& tensor : *tensors)
        {
            if ((tensor != nullptr) && tensor->inSram())
            {
                EAGER_ASSERT(tensor->isAuxTensor(), "No support for SRAM yet");
                // Currently scratch pad will be in workspace
                tensor->setTensorInWorkspace();
            }
        }
    }

    m_descGenPtr = ChipInfo::createDescGenerator(m_graph, node, m_graph.getChipType(), node.getEngineType());
    if (!m_descGenPtr)
    {
        EAGER_LOG_WARN("Failed to generate descriptor for node {}", node->getGUID());
        return false;
    }

    return true;
}

bool SingleNode2Desc::generateDescriptors()
{
    EAGER_ASSERT_PTR(m_descGenPtr);
    if (!m_descGenPtr->generateDesc())
    {
        return false;  // Node is not supported by eager yet - need to fallback to Graph mode
    }

    return true;
}

DescGeneratorBase* SingleNode2Desc::getDescGen(synNodeId id)
{
    EAGER_ASSERT_PTR(m_descGenPtr);
    EAGER_ASSERT(m_descGenPtr->getNode().getId() == id, "Invalid node ID");
    return m_descGenPtr.get();
}

const DescGeneratorBase& SingleNode2Desc::getDescGen() const
{
    EAGER_ASSERT_PTR(m_descGenPtr);
    return *m_descGenPtr;
}

DescGeneratorBase& SingleNode2Desc::getDescGen()
{
    EAGER_ASSERT_PTR(m_descGenPtr);
    return *m_descGenPtr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Node2DescContainer
///////////////////////////////////////////////////////////////////////////////////////////////////

bool Node2DescContainer::init(const EagerNodes& nodes, const VecNodes<NodesNrType>& latestPhysicalProducers)
{
    EAGER_ASSERT(!m_isInitialized, "Repeated init");
    EAGER_ASSERT(nodes.getPhysicalNodesNr() != 0, "Execution sequence is not available");
    EAGER_ASSERT(nodes.size() == latestPhysicalProducers.size(), "Inconsistent nodes information");
    m_execSequence.reserve(nodes.getPhysicalNodesNr());

    for (NodesNrType i = 0; i < nodes.size(); ++i)
    {
        const EagerNode& node = nodes[i];
        if (node->isLogicalOperation()) continue;

        const EagerNode* latestPhysicalProducer =
            (latestPhysicalProducers[i] < nodes.size()) ? &nodes[latestPhysicalProducers[i]] : nullptr;

        m_execSequence.emplace_back(m_graph);
        SingleNode2Desc& eagerNode = m_execSequence.back();
        if (unlikely(!eagerNode.init(node, latestPhysicalProducer)))
        {
            m_stats = {};
            m_execSequence.clear();
            return false;
        }

        // Update statistics
        {
            const auto engineId = static_cast<unsigned>(eagerNode.getDescGen().getEngineType());
            ++m_stats[engineId].nodeNum;
        }
    }

    m_isInitialized = true;
    return m_isInitialized;
}

bool Node2DescContainer::generateDescriptors()
{
    EAGER_ASSERT(!m_execSequence.empty(), "Execution sequence is not available");
    for (SingleNode2Desc& node : m_execSequence)
    {
        if (!node.generateDescriptors())
        {
            return false;
        }

        // Update statistics
        {
            const auto engineId = static_cast<unsigned>(node.getDescGen().getEngineType());
            m_stats[engineId].activationNum += node.getDescGen().getActivationNr();
        }
    }
    return true;
}

}  // namespace eager_mode