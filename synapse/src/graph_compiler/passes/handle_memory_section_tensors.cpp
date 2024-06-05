#include "passes.h"
#include "habana_graph.h"
#include "handle_memory_reuse.h"
#include "register_memory_coherence.h"

namespace
{
struct TensorComparatorBySectionOffset
{
    bool operator()(const TensorPtr& t1, const TensorPtr& t2) const
    {
        return std::make_pair(t1->getMemorySectionOffset(), t1->getId()) <
               std::make_pair(t2->getMemorySectionOffset(), t2->getId());
    }
};
}  // namespace

using TensorOrderedSet     = std::set<TensorPtr, TensorComparatorBySectionOffset>;
using SectionDescriptorMap = std::unordered_map<uint64_t, TensorOrderedSet>;

class MemorySectionHandler
{
public:
    enum eSectionType
    {
        ePersistentSection,
        eWorkspaceSection,
        eInvalid
    };

    MemorySectionHandler(const HabanaGraph& g, uint32_t logLevelOnError, bool useRealProducersConsumers)
    : m_g(g), m_useRealProducersConsumers(useRealProducersConsumers), m_logLevelOnError(logLevelOnError) {};

    bool runSpecificMemorySectionTensorsValidation(eSectionType type) const;
    static bool validateMemoryCoherence(const HabanaGraph& graph);

private:
    static uint64_t getSectionId(const pTensor& tensor);
    static uint64_t getSectionOffset(const pTensor& tensor);
    NodeSet         getTensorProducers(const TensorPtr& t) const;
    NodeSet         getTensorConsumers(const TensorPtr& t) const;
    void            buildSectionMap(SectionDescriptorMap& sectionMap, eSectionType type) const;
    bool            checkProducerProducerNodesOrder(const TensorPtr& tensor1, const TensorPtr& tensor2) const;
    bool            checkProducerConsumerNodesOrder(const TensorPtr& tensor1, const TensorPtr& tensor2) const;
    bool checkNodesOrder(const NodePtr& n1, const NodePtr& n2, const TensorPtr& t1, const TensorPtr& t2) const;
    bool            checkSectionTensors(const TensorOrderedSet& tensorSet) const;
    bool
    checkActualOverlap(const NodePtr& n1, const NodePtr& n2, const TensorPtr& t1Real, const TensorPtr& t2Real) const;
    void            printSectionMap(const SectionDescriptorMap& sectionMap) const;

    const HabanaGraph& m_g;
    bool               m_useRealProducersConsumers;
    uint32_t           m_logLevelOnError;
};

NodeSet MemorySectionHandler::getTensorProducers(const TensorPtr& t) const
{
    if (m_useRealProducersConsumers)
    {
        return m_g.getRealProducers(t);
    }
    else
    {
        return NodeSet({m_g.getTensorProducer(t)});
    }
}

NodeSet MemorySectionHandler::getTensorConsumers(const TensorPtr& t) const
{
    if (m_useRealProducersConsumers)
    {
        return m_g.getRealConsumers(t);
    }
    else
    {
        const NodeList& consumers = m_g.getTensorConsumers(t);
        return NodeSet(consumers.begin(), consumers.end());
    }
}

uint64_t MemorySectionHandler::getSectionId(const pTensor& tensor)
{
    const pTensor& realTensor = Tensor::getRealTensor(tensor);
    if (realTensor->isPartOfWorkspaceSection())
    {
        return tensor->getTensorAnnotation().nonPersistentSectionInfo.sectionId.value();
    }
    else if (tensor->isPersistent())
    {
        return realTensor->getMemorySectionID();
    }
    else
    {
        HB_ASSERT(false, "get section ID for invalid tensor");
        return 0;
    }
}

uint64_t MemorySectionHandler::getSectionOffset(const pTensor& tensor)
{
    const pTensor& realTensor = Tensor::getRealTensor(tensor);
    if (realTensor->isPartOfWorkspaceSection())
    {
        return realTensor->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.value();
    }
    else if (realTensor->isPersistent())
    {
        return realTensor->getMemorySectionOffset();
    }
    else
    {
        HB_ASSERT(false, "get section ID for invalid tensor");
        return 0;
    }
}

void MemorySectionHandler::buildSectionMap(SectionDescriptorMap& sectionMap, eSectionType type) const
{
    for (const auto& tensor : m_g.getTensors())
    {
        if (tensor->getTotalSizeInBytes() == 0) continue;  // ZST tensor

        bool relevantTensor = false;
        switch (type)
        {
            case eWorkspaceSection:
                relevantTensor = tensor->isPartOfWorkspaceSection();
                break;
            case ePersistentSection:
                relevantTensor = tensor->isPersistent() &&
                                 // For unit tests which do not set section Id for persistent tensors.
                                 tensor->getMemorySectionID() != MEMORY_ID_RESERVED_FOR_WORKSPACE;
                break;
            default:
                relevantTensor = false;
        }
        if (relevantTensor)
        {
            uint64_t sectionId = getSectionId(tensor);
            sectionMap[sectionId].insert(tensor);
        }
    }
}

/*
    in some cases, 2 nodes read/write into the same real tensor, but access non-overlapping parts of it.
    if that is the case, we do not want to fail compilation.
*/
bool MemorySectionHandler::checkActualOverlap(const NodePtr&   n1,
                                              const NodePtr&   n2,
                                              const TensorPtr& t1Real,
                                              const TensorPtr& t2Real) const
{
    for (const TensorPtr& t1 : n1->getOperands())
    {
        // find all operands that are mapped to real tensor
        if (!t1 || Tensor::getRealTensor(t1) != t1Real) continue;
        for (const TensorPtr& t2 : n2->getOperands())
        {
            if (!t2 || Tensor::getRealTensor(t2) != t2Real) continue;

            // check if they are actually overlapping (dense overlap)
            if (MemoryReuseHandler::isDenseOverlap(t1, t2)) return true;
        }
    }
    return false;  // the 2 nodes don't actually overlap
}

bool MemorySectionHandler::checkNodesOrder(const NodePtr&   n1,
                                           const NodePtr&   n2,
                                           const TensorPtr& t1,
                                           const TensorPtr& t2) const
{
    if (n1 && n2 && m_g.getNumberOfPaths(n1, n2, Node::TENSOR_TYPE_ALL) == 0 &&
        m_g.getNumberOfPaths(n2, n1, Node::TENSOR_TYPE_ALL) == 0 && !n1->getNodeAnnotation().bundleInfo.is_set() &&
        !n2->getNodeAnnotation().bundleInfo.is_set() && checkActualOverlap(n1, n2, t1, t2))
    {
        SYN_LOG(synapse::LogManager::LogType::GC,
                m_logLevelOnError,
                "Tensors {} (id={}) and {} (id={}) share memory but no explicit order between nodes {} (id={}) "
                "and {} "
                "(id={})",
                t1->getName(),
                t1->getId(),
                t2->getName(),
                t2->getId(),
                n1->getNodeName(),
                n1->getId(),
                n2->getNodeName(),
                n2->getId());
        return false;
    }
    return true;
}

bool MemorySectionHandler::checkProducerProducerNodesOrder(const TensorPtr& tensor1, const TensorPtr& tensor2) const
{
    bool ret = true;
    for (const NodePtr& node1 : getTensorProducers(tensor1))
    {
        for (const NodePtr& node2 : getTensorProducers(tensor2))
        {
            ret &= checkNodesOrder(node1, node2, tensor1, tensor2);
        }
    }
    return ret;
}

bool MemorySectionHandler::checkProducerConsumerNodesOrder(const TensorPtr& tensor1, const TensorPtr& tensor2) const
{
    bool            ret    = true;
    for (const NodePtr& node1 : getTensorProducers(tensor1))
    {
        for (const NodePtr& node2 : getTensorConsumers(tensor2))
        {
            ret &= checkNodesOrder(node1, node2, tensor1, tensor2);
        }
    }
    return ret;
}

// Tensors can't be ZST since we skip ZST in buildSectionMap, hence the validation should pass
bool MemorySectionHandler::checkSectionTensors(const TensorOrderedSet& tensorSet) const
{
    bool ret = true;
    for (auto it1 = tensorSet.begin(); it1 != tensorSet.end(); ++it1)
    {
        const TensorPtr& t1     = *it1;
        const auto       start1 = getSectionOffset(t1);
        const auto       size1  = t1->getTotalSizeInBytes();
        HB_DEBUG_VALIDATE(size1 != 0 && "Unexpected ZST");
        for (auto it2 = std::next(it1); it2 != tensorSet.end(); ++it2)
        {
            const TensorPtr& t2     = *it2;
            const auto       start2 = getSectionOffset(t2);
            // Tensors are sorted based on the starting points so that start2 >= start1.
            // If t2 comes after the end of t1, all future tensors will be >= t2 and will also come after it.
            if (start2 >= start1 + size1) break;
            // In order to avoid short circuit and get nicer logs in the rare case of failures, we split to 3 checks.
            ret &= checkProducerConsumerNodesOrder(t1, t2);
            ret &= checkProducerConsumerNodesOrder(t2, t1);
            ret &= checkProducerProducerNodesOrder(t1, t2);
        }
    }
    return ret;
}

void MemorySectionHandler::printSectionMap(const SectionDescriptorMap& sectionMap) const
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(VALIDATION))
    {
        return;
    }

    for (const auto& section : sectionMap)
    {
        LOG_TRACE(VALIDATION, "Section {}", section.first);
        for (const auto& tensor : section.second)
        {
            LOG_TRACE(VALIDATION,
                      "    Tensor {} (id={}): 0x{} - 0x{}",
                      tensor->getName(),
                      tensor->getId(),
                      getSectionOffset(tensor),
                      getSectionOffset(tensor) + tensor->getTotalSizeInBytes());
            LOG_TRACE(VALIDATION, "        Producer nodes:");
            for (const NodePtr& producerNode : getTensorProducers(tensor))
            {
                if (producerNode)
                {
                    LOG_TRACE(VALIDATION, "            {} (id={})", producerNode->getNodeName(), producerNode->getId());
                }
            }
            LOG_TRACE(VALIDATION, "        Consumer nodes:");
            for (const NodePtr& node : getTensorConsumers(tensor))
            {
                if (node)
                {
                    LOG_TRACE(VALIDATION, "            {} (id={})", node->getNodeName(), node->getId());
                }
            }
        }
    }
}
bool MemorySectionHandler::runSpecificMemorySectionTensorsValidation(eSectionType type) const
{
    SectionDescriptorMap sectionMap;
    buildSectionMap(sectionMap, type);
    printSectionMap(sectionMap);
    bool ret = true;
    for (const auto& section : sectionMap)
    {
        if (section.second.size() > 1)
        {
            ret &= checkSectionTensors(section.second);
        }
    }
    return ret;
}

bool MemorySectionHandler::validateMemoryCoherence(const HabanaGraph& graph)
{
    HB_ASSERT_PTR(graph.getGraphAnnotation().memoryCoherence);  // created at the beginning of the compilation
    const auto& memoryCoherence = graph.getGraphAnnotation().memoryCoherence;
    return memoryCoherence->validatePostGraphMemoryCoherence();
}

static bool runMemorySectionTensorsValidation(const HabanaGraph& g, uint32_t logLevelOnError, bool useRealProducersConsumers)
{
    MemorySectionHandler handler(g, logLevelOnError, useRealProducersConsumers);
    bool ret = handler.runSpecificMemorySectionTensorsValidation(MemorySectionHandler::ePersistentSection);
    ret &= handler.runSpecificMemorySectionTensorsValidation(MemorySectionHandler::eWorkspaceSection);
    return ret;
}

// Pass to validate pre graph memory sections
bool validateUserMemorySections(HabanaGraph& g)
{
    if (GCFG_VALIDATE_MEMORY_SECTION_TENSORS.value())
    {
        return runMemorySectionTensorsValidation(g, 4, false);
    }
    return true;
}

// Pass to validate post graph memory sections
bool validateMemorySectionTensors(HabanaGraph& g)
{
    if (GCFG_VALIDATE_MEMORY_SECTION_TENSORS.value())
    {
        return runMemorySectionTensorsValidation(g, 4, true) && MemorySectionHandler::validateMemoryCoherence(g);
    }
    return true;
}
