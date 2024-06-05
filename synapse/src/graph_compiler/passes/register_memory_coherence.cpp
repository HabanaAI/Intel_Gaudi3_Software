#include "register_memory_coherence.h"
#include "habana_graph.h"
#include "handle_memory_reuse.h"

static bool areConnected(const HabanaGraph& g, const NodePtr& n1, const NodePtr& n2)
{
    return g.getNumberOfPaths(n1, n2, Node::eTensorType::TENSOR_TYPE_ALL) > 0;
}

TensorCoherenceMapping::TensorCoherenceMapping(const HabanaGraph& g)
: m_mappedTensors {TensorCoherence(
                       g.getCodeGenerator()->getNumberOfMemorySections(SectionIDGenerator::USER_ALLOCATED_SECTIONS)),
                   TensorCoherence(
                       g.getCodeGenerator()->getNumberOfMemorySections(SectionIDGenerator::GC_ALLOCATED_SECTIONS))},
  m_g(g)
{
    // create coherence data - list of tensors for every section, according to their creating order
    TensorSet seenTensors;
    for (const NodePtr& n : m_g.getTopoSortedNodes())
    {
        for (const TensorPtr& t : n->getInputs())
        {
            if (skipTensor(t) || seenTensors.find(t) != seenTensors.end()) continue;
            getTensorCoherence(t)[getTensorSectionId(t)].push_back(t);
            seenTensors.insert(t);
        }
        for (const TensorPtr& t : n->getOutputs())
        {
            if (skipTensor(t)) continue;
            getTensorCoherence(t)[getTensorSectionId(t)].push_back(t);
            seenTensors.insert(t);
        }
    }
}

TensorCoherenceMapping::SectionType TensorCoherenceMapping::getTensorCoherencyType(const TensorPtr& t)
{
    if (t->isPersistent())
    {
        return SectionType::USER_ALLOCATED_SECTIONS;
    }
    else if (isMemoryCoherencyTensor(t))
    {
        return SectionType::GC_ALLOCATED_SECTIONS;
    }

    return SectionType::NOF_ALLOCATION_TYPES;
}

bool TensorCoherenceMapping::skipTensor(const TensorPtr& t) const
{
    if (t == nullptr) return true;
    if (!isMemoryCoherencyTensor(t)) return true;  // is not part of any pre-defined memory section

    return false;
}

bool TensorCoherenceMapping::overlapsWithOthersInSection(const TensorPtr& t) const
{
    if (t == nullptr) return false;
    if (!isMemoryCoherencyTensor(t)) return false;  // is not part of any pre-defined memory section
    if (findPreviousCoherencyTensors(t).empty() && findNextCoherencyTensors(t).empty()) return false;

    return true;
}

bool TensorCoherenceMapping::previousOverlapsWithOthersInSection(const TensorPtr& t) const
{
    if (t == nullptr) return false;
    if (!isMemoryCoherencyTensor(t)) return false;  // is not part of any pre-defined memory section
    if (findPreviousCoherencyTensors(t).empty()) return false;

    return true;
}

const TensorCoherenceMapping::TensorCoherence& TensorCoherenceMapping::getTensorCoherence(const TensorPtr& t) const
{
    HB_ASSERT(isMemoryCoherencyTensor(t), "tensor {} is not reserved for memory section", t->getName());
    return m_mappedTensors[getTensorCoherencyType(t)];
}

TensorCoherenceMapping::TensorCoherence& TensorCoherenceMapping::getTensorCoherence(const TensorPtr& t)
{
    return const_cast<TensorCoherence&>(static_cast<const TensorCoherenceMapping&>(*this).getTensorCoherence(t));
}

uint64_t TensorCoherenceMapping::getTensorSectionId(const TensorPtr& t) const
{
    HB_ASSERT(isMemoryCoherencyTensor(t), "tensor {} is not reserved for memory section", t->getName());
    return t->isPersistent() ? t->getMemorySectionID()
                             : t->getTensorAnnotation().nonPersistentSectionInfo.sectionId.value();
}

TensorVector TensorCoherenceMapping::findNextCoherencyTensors(const TensorPtr& t) const
{
    return findCoherencyTensors(t, false);
}

TensorVector TensorCoherenceMapping::findPreviousCoherencyTensors(const TensorPtr& t) const
{
    return findCoherencyTensors(t, true);
}

bool TensorCoherenceMapping::doesCoherencyTensorExist(const TensorPtr& t) const
{
    if (!isMemoryCoherencyTensor(t)) return false;

    const auto& allSectionTensors = getTensorCoherence(t);
    auto        sectionIt         = allSectionTensors.find(getTensorSectionId(t));
    if (sectionIt == allSectionTensors.end()) return false;  // section doesn't exist

    const TensorVector& sectionTensors = sectionIt->second;
    auto                it             = std::find(sectionTensors.begin(), sectionTensors.end(), t);
    if (it == sectionTensors.end()) return false;  // tensor doesn't exist

    return true;
}

TensorVector TensorCoherenceMapping::findCoherencyTensors(const TensorPtr& t, bool previousTensors) const
{
    /*
        find all tensors that overlap with t, and are produced before/after it.
    */
    TensorVector ret;

    if (!doesCoherencyTensorExist(t))
    {
        LOG_DEBUG(MEM_COHERENCE, "{} is not a memory coherency tensor!", t->getName());
        return ret;
    }

    const auto& allSectionTensors = getTensorCoherence(t);
    uint64_t    sectionId         = getTensorSectionId(t);

    auto sectionIt = allSectionTensors.find(sectionId);

    HB_ASSERT(sectionIt != allSectionTensors.end(),
              "persistent tensor {} resides in illegal memory section",
              t->getName());

    const TensorVector& sectionTensors = sectionIt->second;
    auto                it             = std::find(sectionTensors.begin(), sectionTensors.end(), t);
    HB_ASSERT(it != sectionTensors.end(), "tensor {} not found in coherency list", t->getName());

    // search for overlapping tensors before/after 't'
    auto startIt = previousTensors ? sectionTensors.begin() : std::next(it);
    auto endIt   = previousTensors ? it : sectionTensors.end();
    for (auto otherIt = startIt; otherIt != endIt; otherIt++)
    {
        // skip tensors that are not part of the graph anymore.
        if (m_g.getNumberOfTensorConsumers(*otherIt) == 0 && m_g.getNumberOfTensorProducers(*otherIt) == 0) continue;
        if (!MemoryReuseHandler::isDenseOverlap(t, *otherIt)) continue;
        ret.push_back(*otherIt);
    }
    return ret;
}

NodeSet TensorCoherenceMapping::calculateBlockedNodes(const HabanaGraph& g, const NodePtr& blockingNode) const
{
    /*
        find all nodes that are blocked by 'blockingNode'. there are 3 types of memory dependencies:
        1. WAR - write after read; producers of an operand that is produced after an input of 'blockingNode'
        2. WAW - write after write; producers of an operand that is produced after an output of 'blockingNode'
        3. RAW - read after write; consumers of an operand that is produced after an output of 'blockingNode'.
            for legacy purposes, we currently support RAW for tensors that don't have consumers/producers.
            i.e.: (Producer) -> [t1];    [t2] -> (Consumer).  where [t1] and [t2] are overlapping, [t1] has no consumers
       and [t2] has no producer
    */
    NodeSet blockedNodes;

    for (const TensorPtr& output : blockingNode->getOutputs())
    {
        // only persistent tensors can have the same address for different tensors
        if (!output || !isMemoryCoherencyTensor(output)) continue;

        /* no need to block anyone, the consumers of 'output' will block producers of overlapping tensors. example:
            (prod1) -> [t_output] -> (cons)
            (prod2) -> [t_other]               where t_other overlaps with t_output, and is produced later.

            in this case, there is no need for a WAW control edge between (prod1) -> (prod2), since we will get anyway
             the control edge (cons) -> (prod2) that will enforce both dependencies (WAW and WAR) */
        if (g.getNumberOfTensorConsumers(output) != 0) continue;

        // for each next coherency tensor (operand produced after 'output')
        for (const TensorPtr& other : findNextCoherencyTensors(output))
        {
            const NodePtr& producer = g.getTensorProducer(other);
            if (producer)  // write after write (2)
            {
                if (producer == blockingNode) continue;  // happens in-place ops
                blockedNodes.insert(producer);
            }
            else  // read after write (3). if no producer, need to block all the consumers
            {
                for (const NodePtr& consumer : g.getTensorConsumers(other))
                {
                    HB_ASSERT(consumer != blockingNode,
                              "detected cycle due to memory coherence. node {}",
                              consumer->getNodeName());
                    blockedNodes.insert(consumer);
                }
            }
        }
    }
    for (const TensorPtr& input : blockingNode->getInputs())
    {
        // only persistent tensors can have the same address for different tensors
        if (!input || !isMemoryCoherencyTensor(input)) continue;

        // for each next coherency tensor (operand produced after 'input')
        for (const TensorPtr& other : findNextCoherencyTensors(input))
        {
            const NodePtr& producer = g.getTensorProducer(other);
            if (producer)  // write after read (1)
            {
                if (producer == blockingNode) continue;  // in-place operation
                blockedNodes.insert(producer);
            }
        }
    }
    return blockedNodes;
}

NodeSet TensorCoherenceMapping::calculateBlockingNodes(const HabanaGraph& g, const NodePtr& blockedNode) const
{
    /*
        find all nodes that are blocking 'blockedNode'. there are 3 types of memory dependencies:
        1. WAR - write after read; consumers of an operand that is produced before an output of 'blockedNode'
        2. WAW - write after write; producers of an operand that is produced before an output of 'blockedNode'
        3. RAW - read after write; producers of an operand that is produced before an input of 'blockingNode'.
            for legacy purposes, we currently support RAW for tensors that don't have consumers/producers.
            i.e.: (Producer) -> [t1];    [t2] -> (Consumer).  where [t1] and [t2] are overlapping, [t1] has no consumers
       and [t2] has no producer
    */
    NodeSet blockingNodes;

    for (const TensorPtr& output : blockedNode->getOutputs())
    {
        // only persistent tensors can have the same address for different tensors
        if (!output || !isMemoryCoherencyTensor(output)) continue;

        // for each previous coherency tensor (operand produced before 'output')
        for (const TensorPtr& other : findPreviousCoherencyTensors(output))
        {
            const auto& consumers = g.getTensorConsumers(other);
            if (!consumers.empty())
            {
                for (const NodePtr& consumer : consumers)  // write after read (1)
                {
                    if (consumer == blockedNode) continue;  // in-place operation
                    blockingNodes.insert(consumer);
                }
            }
            else  // write after write (2), if there are consumers of 'other' then they will block writing 'output'
            {
                const NodePtr& producer = g.getTensorProducer(other);
                if (!producer) continue;
                if (producer == blockedNode)
                {
                    LOG_WARN(GC, "node {} has 2 overlapping persistent outputs", producer->getNodeName());
                    continue;
                }
                blockingNodes.insert(producer);
            }
        }
    }
    for (const TensorPtr& input : blockedNode->getInputs())
    {
        // only persistent tensors can have the same address for different tensors
        if (!input || !isMemoryCoherencyTensor(input)) continue;

        // if 'input' has a producer then there is no external RAW dependency
        if (g.getTensorProducer(input) != nullptr) continue;
        // for each previous coherency tensor (operand produced before 'input')
        for (const TensorPtr& other : findPreviousCoherencyTensors(input))
        {
            const NodePtr& producer = g.getTensorProducer(other);
            if (producer == nullptr) continue;
            HB_ASSERT(producer != blockedNode,
                      "detected cycle due to memory coherence. node {}",
                      producer->getNodeName());
            blockingNodes.insert(producer);  // read after write external dependency
        }
    }
    return blockingNodes;
}

bool TensorCoherenceMapping::doReadAfterWriteExternalDependenciesExist() const
{
    for (const TensorCoherence& coherenceList : getAllSectionsTensorCoherence())
    {
        for (const auto& sectionIdAndTensors : coherenceList)
        {
            const TensorVector& sectionTensors = sectionIdAndTensors.second;
            for (auto firstIt = sectionTensors.begin(); firstIt != sectionTensors.end(); firstIt++)
            {
                for (auto secondIt = std::next(firstIt); secondIt != sectionTensors.end(); secondIt++)
                {
                    const TensorPtr& blockingTensor = *firstIt;
                    const TensorPtr& blockedTensor  = *secondIt;

                    // not a RAW external dependency - tensor has an explicit producer
                    if (m_g.getTensorProducer(blockedTensor) != nullptr) continue;

                    if (!MemoryReuseHandler::isDenseOverlap(blockingTensor, blockedTensor)) continue;
                    const NodePtr& producer = m_g.getTensorProducer(blockingTensor);
                    if (!producer) continue;
                    for (const NodePtr& consumer : m_g.getTensorConsumers(blockedTensor))
                    {
                        if (producer != consumer)
                        {
                            LOG_WARN(MEM_COHERENCE,
                                     "found RAW external dependency {} --> {} with tensors {}, {}",
                                     producer->getNodeName(),
                                     consumer->getNodeName(),
                                     blockingTensor->getName(),
                                     blockedTensor->getName());
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

void TensorCoherenceMapping::printMemoryCoherence() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(MEM_COHERENCE)) return;
    LOG_DEBUG(MEM_COHERENCE, "memory coherence for user memory section tensors:");
    printMemoryCoherence(m_mappedTensors[SectionType::USER_ALLOCATED_SECTIONS]);
    LOG_DEBUG(MEM_COHERENCE, "memory coherence for gc memory section tensors:");
    printMemoryCoherence(m_mappedTensors[SectionType::GC_ALLOCATED_SECTIONS]);
}

void TensorCoherenceMapping::printMemoryCoherence(const TensorCoherence& allSectionsTensors) const
{
    for (auto it : allSectionsTensors)
    {
        const TensorVector& sectionTensors = it.second;
        if (sectionTensors.empty()) continue;
        unsigned section = getTensorSectionId(sectionTensors.front());
        LOG_DEBUG(MEM_COHERENCE, "memory coherence for section {}:", section);
        for (const TensorPtr& t : sectionTensors)
        {
            LOG_DEBUG(MEM_COHERENCE, "\t{}", t->getName());
        }
    }
}

void TensorCoherenceMapping::validateMemoryCoherence() const
{
    for (const TensorCoherence& coherenceList : m_mappedTensors)
    {
        validateMemoryCoherence(coherenceList);
    }
}

void TensorCoherenceMapping::validateMemoryCoherence(const TensorCoherence& allSectionsTensors) const
{
    for (auto it : allSectionsTensors)
    {
        const TensorVector& sectionTensors = it.second;
        for (auto firstIt = sectionTensors.begin(); firstIt != sectionTensors.end(); firstIt++)
        {
            for (auto secondIt = std::next(firstIt); secondIt != sectionTensors.end(); secondIt++)
            {
                if (!MemoryReuseHandler::isDenseOverlap(*firstIt, *secondIt)) continue;
                const NodePtr& producer = m_g.getTensorProducer(*secondIt);
                if (!producer) continue;
                for (const NodePtr& consumer : m_g.getTensorConsumers(*firstIt))
                {
                    HB_ASSERT(areConnected(m_g, consumer, producer),
                              "node {} and node {} are not connected",
                              consumer->getNodeName(),
                              producer->getNodeName());
                }
            }
        }
    }
}

bool TensorCoherenceMapping::isMemoryCoherencyTensor(const TensorPtr& t, SectionType type)
{
    return getTensorCoherencyType(t) == type;
}

/*
    build a map containing the alias tensors that are mapped to every real tensor
    "real tensor" --> {All alias tensors}
*/
TensorCoherenceMapping::RealTensorMap TensorCoherenceMapping::buildRealTensorMapping(SectionType type) const
{
    RealTensorMap realTensorMap;
    for (const TensorPtr& t : m_g.getTensors())
    {
        if (!t) continue;
        // in some unique cases, a coherency tensor can be aliased. in that case, save 't'
        TensorPtr real = isMemoryCoherencyTensor(t, type) ? t : Tensor::getRealTensor(t);
        if (isMemoryCoherencyTensor(real, type))
        {
            realTensorMap[real].insert(t);
        }
    }
    return realTensorMap;
}

// validate that 2 real nodes that write into the same memory are scheduled in the correct order.
bool TensorCoherenceMapping::validateWriteAfterWrite(const TensorPtr& blocking, const TensorPtr& blocked) const
{
    const NodePtr& blockingProducer = m_g.getTensorProducer(blocking);
    const NodePtr& blockedProducer  = m_g.getTensorProducer(blocked);
    if (!blockingProducer || blockingProducer->isLogicalOperation()) return true;
    if (!blockedProducer || blockedProducer->isLogicalOperation()) return true;
    if (blockingProducer->getExecutionOrderedIndex() >= blockedProducer->getExecutionOrderedIndex())
    {
        LOG_ERR(VALIDATION,
                "missing a path in graph between node {} and {}, due to WAW tensors {} and {}",
                blockingProducer->getNodeName(),
                blockedProducer->getNodeName(),
                blocking->getName(),
                blocked->getName());
        return false;
    }
    return true;
}

// validate that 2 real nodes that read/write from/to the same memory are scheduled in the correct order.
bool TensorCoherenceMapping::validateWriteAfterRead(const TensorPtr& blocking, const TensorPtr& blocked) const
{
    bool           ret             = true;
    const NodePtr& blockedProducer = m_g.getTensorProducer(blocked);
    if (!blockedProducer || blockedProducer->isLogicalOperation()) return ret;
    for (const NodePtr& blockingConsumer : m_g.getTensorConsumers(blocking))
    {
        if (!blockingConsumer || blockingConsumer->isLogicalOperation()) continue;
        // blockingConsumer == blockedProducer is allowed as a "in-place" operation.
        if (blockingConsumer->getExecutionOrderedIndex() > blockedProducer->getExecutionOrderedIndex())
        {
            LOG_ERR(VALIDATION,
                    "missing a path in graph between node {} and {}, due to WAR tensors {} and {}",
                    blockingConsumer->getNodeName(),
                    blockedProducer->getNodeName(),
                    blocking->getName(),
                    blocked->getName());
            ret = false;
        }
    }
    return ret;
}

bool TensorCoherenceMapping::validatePostGraphCoherencyTensors(const RealTensorMap& realTensorMap,
                                                               const TensorPtr&     blockingRealTensor,
                                                               const TensorPtr&     blockedRealTensor) const
{
    // no overlap between blocking and blocked tensor
    if (!MemoryReuseHandler::isDenseOverlap(blockingRealTensor, blockedRealTensor)) return true;

    // if tensor doesn't exist, it probably means that it is a tensor that existed at the beginning of the compilation
    // and doesn't exist anymore. this can happen in various scenarios like gradient buckets that are optimzed out.
    if (realTensorMap.find(blockingRealTensor) == realTensorMap.end())
    {
        LOG_DEBUG(VALIDATION, "blocking real tensor {} not found in map", blockingRealTensor->getName());
        return true;
    }
    if (realTensorMap.find(blockedRealTensor) == realTensorMap.end())
    {
        LOG_DEBUG(VALIDATION, "blocked real tensor {} not found in map", blockedRealTensor->getName());
        return true;
    }

    bool ret = true;
    for (const TensorPtr& blockingTensor : realTensorMap.at(blockingRealTensor))
    {
        for (const TensorPtr& blockedTensor : realTensorMap.at(blockedRealTensor))
        {
            // check if the 2 operands are really overlapping
            if (!MemoryReuseHandler::isStridedOverlap(blockingTensor, blockedTensor)) continue;
            ret &= validateWriteAfterWrite(blockingTensor, blockedTensor);  // WAW dependencies
            ret &= validateWriteAfterRead(blockingTensor, blockedTensor);   // WAR dependencies
        }
    }
    return ret;
}

bool TensorCoherenceMapping::validatePostGraphCoherencyMapping(SectionType          type,
                                                               const RealTensorMap& realTensorMap) const
{
    bool                   ret              = true;
    const TensorCoherence& coherencyMapping = m_mappedTensors[type];
    for (const auto& sectionIdTensors : coherencyMapping)
    {
        // this holds all the real tensors in the current section, by the order of their creation.
        const TensorVector& sectionTensors = sectionIdTensors.second;
        if (sectionTensors.size() < 2) continue;  // no overlapping tensors in this section
        // validate that for each tensor, all tensor that follow it (in respect to creation order) are valid
        for (auto firstIt = sectionTensors.begin(); std::next(firstIt) != sectionTensors.end(); firstIt++)
        {
            for (auto secondIt = std::next(firstIt); secondIt != sectionTensors.end(); secondIt++)
            {
                ret &= validatePostGraphCoherencyTensors(realTensorMap, *firstIt, *secondIt);
            }
        }
    }
    return ret;
}

bool TensorCoherenceMapping::validatePostGraphMemoryCoherence(SectionType type) const
{
    RealTensorMap realTensorMap = buildRealTensorMapping(type);
    return validatePostGraphCoherencyMapping(type, realTensorMap);
}

// for make sure that all readers/writers of overlapping tensors in graph have the necessary dependencies
bool TensorCoherenceMapping::validatePostGraphMemoryCoherence() const
{
    bool ret = true;
    m_g.getExeSortedNodes();  // make sure execution order indices are set [SW-124132]
    ret &= validatePostGraphMemoryCoherence(SectionType::USER_ALLOCATED_SECTIONS);
    ret &= validatePostGraphMemoryCoherence(SectionType::GC_ALLOCATED_SECTIONS);
    return ret;
}

TensorCoherenceMapping::CoherencyComparator::CoherencyComparator(const TensorCoherenceMapping& coherenceMapping)
: m_coherenceMapping(coherenceMapping)
{
}

bool TensorCoherenceMapping::CoherencyComparator::operator()(const TensorPtr& t1, const TensorPtr& t2) const
{
    if (t1 == t2) return false;

    if (!m_coherenceMapping.isMemoryCoherencyTensor(t1)) return false;
    if (!m_coherenceMapping.isMemoryCoherencyTensor(t2)) return true;

    SectionType type1 = m_coherenceMapping.getTensorCoherencyType(t1);
    SectionType type2 = m_coherenceMapping.getTensorCoherencyType(t2);
    if (type1 < type2) return true;
    if (type1 > type2) return false;

    uint64_t section1 = m_coherenceMapping.getTensorSectionId(t1);
    uint64_t section2 = m_coherenceMapping.getTensorSectionId(t2);
    if (section1 < section2) return true;
    if (section1 > section2) return false;

    const TensorVector& coherency = m_coherenceMapping.getTensorCoherence(t1).at(section1);
    for (const TensorPtr& t : coherency)
    {
        if (t == t1) return true;
        if (t == t2) return false;
    }
    HB_ASSERT(0, "tensor {} and {} were not found in coherency mapping", t1->getName(), t2->getName());
    return t1->getId() < t2->getId();
}

/*
    register all section tensors according to the order of appearance.
    this will later be used to enforce read/write of these tensors in the same order
*/
bool registerMemoryCoherence(HabanaGraph& g)
{
    g.getGraphAnnotation().memoryCoherence = TensorCoherencePtr(new TensorCoherenceMapping(g));
    const auto& memoryCoherence            = g.getGraphAnnotation().memoryCoherence;
    memoryCoherence->printMemoryCoherence();

    if (GCFG_VALIDATE_MEMORY_SECTION_TENSORS.value())
    {
        memoryCoherence->validateMemoryCoherence();
    }
    return true;
}