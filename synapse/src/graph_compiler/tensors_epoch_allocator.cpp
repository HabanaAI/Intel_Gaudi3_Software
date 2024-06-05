#include "tensors_epoch_allocator.h"

#include "allocators_utils.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "memory_management/heap_allocator.h"

#include "types_exception.h"

class NoMemory : public std::exception
{
};

template<class Container, class Item>
bool contains(const Container& container, const Item& item)
{
    return container.end() != std::find(container.begin(), container.end(), item);
}

TensorsEpochAllocator::TensorsEpochAllocator(HabanaGraph*         graph,
                                             TensorsCompatibility tensorsCompatibility,
                                             uint64_t             maxEpochSize,
                                             const std::string&   name,
                                             std::unique_ptr<NonPersistentSectionAllocTracker> allocTracker)
: m_graph(graph),
  m_tensorsCompatibility(tensorsCompatibility),
  m_livenessAnalysis(new LivenessAnalysis(m_graph, tensorsCompatibility)),
  m_currBundleIndex(std::numeric_limits<uint64_t>::max()),
  m_name(name),
  m_nonPersistentSectionAllocTracker(std::move(allocTracker))
{
    HeapAllocator& alloc  = (HeapAllocator&)m_graph->getCodeGenerator()->getSramAllocator();

    Range maxRange = alloc.getMaxFreeRange();
    maxRange.size = std::min(maxEpochSize, maxRange.size);

    if (!alloc.allocateReqRange(maxRange, 0))
    {
        LOG_ERR(EPOCH_ALLOC, "Failed to allocate requested range of size {}B with base 0x{:x}",
                maxRange.size,
                maxRange.base);

        throw NoMemory();
    }

    auto tmp = createAllocator(MEMORY_HEAP_ALLOCATOR, "SRAM");
    HB_ASSERT_PTR(dynamic_cast<HeapAllocator*>(tmp.get()));
    m_sramAllocator.reset(static_cast<HeapAllocator*>(tmp.release()));

    m_sramAllocator->Init(maxRange.size, maxRange.base);

    if (LOG_LEVEL_AT_LEAST_DEBUG(HEAP_ALLOC))
    {
        m_sramAllocator->SetPrintStatus(true);
    }

    LOG_TRACE(EPOCH_ALLOC, "- {}: maxRange.size {}", HLLOG_FUNC, maxRange.size);
}

TensorsEpochAllocator::~TensorsEpochAllocator() = default;

bool TensorsEpochAllocator::allocateTensorsMemorySpace()
{
    try
    {
        for (const pNode& node : m_graph->getExeSortedNodes())
        {
            handleNodeTensors(node);
        }

        // Allocate leftovers
        handleEpochMemory();
    }
    catch (const NoMemory& e)
    {
        m_graph->getGraphAnnotation().errors.memoryAllocationError = true;
        LOG_ERR(EPOCH_ALLOC, "Allocate SRAM failed!");
        return false;
    }

    if (m_nonPersistentSectionAllocTracker)
    {
        m_nonPersistentSectionAllocTracker->verifyAllDone();
    }

    return true;
}

void TensorsEpochAllocator::allocateTensorInDram(const pTensor& tensor)
{
    if (!m_graph->getGraphAnnotation().memoryStrategyParams.dramInfo.enableDramAlloc)
    {
        LOG_ERR(EPOCH_ALLOC, "- {}: Tensor: {} enableDramAlloc is disabled", HLLOG_FUNC, tensor->getName());
        throw NoMemory();
    }

    if (!isNonPersistentActivationTensor(m_graph, tensor) &&
        !isAllocInDramForced(tensor)) // all non-activation/persistent tensors are allocated.
    {
        if (!::allocateTensorInDram(*m_graph,
                                    tensor,
                                    /*allocateRealTensor*/ true,
                                    /*allowFailure*/ false,
                                    /*alloc*/ nullptr,
                                    m_nonPersistentSectionAllocTracker.get()))
        {
            LOG_ERR(EPOCH_ALLOC, "- {}: Allocate Tensor {} in DRAM failed", HLLOG_FUNC, tensor->getName());
            throw NoMemory();
        }
        LOG_TRACE(EPOCH_ALLOC, "- {}: Allocate Tensor {} in DRAM", HLLOG_FUNC, tensor->getName());
    }
    else // activations allocation is deferred.
    {
        // instead of allocating the tensor in DRAM here, mark it to deffer allocation to the DramTensorsAllocator.
        tensor->getTensorAnnotation().memory.location = TENSOR_IN_DRAM;
        LOG_TRACE(EPOCH_ALLOC, "- {}: Mark Tensor {} for allocation in DRAM", HLLOG_FUNC, tensor->getName());
    }
}

bool TensorsEpochAllocator::handleFallback(bool allowDramAllocation)
{
    LOG_TRACE(EPOCH_ALLOC,
              "{}: Failed to allocate one or more tensors belonging to the last bundle due to fragmentation. "
              "Fallback to the previous bundle, allowDramAllocation = {}.",
              HLLOG_FUNC,
              allowDramAllocation);

    // Free last bundle tensors and unset their addresses
    for (auto const& iter : m_lastBundleTensors)
    {
        const auto& tensor = iter.first;
        if (tensor == nullptr || tensor->isShapeTensor()) continue;
        if (tensor->tensorAllocatedInSram())
        {
            LOG_TRACE(EPOCH_ALLOC,
                      "{}: Fallback - Free allocated tensor {} and unset it's SRAM offset",
                      HLLOG_FUNC,
                      tensor->getName());

            // Free the allocated address and mark the tensor as not allocated yet to be allocated as part of the
            // fallback with no fragmentation.
            const auto& msOff = tensor->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase;
            const auto  offset = tensor->getSramOffset() - (msOff.is_set() ? msOff.value() : 0);
            if (m_sramAllocator->IsAllocated(offset))
            {
                freeTensorFromSram(*m_graph,
                                   tensor,
                                   /*freeRealTensor*/ false,
                                   m_sramAllocator.get(),
                                   /*rollback*/ true,
                                   m_nonPersistentSectionAllocTracker.get());
            }
            else if (m_nonPersistentSectionAllocTracker != nullptr)
            {
                // Dead tensors were already freed without a rollback and require a status change (freed -> unallocated)
                m_nonPersistentSectionAllocTracker->handleUnallocatedTensorFallback(tensor);
            }
            tensor->unsetSramOffset();
        }
        m_handledTensors.remove(tensor);
    }

    if (m_nonPersistentSectionAllocTracker)
    {
        m_nonPersistentSectionAllocTracker->discardPlannedTensors();
    }

    // Start a new epoch
    m_lastNode     = m_lastBundleTensors.begin()->second;
    m_currentEpoch = Epoch(m_lastNode->getNodeAnnotation().sliceIndex);

    pNode curNode = m_lastNode;
    // Reallocate last bundle tensors
    for (auto const& iter : m_lastBundleTensors)
    {
        if (iter.second != curNode)  // moved to a new node, means previous node allocation was finished
        {
            m_lastNode = curNode;  // update m_lastNode to be the previous "finished" node for liveness analysis
        }
        curNode = iter.second;
        if (iter.first != nullptr)
        {
            LOG_TRACE(EPOCH_ALLOC,
                      "{}: call handleNodeTensor for tensor {} while fallback is disabled",
                      HLLOG_FUNC,
                      iter.first->getName());
            const auto& [success, writeSpaceForTensor] =
                handleNodeTensor(iter.second /* node */, iter.first /* tensor */, false /* allowFallback */);
            if (!success)
            {
                if (allowDramAllocation)
                {
                    handleTensorDoesNotFitInSram(iter.first, writeSpaceForTensor);
                }
                else
                {
                    return false;
                }
            }
        }
    }
    m_lastBundleTensors.clear();
    return true;
}

std::pair<bool, uint64_t> TensorsEpochAllocator::handleNodeTensor(const pNode& node, pTensor tensor, bool allowFallback)
{
    GET_REAL_TENSOR_IF_NULL_RETURN_VAL(tensor, std::make_pair(true, 0));
    if (!m_tensorsCompatibility(tensor) || (tensor->tensorIsAllocated()) || contains(m_handledTensors, tensor) ||
        isAllocInDramForced(tensor) || tensor->isPersistent())
    {
        // in fall back mode (=training graph) we use m_lastBundleTensors in order to try to
        // reallocate all bundle tensors from the bundle beginning. For correct estimation of the available capacity, we
        // use liveness analysis on m_lastNode to know which tensors are dead after that node is finished and hence
        // can be released. For that purpose we must have in m_lastBundleTensors all the nodes that appear in the
        // execution schedule after the first node in the bundle - this will guarantee the same behaviour as before
        // reaching handleFallback().
        if (allowFallback && !m_lastBundleTensors.empty() && m_lastBundleTensors.back().second != node)
        {
            m_lastBundleTensors.emplace_back(nullptr, node);
        }
        return {true, 0};
    }

    uint64_t writeSpaceForTensor = getWriteSpaceForTensor(tensor);
    if (m_nonPersistentSectionAllocTracker && m_nonPersistentSectionAllocTracker->trackPlannedNonPersistentSectionTensor(tensor))
    {
        LOG_TRACE(EPOCH_ALLOC,
                  "- {}: {} is reusing non-persistent section {}, no need for additional write space.",
                  HLLOG_FUNC,
                  tensor->getName(),
                  tensor->getTensorAnnotation().nonPersistentSectionInfo.sectionId.value());
        writeSpaceForTensor = 0;
    }

    // The space needed for the tensor might fit in new epoch but not in the current epoch
    // allocate the current epoch and create a new epoch- handleNodeTensor
    if (shouldCreateNewEpoch(node, writeSpaceForTensor))
    {
        handleEpochMemory(m_lastNode);

        // Discard all planned tensors except for the current one since it moves to the next epoch
        if (m_nonPersistentSectionAllocTracker)
        {
            m_nonPersistentSectionAllocTracker->discardPlannedTensors();
            if (m_nonPersistentSectionAllocTracker->trackPlannedNonPersistentSectionTensor(tensor))
            {
                HB_ASSERT(false, "First tensor in epoch cannot belong to a planned non-persistent section");
            }
        }

        LOG_TRACE(EPOCH_ALLOC,
                  "Start a new epoch for tensor {}, Sram size for epoch : {}",
                  tensor->getName(),
                  m_sramAllocator->getMaxFreeContiguous());
        m_currentEpoch = Epoch(node->getNodeAnnotation().sliceIndex);

        // This tensor has write space bigger than free Sram size, and need to send this tensor to DRAM
        if (writeSpaceForTensor > m_sramAllocator->getMaxFreeContiguous())
        {
            return {false, writeSpaceForTensor};
        }
    }

    if (allowFallback)
    {
        // Fallback is allowed - gather all tensors of the last bundle in case they will be needed.
        // Clear those tensors when starting a new bundle.
        if (!node->getNodeAnnotation().bundleInfo.is_set())
        {
            m_lastBundleTensors.clear();
        }
        else if (node->getNodeAnnotation().bundleInfo->bundleIndex != m_currBundleIndex)
        {
            m_lastBundleTensors.clear();
            m_currBundleIndex = node->getNodeAnnotation().bundleInfo->bundleIndex;
        }
        m_lastBundleTensors.emplace_back(tensor, node);

        LOG_TRACE(EPOCH_ALLOC,
                  "Add Tensor {} to m_lastBundleTensors with bundle index = {}",
                  tensor->getName(),
                  node->getNodeAnnotation().bundleInfo.is_set() ? node->getNodeAnnotation().bundleInfo->bundleIndex : -1);
    }

    m_currentEpoch.addTensor(node, tensor, writeSpaceForTensor);
    m_handledTensors.push_back(tensor);
    return {true, writeSpaceForTensor};
}

bool TensorsEpochAllocator::shouldHandleFallbackForNode(const pNode& node)
{
    return !m_lastBundleTensors.empty() &&                     // There are tensors to reallocate
           (!node->getNodeAnnotation().bundleInfo.is_set() ||  // The current node doesnt have a bundle index
            !m_lastBundleTensors.begin()->second->getNodeAnnotation().bundleInfo.is_set() ||  // Last bundle tensors
                                                                                              // dont have bundle index
            m_lastBundleTensors.begin()->second->getNodeAnnotation().bundleInfo->bundleIndex ==
                node->getNodeAnnotation().bundleInfo->bundleIndex);  // current node belongs to the last bundle
}

void TensorsEpochAllocator::handleNodeTensors(const pNode& node)
{
    const bool allowFallback = true;
    for (const TensorPtr& tensor : node->getOperands())
    {
        // shape tensors require no memory space and const tensors are allocated by the user
        if (tensor != nullptr && !tensor->isShapeTensor() && !tensor->getRealTensor(tensor)->inConstSection())
        {
            const auto& [success, writeSpaceForTensor] = handleNodeTensor(node, tensor, allowFallback);
            if (!success)
            {
                if (allowFallback && shouldHandleFallbackForNode(node))
                {
                    // Starting a new epoch might not help as there might be tensors belonging to the same bundle
                    // causing fragmentation. Free those tensors, start a epoch and reallocate last bundle tensors
                    m_lastBundleTensors.emplace_back(tensor, node);
                    if (!handleFallback(false))
                    {
                        // In case the first fallback failed - try to re-allocate m_lastBundleTensors using best fit
                        // policy (instead of first fit) as it might help with fragmentation issues.
                        LOG_WARN(EPOCH_ALLOC,
                                 "- {}: Second fallback for tensor {} with size {} - try to re-allocate using best fit "
                                 "allocation policy",
                                 HLLOG_FUNC,
                                 tensor->getName(),
                                 writeSpaceForTensor);
                        m_sramAllocator->setBestFitAllocation(true);
                        handleFallback(true);
                        m_sramAllocator->setBestFitAllocation(false);
                    }
                }
                else
                {
                    handleTensorDoesNotFitInSram(tensor, writeSpaceForTensor);
                }
            }
        }
    }
    m_lastNode = node;
}

void TensorsEpochAllocator::handleTensorDoesNotFitInSram(const pTensor& tensor, uint64_t writeSpaceForTensor)
{
    // If AuxTensor requires SRAM and it does not fit in SRAM, the compilation should fail
    if (tensor->isAuxTensor() && tensor->inSram())
    {
        LOG_TRACE(EPOCH_ALLOC,
                  "- {}: aux tensor {} with size {} doesn't fit SRAM (free space : {})",
                  HLLOG_FUNC,
                  tensor->getName(),
                  writeSpaceForTensor,
                  m_sramAllocator->getMaxFreeContiguous());
        throw InvalidTensorSizeException(tensor->getName());
    }
    LOG_TRACE(EPOCH_ALLOC,
              "- {}: tensor {} with size {} doesn't fit SRAM (free space : {}) and will be allocated with DRAM",
              HLLOG_FUNC,
              tensor->getName(),
              writeSpaceForTensor,
              m_sramAllocator->getMaxFreeContiguous());

    allocateTensorInDram(tensor);
}

void TensorsEpochAllocator::pushBackBornAndDiedTensor(const pTensor& tensor)
{
    auto& lst = m_currentEpoch.m_bornAtThisEpoch;
    // if the tensor exists, move it to the end of the list
    if (auto it = std::find(lst.begin(), lst.end(), tensor); it != lst.end())
    {
        lst.splice(lst.end(), lst, it);
    }
}

void TensorsEpochAllocator::handleEpochMemory(const pNode& node)
{
    TensorList tensorsToBeFree;

    for (const pTensor& tensor: m_handledTensors)
    {
        if ((node == nullptr || !m_livenessAnalysis->isRealTensorAliveAfterNode(node, tensor)) &&
            !tensor->isShapeTensor() && !tensor->getRealTensor(tensor)->inConstSection())
        {
            tensorsToBeFree.push_back(tensor);
            pushBackBornAndDiedTensor(tensor);
        }
    }

    allocateTensorsInSram();

    //now free them
    for (const TensorPtr& tensor : tensorsToBeFree)
    {
        m_handledTensors.remove(tensor);
        if (tensor->isZeroSizedDataTensor()) continue;
        LOG_TRACE(EPOCH_ALLOC,"Freeing dead tensor {}", tensor->getName());

        freeTensorFromSram(*m_graph,
                           tensor,
                           /*freeRealTensor*/ false,
                           m_sramAllocator.get(),
                           /*rollback*/ false,
                           m_nonPersistentSectionAllocTracker.get());
    }

    LOG_DEBUG(EPOCH_ALLOC,
              "{}::{}: Add epoch {}, epoch size: {}",
              m_name,
              HLLOG_FUNC,
              m_epochs.size(),
              m_currentEpoch.m_epochSize);
    m_epochs.push_back(m_currentEpoch);
}

void TensorsEpochAllocator::allocateTensorsInSram()
{
    LOG_DEBUG(EPOCH_ALLOC, "{}::Allocate {} Tensors in Sram, for epoch {}, epoch size: {}",
              m_name, m_currentEpoch.m_bornAtThisEpoch.size(), m_epochs.size(), m_currentEpoch.m_epochSize);

    for (const pTensor& tensor : m_currentEpoch.m_bornAtThisEpoch)
    {
        // shape tensors require no memory space and const tensors are allocated by the user
        if (tensor->isShapeTensor() || tensor->getRealTensor(tensor)->inConstSection()) continue;

        LOG_TRACE(EPOCH_ALLOC, "{}: Allocating {}", HLLOG_FUNC, tensor->getName());

        if (!::allocateTensorInSram(
                *m_graph,
                tensor,
                false,
                m_graph->getGraphAnnotation().memoryStrategyParams.dramInfo.enableDramAlloc,
                m_sramAllocator.get(),
                m_nonPersistentSectionAllocTracker.get()))
        {
            LOG_ERR(EPOCH_ALLOC, "{}: Failed to allocate memory in SRAM for tensor {}", HLLOG_FUNC, tensor->getName());
            throw NoMemory();
        }

        if (tensor->isStaticParam())
        {
            if (!::allocateTensorInDram(*m_graph,
                                        tensor,
                                        /*allocateRealTensor*/ true,
                                        /*allowFailure*/ false,
                                        /*alloc*/ &m_graph->getCodeGenerator()->getAllocatorForProgramData(),
                                        m_nonPersistentSectionAllocTracker.get()))
            {
                LOG_ERR(EPOCH_ALLOC, "{}: Failed to allocate memory in DRAM for tensor {}", HLLOG_FUNC, tensor->getName());
                throw NoMemory();
            }

            // if model parameter, mark it as prefetched
            TensorAnnotation &ann = tensor->getTensorAnnotation();
            ann.memorySpaceInfo.prefetchInfo.prefetch = true;
            ann.memorySpaceInfo.prefetchInfo.epoch = m_epochs.size();
            if (!m_epochLastNode.empty())
            {
                // Add the last node of the previous epoch as a barrier for prefetching this model parameter.
                tensor->getTensorAnnotation().memorySpaceInfo.barriers.push_back(m_epochLastNode.back());
            }
        }
    }
}

TensorsEpochAllocator::Epoch::Epoch(uint32_t slice) : m_currentSlice(slice)
{
}

void TensorsEpochAllocator::Epoch::addTensor(const pNode& node,
                                             const pTensor& tensor,
                                             uint64_t writeSpaceForTensor)
{
    if (!contains(m_bornAtThisEpoch, tensor))
    {
        LOG_TRACE(EPOCH_ALLOC,
                  "{}: Node: {}, {}\t\tepoch_size: {{ before: {} after: {} }}",
                  HLLOG_FUNC,
                  node->getNodeName(),
                  tensor->getName(),
                  m_epochSize,
                  m_epochSize + writeSpaceForTensor);
        m_bornAtThisEpoch.push_back(tensor);
        m_epochSize += writeSpaceForTensor;
        return;
    }

    LOG_TRACE(EPOCH_ALLOC, "{}: tensor already exists: {}", HLLOG_FUNC, tensor->getName());
}

bool StaticTensorsEpochAllocator::shouldCreateNewEpoch(const pNode& node, uint64_t writeSpaceForTensor)
{
    return writeSpaceForTensor + m_currentEpoch.m_epochSize > m_sramAllocator->getMaxFreeContiguous();
}

bool ActivationTensorsEpochAllocator::shouldCreateNewEpoch(const pNode& node, uint64_t writeSpaceForTensor)
{
    return writeSpaceForTensor + m_currentEpoch.m_epochSize > m_sramAllocator->getMaxFreeContiguous() ||
           m_currentEpoch.m_currentSlice != node->getNodeAnnotation().sliceIndex;
}

bool SramTensorsEpochAllocator::shouldCreateNewEpoch(const pNode& node, uint64_t writeSpaceForTensor)
{
    return writeSpaceForTensor + m_currentEpoch.m_epochSize > m_sramAllocator->getMaxFreeContiguous();
}

void SramTensorsEpochAllocator::allocateTensorInDram(const pTensor& tensor)
{
    LOG_WARN(EPOCH_ALLOC, "{}: Unexpectedly failed to allocate {} in SRAM.", HLLOG_FUNC, tensor->getName());

    HB_ASSERT(0, "SramTensorsEpochAllocator should not allocate tensors in DRAM!");

    LOG_WARN(EPOCH_ALLOC, "{}: trying to allocate the tensor {} in DRAM.", HLLOG_FUNC, tensor->getName());

    TensorsEpochAllocator::allocateTensorInDram(tensor);
}
