#include "allocate_pinned_static_tensors.h"

#include "allocators_utils.h"
#include "habana_graph.h"
#include "hal_reader/hal_reader.h"
#include "memory_management/memory_allocator.h"
#include "tensors_allocator.h"
#include "tensors_epoch_allocator.h"

#include <algorithm>
#include <map>
#include <memory>
#include <queue>
#include <set>

static float computeTensorScore(HabanaGraph& g, const std::shared_ptr<Tensor> &tensor, const std::shared_ptr<Tensor> &IFM)
{
    float mmeH = g.getHALReader()->getMmeAccumulatorH() / (float)tensor->getElementSizeInBytes();
    float reuse = g.getNumTensorConsumersIgnoreLogicals(tensor);        // Reuse on tensor
    float BHW = IFM->getTotalElements() / (float)IFM->getSizeInElements(0);  // NHWC / C = NHW

    LOG_TRACE(GC, " * computeTensorScore of tensor {}: MME_H = {}, reuse = {}, BHW = {}, score = {}",
            tensor->getName(), mmeH, reuse, BHW, ( (BHW / mmeH) / reuse ));
    return ( (BHW / mmeH) / reuse );
}

static bool pinSortedStaticTensors(HabanaGraph&     g,
                                   std::priority_queue<TensorScore, std::vector<TensorScore>, Compare> &sortedStaticTensors,
                                   deviceAddrOffset pinningBuffBase,
                                   uint64_t         pinningBuffSize,
                                   uint64_t         &allocatedSpace)
{
    while ((!sortedStaticTensors.empty()) && (allocatedSpace < pinningBuffSize))
    {
        std::shared_ptr<Tensor> tensor = sortedStaticTensors.top().m_tensor;
        sortedStaticTensors.pop();
        // Do not allocate same tensor more than once.
        // Since we loop over nodes inputs in allocatePinnedStaticTensors we can have same tensor several times in sortedStaticTensors queue
        if (tensor->tensorIsAllocated()) continue;

        uint64_t tensorSize = round_to_multiple( tensor->getTotalSizeInBytes() + tensor->getTensorAnnotation().memory.offset,
                                                 tensor->getTensorAnnotation().memory.alignment);

        LOG_TRACE(GC, "AllocatePinnedStaticTensors::Apply - trying to pin tensor {} of size {}. allocatedSpace {} pinningBuffSize {}",
                  tensor->getName(), tensorSize, allocatedSpace, pinningBuffSize);
        if (tensorSize + allocatedSpace <= pinningBuffSize)
        {
            // Allocate static tensor in the pinned buffer
            // No need to align as both base and size are aligned
            tensor->setSramOffset(pinningBuffBase + allocatedSpace);
            tensor->getTensorAnnotation().memory.pinned = true;
            allocatedSpace += tensorSize;

            if (!allocateTensorInDram(g, tensor, true, false, &g.getCodeGenerator()->getAllocatorForProgramData()))
            {
                LOG_ERR(GC, "AllocatePinnedStaticTensors::Apply - Failed to allocate tensor {} of size {} in DRAM",
                        tensor->getName(), tensorSize);
                return false;
            }
            LOG_TRACE(GC, "AllocatePinnedStaticTensors:: Apply - tensor {} allocated in SRAM addr {} (0x{:x}) DRAM addr {} (0x{:x})",
                      tensor->getName(), tensor->getSramOffset(), tensor->getSramOffset(),
                      tensor->getDramOffset(), tensor->getDramOffset());
        }
    }

    return true;
}

bool allocatePinnedStaticTensors(HabanaGraph& g)
{
    LOG_DEBUG(GC, "AllocatePinnedStaticTensors");

    MemoryStrategyParams    memParams            = g.getGraphAnnotation().memoryStrategyParams;
    MemoryAllocator         &sramAlloc           = g.getCodeGenerator()->getSramAllocator();
    unsigned                pinningBuff          = std::min<unsigned>(memParams.sramRegionsInfo.pinningSramBufferSize,
                                                                      sramAlloc.getMaxFreeContiguous());
    std::priority_queue<TensorScore, std::vector<TensorScore>, Compare> sortedStaticTensors;
    std::priority_queue<TensorScore, std::vector<TensorScore>, Compare> sortedSparsedStaticTensors;

    if (pinningBuff == 0)
    {
        LOG_DEBUG(GC, "AllocatePinnedStaticTensors::Apply - no buffer allocated for pinning static tensors");
        return true;
    }

    // Try to allocate pinned buffer in SRAM
    Settable<deviceAddrOffset> pinningBuffAdd = sramAlloc.Allocate(pinningBuff,
                                                                   g.getHALReader()->getCacheLineSizeInBytes(),
                                                                   0,
                                                                   true /* allowFailure */);
    if ((!pinningBuffAdd.is_set()) && ((pinningBuff - g.getHALReader()->getCacheLineSizeInBytes()) > 0 ))
    {
        // Failed to allocate the original requested size, allocate size - alignment
        pinningBuff -= g.getHALReader()->getCacheLineSizeInBytes();
        pinningBuffAdd = sramAlloc.Allocate(pinningBuff,
                                            g.getHALReader()->getCacheLineSizeInBytes(),
                                            0,
                                            true /* allowFailure */);
    }

    if (!pinningBuffAdd.is_set())
    {
        LOG_WARN(GC, "AllocatePinnedStaticTensors::Apply - no buffer allocated for pinning static tensors, "
                       "while original requested buffer size is {} MB", bToMb(memParams.sramRegionsInfo.pinningSramBufferSize));
        HB_ASSERT(pinningBuff == 0, "allocatePinnedStaticTensors: failed to allocate buffer for pinned static tensors.");
        return true;
    }

    LOG_TRACE(GC, "AllocatePinnedStaticTensors::Apply: allocated {} ({}MB) for pinned buffer.",
              pinningBuff,
              bToMb(pinningBuff));
    deviceAddrOffset base = pinningBuffAdd.value();

    unsigned tensorCount = 0; // for debug
    unsigned nodeCount = 0;
    LOG_TRACE(GC, "AllocatePinnedStaticTensors::Apply - Sort the graphs static parameters by their score");
    for (std::shared_ptr<Node> node : g.getExeSortedNodes())
    {
        for (std::shared_ptr<Tensor> tensor : node->getInputs())
        {
            if (tensor == nullptr) continue;

            pTensor realTensor = Tensor::getRealTensor(tensor);

            // Const tensors are allocated by the user and should not be placed on the SRAM
            if (realTensor->isStaticParam() && !realTensor->inConstSection())
            {
                float score = 0;
                unsigned alignedSize = round_to_multiple(realTensor->getTotalSizeInBytes() + realTensor->getTensorAnnotation().memory.offset,
                        realTensor->getTensorAnnotation().memory.alignment);
                // Static tensors of first 2 layers with size <= 100KB will be given the lowest score (0) to prioritize pinning them over other tensors.
                if ((nodeCount >= NUM_OF_FIRST_LAYERS_TO_PIN) || (alignedSize > MAX_SIZE_IN_FIRST_LAYERS_TO_PIN))
                {
                    std::shared_ptr<Tensor> IFM = node->getInput(TENSOR_IFM);
                    // Compute static tensor score
                    score = computeTensorScore(g, realTensor, IFM); // TODO - score should be computed according to node type (TPC/MME)
                }
                // Add to the static tensors sorted list
                TensorScore tensorScore;
                tensorScore.m_tensor = realTensor;
                tensorScore.m_score = score;
                tensorScore.id = tensorCount++;

                if (HabanaGraph::runsOnTPC(node) && realTensor->getTensorAnnotation().sparseAccess)
                {
                    sortedSparsedStaticTensors.push(tensorScore);
                }
                else
                {
                    sortedStaticTensors.push(tensorScore);
                }
            }
        }
        nodeCount++;
    }

    LOG_TRACE(GC, "AllocatePinnedStaticTensors::Apply - Go over sorted list and choose tensors with lowest score");
    uint64_t allocatedSpace = 0;
    if (!pinSortedStaticTensors(g, sortedStaticTensors, base, pinningBuff, allocatedSpace))
    {
        LOG_ERR(GC, "AllocatePinnedStaticTensors::Apply - pinSortedStaticTensors failed for sortedStaticTeonsors");
        return false;
    }

    LOG_TRACE(GC, "AllocatePinnedStaticTensors::Apply - Go over sorted sparsed list and choose tensors with lowest score");
    if (!pinSortedStaticTensors(g, sortedSparsedStaticTensors, base, pinningBuff, allocatedSpace))
    {
        LOG_ERR(GC, "AllocatePinnedStaticTensors::Apply - pinSortedStaticTensors failed for sortedSparsedStaticTensors");
        return false;
    }

    return true;
}
