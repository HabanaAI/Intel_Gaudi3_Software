#include "passes.h"
#include "habana_graph.h"
#include "reduction_node.h"

constexpr bool isOverlap(const Tensor::AddressRange& a, const Tensor::AddressRange& b)
{
    return a.first < b.second && b.first < a.second;
}

bool validateMemoryAllocation(HabanaGraph& g)
{
    auto exeSched = g.getExeSortedNodes();
    for (const auto& node : exeSched)
    {
        // validate all tensors were allocated
        for (const auto&  t : node->getOperands())
        {
            if (t == nullptr) continue;

            // Non-device shape tensors are not allocated
            if (t->isShapeTensor()) continue;

            if (!t->tensorIsAllocated())
            {
                LOG_ERR(GC, "Node {} tensor {} was not allocated", node->getNodeName(), t->getName());
                return false;
            }
            if (t->tensorAllocatedInDram() && t->tensorAllocatedInSram())
            {
                LOG_ERR(GC, "Node {} tensor {} was allocated both in Dram and Sram, this is forbiden in Gaudi",
                        node->getNodeName(), t->getName());
                return false;
            }
        }

        // validate reduction output is in SRAM
        if (node->getNodeType() == Node::TYPE_INTERNAL_REDUCTION && g.getDeviceType() == synDeviceGaudi)
        {
            HB_ASSERT(node->getNumOutputs() == 1, "Reduction node should have a single output");

            auto reductionNode = std::dynamic_pointer_cast<ReductionNode>(node);
            // REDUCTION_SET does not require specific memory hw type
            if (reductionNode != nullptr && !ReductionInfo::isReductionSet(reductionNode->getReductionOperation()))
            {
                if (!node->getOutput(0)->tensorAllocatedInSram())
                {
                    LOG_ERR(GC,
                            "Reduction node {} output is not in SRAM, cannot perform reduction",
                            node->getNodeName());
                    return false;
                }
                for (const auto& tensor : node->getOperands())
                {
                    // Reduction nodes cannot have shape tensor operands
                    if (tensor->isShapeTensor())
                    {
                        LOG_ERR(GC,
                                "Reduction node {} tensor is a shape tensor, cannot perform reduction",
                                node->getNodeName());
                        return false;
                    }

                    if (tensor->getElementType() != syn_type_float)
                    {
                        LOG_ERR(GC,
                                "Reduction node {} tensor is not in f32, cannot perform reduction",
                                node->getNodeName());
                        return false;
                    }
                }
            }
        }

        // validate MME inputs are in SRAM
        if (!g.getHALReader()->isCacheSupported() && HabanaGraph::runsOnMME(node))
        {
            for (const pTensor& input : node->getInputs())
            {
                if (input && !input->tensorAllocatedInSram() && !input->isShapeTensor())
                {
                    LOG_WARN(SRAM_SLICE, "Node {} reads input tensor {} from HBM", node->getNodeName(), input->getName());
                }
            }
        }
    }

    // This checks overlaps between each node input and addresses discarded by nodes scheduled between the input
    // producer and the current consumer, to catch read-after-discard bugs..
    if (g.getHALReader()->isCacheSupported())
    {
        auto overlaps = [](const TensorPtr& t1, const TensorPtr& t2) {
            for (const auto& dr1 : t1->getAddressRange())
            {
                for (const auto& dr2 : t2->getAddressRange())
                {
                    if (isOverlap(dr1, dr2)) return true;
                }
            }
            return false;
        };

        auto currIter = exeSched.begin();
        while (currIter != exeSched.end())
        {
            const auto& currNode = *currIter;
            if (!currNode->isLogicalOperation())
            {
                LOG_TRACE(LB_CACHE_MNGR, "Validating no read after discard for node: {}", currNode->getNodeName());
                for (const auto& in : currNode->getInputs())
                {
                    if (!in || in->isShapeTensor()) continue;
                    LOG_TRACE(LB_CACHE_MNGR, "> Checking input: {}", in->getName());
                    const auto& producer = g.getTensorProducer(in);
                    auto        prevIter = exeSched.begin();
                    if (producer)
                    {
                        prevIter = std::find(exeSched.begin(), currIter, producer);
                        HB_ASSERT(prevIter != currIter,
                                  "Producer wasn't found before consumer for produced tensor: {} (producer: {}, "
                                  "consumer: {})",
                                  in->getName(),
                                  producer->getNodeName(),
                                  currNode->getNodeName());
                        prevIter++;
                    }
                    while (prevIter != currIter && prevIter != exeSched.end())
                    {
                        const auto& prevNode = *prevIter;
                        if (!prevNode->isLogicalOperation())
                        {
                            LOG_TRACE(LB_CACHE_MNGR,
                                      "-> Checking against previous executing node: {}",
                                      prevNode->getNodeName());
                            for (size_t prevInIdx = 0; prevInIdx < prevNode->getNumInputs(); prevInIdx++)
                            {
                                const auto& prevIn = prevNode->getInput(prevInIdx);
                                if (!prevIn || prevIn->isShapeTensor()) continue;
                                if (prevNode->getNodeAnnotation().inputsCacheMetaData.at(prevInIdx).cmAction ==
                                    CacheMaintenanceAction::DISCARD)
                                {
                                    HB_ASSERT(!overlaps(in, prevIn),
                                              "Discard before last read! Reader: {}, Reader input: {}, Discarder: {}, "
                                              "Discarder input: {}",
                                              currNode->getNodeName(),
                                              in->getName(),
                                              prevNode->getNodeName(),
                                              prevIn->getName());
                                    // Assertion was not raised...
                                    LOG_TRACE(LB_CACHE_MNGR,
                                              "--> Discarded input {} is not overlapping with {}",
                                              prevIn->getName(),
                                              in->getName());
                                }
                            }
                        }
                        prevIter++;
                    }
                }
            }
            currIter++;
        }
    }
    return true;
}
