#include "habana_graph.h"
#include "habana_nodes.h"

constexpr bool isOverlap(const Tensor::AddressRange& a, const Tensor::AddressRange& b)
{
    return a.first < b.second && b.first < a.second;
}

/*
 * Validate DMA memcopy does not have source and dest overlap
 */
bool validateDmaNodesSrcDstOvelap(HabanaGraph& graph)
{
    for (const pNode& node : graph.getNodes())
    {
        if (!node->isDma()) continue;

        const auto dmaNode = std::static_pointer_cast<DMANode>(node);
        if (dmaNode->isMemset())
        {
            LOG_TRACE(DMA_RANGE, "Skipping node: {} since it is memset (output only)", node->getNodeName());
            continue;
        }

        std::list<DMA_TYPE> hostDmaTypes{
                DMA_TYPE_UPSTREAM,
                DMA_TYPE_DOWNSTREAM,
                DMA_TYPE_INTERMEDIATES};
        if (std::find(hostDmaTypes.begin(), hostDmaTypes.end(), dmaNode->getDmaType()) != hostDmaTypes.end())
        {
            // Host and device address overlap should not matter.
            LOG_TRACE(DMA_RANGE, "Skipping node: {} since it is between host and device", node->getNodeName());
            continue;
        }

        if (dmaNode->canHaveAdditionalInputs())
        {
            HB_ASSERT(!node->getInputs().empty(), "Unexpected inputs num, expected >=1, received 0");
        }
        else
        {
            HB_ASSERT(node->getNumInputsDataTensors() == 1,
                      "Unexpected data inputs num, expected 1 while received [{}]",
                      node->getNumInputsDataTensors());
        }
        HB_ASSERT(node->getOutputs().size() == 1, "Unexpected outputs num, expected 1 while received [{}]",
                  node->getOutputs().size());

        LOG_TRACE(DMA_RANGE, "Checking overlap between input and output of node: {}", node->getNodeName());

        // inputs and outputs with index >0 do not participate in data transfer
        const pTensor& input  = node->getInput(0);
        const pTensor& output = node->getOutput(0);

        auto inputRanges  = input->getAddressRange();
        auto outputRanges = output->getAddressRange();

        for (const auto& inputRange : inputRanges)
        {
            for (const auto& outputRange : outputRanges)
            {
                if (isOverlap(inputRange, outputRange))
                {
                    if (inputRange == outputRange && isMemcpy(*node)) continue;  // allow exact match for memcopy

                    LOG_ERR(DMA_RANGE, "Overlap of input range: 0x{:x}-0x{:x} with output range: 0x{:x}-0x{:x} on node: {}, src: {}, dest: {}",
                            inputRange.first, inputRange.second, outputRange.first, outputRange.second,
                            node->getNodeName(), input->getName(), output->getName());
                    return false;
                }
            }
        }
    }
    return true;
}

bool validateDmaNodes(HabanaGraph& graph)
{
    // Long validation so will only be performed in debug:
    HB_DEBUG_VALIDATE(validateDmaNodesSrcDstOvelap(graph));

    return true;
}
