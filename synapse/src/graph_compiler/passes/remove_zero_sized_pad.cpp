#include "habana_graph.h"
#include "passes.h"
#include "tpc_node.h"
#include "perf_lib_layer_params.h"


bool removeZeroSizedPad(HabanaGraph& g)
{
    NodeSet        padsToRemove;
    const NodeSet& nodes = g.getNodes();
    for (const NodePtr& node : nodes)
    {
        if (!HabanaGraph::runsOnTPC(node) || !startsWith(node->getGUID(), "pad"))
        {
            continue;
        }

        const ns_PadKernel::Params* padParams =
            (ns_PadKernel::Params*)(std::reinterpret_pointer_cast<TPCNode>(node)->getParams());
        if (std::all_of(std::begin(padParams->pads), std::end(padParams->pads), [](uint32_t i) { return i == 0; }) &&
            node->getInputs().size() == 1 &&
            node->getInput(TENSOR_IFM)->compareGeometry(*(node->getOutput(TENSOR_OFM))))
        {
            // All params are 0 thus pad is the same as identity node and will be candidate for removal.
            LOG_TRACE(GC, "{}: {} Added as candidate to removal", HLLOG_FUNC, node->getNodeName());
            padsToRemove.insert(node);
        }
    }

    for (const NodePtr& pad : padsToRemove)
    {
        LOG_DEBUG(GC, "Trying to remove zero-sized pad node {}", pad->getNodeName());
        HB_ASSERT(pad->getInputs().size() == 1, "pad must have only one input");
        HB_ASSERT(pad->getOutputs().size() == 1, "pad must have only one output");
        ConstTensorPtr padInput  = pad->getInput(TENSOR_IFM);
        ConstTensorPtr padOutput = pad->getOutput(TENSOR_OFM);
        HB_ASSERT(padInput->compareGeometry(*padOutput),
                  "input and output of zero-sized pad are not of the same shape");
        // TODO: [SW-140767] catch failures - print informative log.
        GraphEditor::removeOneToOneNode(g, pad);
    }
    return true;
}