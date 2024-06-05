#include "habana_graph.h"
#include "log_manager.h"

#include "passes.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "graph_editor.h"
#include "habana_graph.h"

void findPattern(HabanaGraph& g, NodeSet& foundPatterns)
{
    // sequence_mask->softmax
    Graph                 pattern;
    bool                  status;
    {
        pTensor smInput1  = std::make_shared<Tensor>();
        pTensor smInput2  = std::make_shared<Tensor>();
        pTensor smOutput  = std::make_shared<Tensor>();
        pNode   smNode    = NodeFactory::createGenericTPCNode({smInput1, smInput2}, {smOutput}, nullptr, "sequence_mask", "");
        pTensor softmaxOutput  = std::make_shared<Tensor>();
        pNode   softmaxNode    = NodeFactory::createGenericTPCNode({smOutput}, {softmaxOutput}, nullptr, "softmax", "");

        status = pattern.addNode(smNode);
        status = status && pattern.addNode(softmaxNode);
    }

    if (status)
    {
        foundPatterns = g.matchPatternWithSingleOutputNode(&pattern, NodeTypeMatchingFunc);
    }
    else
    {
        LOG_DEBUG(GC, "Pattern build failed for SequenceMask-SoftMax.");
        foundPatterns = NodeSet();
    }
}

bool fuseIntoMaskInvalidSoftmax(HabanaGraph& g)
{
    NodeSet foundPatterns;
    findPattern(g, foundPatterns);

    for (auto softmaxNode : foundPatterns)
    {
        HB_ASSERT(softmaxNode->getNumInputs() == 1, "the number of inputs is {} but should be 1",
                  softmaxNode->getNumInputs());
        pNode smNode = g.getTensorProducer(softmaxNode->getInput(0));
        std::shared_ptr<TPCNode> tpcSoftmaxNode = std::dynamic_pointer_cast<TPCNode>(softmaxNode);
        std::shared_ptr<TPCNode> tpcSequenceMaskNode = std::dynamic_pointer_cast<TPCNode>(smNode);
        HB_ASSERT_PTR(tpcSoftmaxNode);
        HB_ASSERT_PTR(tpcSequenceMaskNode);
        HB_ASSERT(tpcSoftmaxNode->isGuidPrefix("softmax"), "softmax node GUID prefix must be 'softmax' ");
        HB_ASSERT(tpcSequenceMaskNode->isGuidPrefix("sequence_mask"),
                  "sequence mask node GUID prefix must be 'sequence_mask' ");

        if (tpcSequenceMaskNode->getNumInputs() != 2 ||
            (!GCFG_SYNAPSE_DATA_TYPE_SELECTION.value() &&
             (tpcSequenceMaskNode->getGUID().find("_i16") == std::string::npos ||
              tpcSoftmaxNode->getGUID().find("_i16") == std::string::npos)) ||
            !GraphEditor::canEliminateTensor(g, tpcSequenceMaskNode->getOutput(0)) ||
            !((ns_SequenceMask::Params*)tpcSequenceMaskNode->getParams())->use_sequence_length)
        {
            continue;
        }

        if (((ns_SequenceMask::Params*)tpcSequenceMaskNode->getParams())->mask_value > -100.0)
        {
            LOG_WARN(GC, "SequenceMask->Softmax pattern works better when mask_value < -100.");
            continue;
        }

        if (((ns_Softmax::Params*)tpcSoftmaxNode->getParams())->dim != 0)
        {
            LOG_WARN(GC, "SequenceMask->Softmax pattern works better when dim = 0.");
            continue;
        }

        pNode maskInvalidSoftmaxNode = NodeFactory::createNode(smNode->getInputs(),
                                                               softmaxNode->getOutputs(),
                                                               tpcSoftmaxNode->getParams(),
                                                               tpcSoftmaxNode->getParamsSize(),
                                                               tpcSoftmaxNode->getGUID(),
                                                               tpcSoftmaxNode->getNodeName());

        if (GraphEditor::replaceNodes(g, {softmaxNode, smNode}, {maskInvalidSoftmaxNode}) != REPLACE_NODE_SUCCESS)
        {
            LOG_ERR(GC, "{}: failed to fuse into maskInvalidSoftmax node", HLLOG_FUNC);
            return false;
        }
    }

    return true;
}
