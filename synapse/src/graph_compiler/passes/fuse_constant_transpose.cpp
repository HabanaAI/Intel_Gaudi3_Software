#include "passes.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "tpc_node.h"
#include "perf_lib_layer_params.h"

#include <limits>
#include "graph_editor.h"
#include <memory>

void fuseConstantTransposeHelper(HabanaGraph& g, const NodePtr& transposeNode)
{
    auto constNode = g.getTensorProducer(transposeNode->getInput(0));
    auto constOut  = constNode->getOutput(0);
    // No fusing if there are other consumers.
    if (g.getNumberOfTensorConsumers(constOut) != 1)
    {
        LOG_TRACE(GC,
                  "Skipping fusion for node {}, since its output tensor has more than one consumer",
                  constNode->getNodeName());
        return;
    }
    if (constNode->getNumInputs() > 0)
    {
        // TODO [SW-91172]: add support for shape tensor
        LOG_TRACE(GC, "Skipping fusion for node {}, since it has more then 0 inputs", constNode->getNodeName());
        return;
    }
    if (constNode->getOutput(0)->getPermutation().has_value())
    {
        LOG_TRACE(GC, "Skipping fusion for node {}, since it has a permuted output", constNode->getNodeName());
        return;
    }
    LOG_TRACE(GC,
              "adding fusion of constant and transpose {}, {}",
              constNode->getNodeName(),
              transposeNode->getNodeName());
    // Now we can fuse.
    std::string newNodeName = constNode->getNodeName();
    newNodeName += "_fused_transpose";

    UserParams params     = (UserParams)constNode->getParamsRawData().data();
    unsigned   paramsSize = constNode->getParamsRawData().size();

    // create constant with flatten shape
    auto      fcdElements = g.getHALReader()->getTpcVectorSize() / constOut->getElementSizeInBytes();
    SizeArray flattenSizes;
    SizeArray flattenMinSizes;
    int       reshapeSize = 1;
    if (constOut->getTotalElements() % fcdElements == 0 && constOut->getMinimalElements() % fcdElements == 0 &&
        constOut->getTotalElements() > fcdElements && constOut->getMinimalElements() > fcdElements)
    {
        auto dim1Elements        = constOut->getTotalElements() / fcdElements;
        auto dim1ElementsMinimal = constOut->getMinimalElements() / fcdElements;
        flattenSizes             = {fcdElements, dim1Elements};
        flattenMinSizes          = {fcdElements, dim1ElementsMinimal};
        reshapeSize              = 2;
    }
    else
    {
        flattenSizes    = {constOut->getTotalElements()};
        flattenMinSizes = {constOut->getMinimalElements()};
    }
    // This cloned tensor will be used as intermediate -> no need to copy the data
    TensorPtr flattendTensor = constOut->clone(false, false, false);
    flattendTensor->reshape(reshapeSize, flattenSizes.data(), nullptr, flattenMinSizes.data());

    auto newConstNode =
        NodeFactory::createNode({}, {flattendTensor}, params, paramsSize, constNode->getGUID(), newNodeName);

    // create reshape to original sizes with constant output
    bool        enforceLogical     = true;
    std::string reshapeOrgNodeName = newNodeName + "_reshape_to_org";
    NodePtr     reshapeNodeOrg     = NodeFactory::createNode({newConstNode->getOutput(0)},
                                                     {constNode->getOutput(0)},
                                                     &enforceLogical,
                                                     "reshape",
                                                     reshapeOrgNodeName);

    // create reshape to transposed sizes with transpose output
    std::string reshapeTransposedNodeName = newNodeName + "_reshape_to_transposed";
    NodePtr     reshapeNodeTransposed     = NodeFactory::createNode({reshapeNodeOrg->getOutput(0)},
                                                            {transposeNode->getOutput(0)},
                                                            &enforceLogical,
                                                            "reshape",
                                                            reshapeTransposedNodeName);

    auto status =
        GraphEditor::replaceNodes(g, {constNode, transposeNode}, {newConstNode, reshapeNodeOrg, reshapeNodeTransposed});
    HB_ASSERT(status == REPLACE_NODE_SUCCESS, "{}: failed to fuse constant + transpose", __FUNCTION__);
}

// fuse constant + transpose to constant with transposed output shape
bool fuseConstantTranspose(HabanaGraph& g)
{
    NodeVector               nodesToHandle;
    static const std::string constGuid = "constant";
    for (const NodePtr& node : g.getNodes())
    {
        if (node->getNodeType() == Node::TYPE_INTERNAL_TRANSPOSE)
        {
            auto constNode = g.getTensorProducer(node->getInput(0));
            if (constNode && constNode->getNodeType() == Node::TYPE_USER &&
                constNode->getGUID().compare(0, constGuid.length(), constGuid) == 0)
            {
                nodesToHandle.emplace_back(node);
            }
        }
    }
    for (const NodePtr& node : nodesToHandle)
    {
        LOG_TRACE(GC, "Trying to add fusion of constant and transpose {}", node->getNodeName());
        fuseConstantTransposeHelper(g, node);
    }
    return true;
}
