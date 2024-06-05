#pragma once

#include "mme_slicing_strategy.h"
#include "habana_graph.h"

/* Responsible for flattening MME Nodes (if possible) from 4-5 dims to 2D,
 * by merging all dimensions except the fcd.
 * flattening is allowed if the node operation can be converted to simple metrix multiplication (GEMM),
 * and will be lowered to 2D anyway.
 * Flattening will help SRAM Management pass to better slice the tensors and fit them in the SRAM memory.
 */
class MMENodeFlattener
{
public:
    explicit MMENodeFlattener(HabanaGraph& graph) : m_graph(graph) {}
    virtual ~MMENodeFlattener() = default;
    //Get the flattened shape of a given tensor.
    static SizeArray getFlattenShape(pTensor& operand, bool minSizes = false);
    // Checks if the given node can be flattened.
    static bool canFlattenMMENode(const pNode& node);

    // Flatten the node and set updated tensors in operands.
    // Returns a pair<status, set of new nodes created as part of flattening>
    std::pair<bool, NodeSet> doFlatten(const NodePtr& node, std::vector<pSlicedOperand>& slicedOperands);
    // Execute flattening (if possible) on all nodes in the graph. This function is used only for unit test
    bool execute();

private:
    // Perform flattening on a specific operand of a node - plant reshape or re-use existing reshape if possible.
    // Update slicing strategy operands if nessecary
    // Returns a pair <status, pointer to newly created reshape node if one was created o.w. nullptr>
    std::pair<bool, NodePtr>
    doFlattenAndAddReshapeNode(const pNode& node, pTensor& operand, std::vector<pSlicedOperand>&);
    // Calculate the new flattened shape of the given tensor.

    // Try to find an already flattened version of the tensor according to the given shape.
    // in case found - this can be reused instead of adding another reshape.
    pTensor getAlreadyFlattenTensor(pTensor origTensor, SizeArray& flatShape);

    // Create and add a rehsape node to the graph.
    // Returns a pair <status, node pointer to the newly created reshape node or nullptr upon failure>.
    std::pair<bool, NodePtr>
    createReshapeNode(pTensor& fromTensor, pTensor& toTensor, const std::string& origTensorName);

    // Copy tensor attributes from operand to flat tensor if necessary.
    void copyOperandAttributes(const TensorPtr& operand, TensorPtr& flatTensor);

    // Checks if the tensor is the output of the given node.
    bool isOutputTensorOfNode(const pNode& node, const pTensor& tensor);
    // Always flatten the tensors to DIM_W.
    static const DimsIndex getLoweredDim() {return DIM_W; /* = 1*/}

    HabanaGraph& m_graph;
};
