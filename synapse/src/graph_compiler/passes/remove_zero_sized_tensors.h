#pragma once

#include "habana_graph.h"
#include "types.h"

class ZeroSizedTensorRemover
{
public:

    friend bool removeZeroSizedTensors(HabanaGraph& g);
    /* c'tor:*/
    ZeroSizedTensorRemover(HabanaGraph& g);

    /**
     * @brief Check whether a node with 1+ zero sized tensors should be replaced with 0+ nodes.
     *
     * @param node       [in]  A node with zero sized tenosr(s) as either input/outputs
     * @param nodesToAdd [out] 0 or more nodes to replace \p node with, if ret value is true else unchanged
     *
     * @return true if \p node should be dropped and replaced with \p nodesToAdd
     */
    bool handleZeroSizedOperand(const NodePtr& node, /*OUT*/ TensorPtr& subTensor);

    /**
     * @brief Check whether a node has zero sized tensor
     *
     * @param node       [in]  A node to check
    *
     * @return true if node has zero sized tensor
     */
    static bool hasZST(const NodePtr& node) { return (findFirstZST(node) != nullptr); }

private:
    /* handling the case where all node's output tensors are zero-sized. */
    bool handleZeroOutputTensorsCase(const NodePtr& node);

    /* handling the case where some of the node's output tensors are not zero-sized*/
    bool handleNonZeroOutputTensorCase(const NodePtr& node, /*OUT*/ TensorPtr& subTensor);

    /* plan modifying the graph by replacing the node with an identity node. */
    void markRemovedNodesOutputsAsConst(const NodePtr& node);

    /* verifying the index space of the node in the graph*/
    bool tpcIndexSpaceIsZero(const TPCNode& tpcNode);

    static TensorPtr findFirstZST(const NodePtr& node);

private:
    tpc_lib_api::DeviceId m_deviceType;
};

bool removeZeroSizedTensors(HabanaGraph& g);