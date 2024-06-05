
#pragma once

#include "convolution_node.h"
#include "habana_graph.h"

class GroupedConvolutionPackingManager
{
public:
    GroupedConvolutionPackingManager(const ConvBaseNode* n, const HabanaGraph& g);
    NodeVector packGroupedConvNode();

private:
    NodePtr createSplitNodeForGConvInput(const std::string& nodeName,
                                         const TensorPtr&   in,
                                         SizeArray          p1MaxSizes,
                                         SizeArray          p1MinSizes,
                                         bool               isShape = false);
    bool    isValidGConvForPacking() const;
    void    resizeForPacking(SizeArray& maxSizes, SizeArray& minSizes, unsigned nGroups, unsigned origGroupsForCurConv);

    using GConvOperands = std::tuple<TensorPtr, TensorPtr, TensorPtr, TensorPtr>;
    using SplitNodes    = std::tuple<NodePtr, NodePtr, NodePtr>;
    struct PackedGConvContext
    {
        // Represent group convolution node
        GConvOperands          operands;
        synConvolutionParamsV2 newParams;
        // Aggregates the new split nodes which are added by the packer.
        SplitNodes splitNodes;
        // Aggregates the new nodes which are added by the packer.
        NodeVector newNodes;
        // Aggregates outputs of the created packed gconvs that should be concatenated to the original gconv output.
        TensorVector pOuts;
        // Aggregates static tensors which their data should be placed in some const section (used in constant folding).
        TensorVector aggConstSectionTensors;
    };

    void handleGroupRemainder(PackedGConvContext& context, unsigned newGConvIdx, unsigned origGroupsForCurConv);
    void addNewGConvNode(PackedGConvContext& context,
                         unsigned            newGConvIdx,
                         const TensorVector& inputs,
                         const TensorPtr&    output,
                         unsigned            origGroupsForCurConv);
    void addGConvFwdAndNewGConv(PackedGConvContext& context,
                                unsigned            newGConvIdx,
                                unsigned            totalNewGConvNodes,
                                unsigned            origGroupsForCurConv,
                                std::string_view    dType);
    void addGConvBwdAndNewGConv(PackedGConvContext& context,
                                unsigned            newGConvIdx,
                                unsigned            origGroupsForCurConv,
                                std::string_view    dType);
    void addConcatNode(PackedGConvContext& context, const TensorPtr& out);

    void runGConvFwdOnCpu(const TensorPtr& input, const TensorPtr& output);
    template<typename T>
    void runGConvFwdOnCpuPerType(const TensorPtr& input, const TensorPtr& output);
    template<typename T>
    void copyElementOfWeightsToInflatedWeights(const TensorPtr&  weights,
                                               const CoordArray& weightsCoordinates,
                                               const TensorPtr&  inflatedWeights);
    bool isSupportedDataType(synDataType type) const;

    const HabanaGraph&  m_graph;
    const ConvBaseNode* m_convNode;
    unsigned m_numOriginalGroups;
    unsigned m_kPerGroup;
    unsigned m_vectorSize;
    unsigned m_groupsPerVector;
    unsigned m_groupsQuotient;
    unsigned m_groupsRemainder;
};