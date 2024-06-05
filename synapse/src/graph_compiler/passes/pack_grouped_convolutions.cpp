#include "habana_graph.h"
#include "node_factory.h"
#include "pack_grouped_convolutions.h"
#include "compilation_hal_reader.h"
#include "dedx_node.h"
#include "dedw_node.h"
#include "data_type_utils.h"
#include "tensor_data_manipulation_util.h"
#include "const_section_util.h"
#include "types.h"
#include <tuple>

GroupedConvolutionPackingManager::GroupedConvolutionPackingManager(const ConvBaseNode* n, const HabanaGraph& g)
:  m_graph(g), m_convNode(n)
{
    TensorPtr IFM = m_convNode->getXOperand();
    TensorPtr WGH = m_convNode->getWOperand();
    TensorPtr OFM = m_convNode->getYOperand();

    m_numOriginalGroups = m_convNode->getConvolutionParams().nGroups;
    m_vectorSize        = g.getHALReader()->getMmeMinimalWidthInElems(IFM->getElementType());
    m_kPerGroup         = WGH->getSizeInElements(WEIGHT_DIM_K) / m_numOriginalGroups;

    //# of ORIGINAL groups in one new gConv node:
    m_groupsPerVector = std::min(m_numOriginalGroups, std::max(1U, m_vectorSize / m_kPerGroup));
    //# of NEW groups in the first new GConv node: (ORIGINAL # of groups in second GConv node = m_groupsPerVector)
    m_groupsQuotient  = m_numOriginalGroups / m_groupsPerVector;
    //# of ORIGINAL groups in the second new GConv node: (NEW # of groups in second GConv node = 1)
    m_groupsRemainder = m_numOriginalGroups % m_groupsPerVector;

    LOG_DEBUG(GC, "{}: Number of groups per mme vector: {}", HLLOG_FUNC, m_groupsPerVector);
    LOG_DEBUG(GC, "{}: Number of packed groups: {}", HLLOG_FUNC, m_groupsQuotient);
    LOG_DEBUG(GC, "{}: Number of original groups in remainder: {}", HLLOG_FUNC, m_groupsRemainder);
}

bool GroupedConvolutionPackingManager::isValidGConvForPacking() const
{
    if (!isSupportedDataType(m_convNode->getWOperand()->getElementType()))
    {
        LOG_WARN(GC, "Grouped convolution packing is only supported for fp8, bf16 and f32 data types");
        return false;
    }
    if (m_convNode->is3DConvolution())
    {
        // TODO: fix for 3d conv
        LOG_WARN(GC, "Grouped convolution packing not supported for 3D convolution");
        return false;
    }
    if (m_convNode->getWOperand()->isDynamicDim(WEIGHT_DIM_C) || m_convNode->getWOperand()->isDynamicDim(WEIGHT_DIM_K))
    {
        LOG_WARN(GC, "Dynamic C/K for grouped convolution packing is not supported");
        return false;
    }

    return true;
}

NodePtr GroupedConvolutionPackingManager::createSplitNodeForGConvInput(const std::string& nodeName,
                                                                       const TensorPtr&   in,
                                                                       SizeArray          p1MaxSizes,
                                                                       SizeArray          p1MinSizes,
                                                                       bool               isShape /* = false */)
{
    HB_ASSERT((unsigned)DIM_C == (unsigned)WEIGHT_DIM_K, "weights are resized on WEIGHT_DIM_K which is the same as DIM_C in enum");
    unsigned  splitDim = DIM_C;
    auto [p2MinSizes, p2MaxSizes] = in->getAllMinMaxSizesInElements();
    p2MaxSizes[splitDim] = p2MaxSizes[splitDim] - p1MaxSizes[splitDim];
    p2MinSizes[splitDim] = p2MinSizes[splitDim] - p1MinSizes[splitDim];

    TensorPtr out1 = std::make_shared<Tensor>(in->getDim(), p1MaxSizes.data(), in->getElementType(), p1MinSizes.data());
    TensorPtr out2 = std::make_shared<Tensor>(in->getDim(), p2MaxSizes.data(), in->getElementType(), p2MinSizes.data());

    if (isShape)
    {
        out1->setTensorType(in->getTensorType());
        out2->setTensorType(in->getTensorType());
    }

    return NodeFactory::createNode({in},
                                   {out1, out2},
                                   &splitDim,
                                   isShape ? NodeFactory::splitShapeNodeTypeName
                                           : NodeFactory::splitNodeInternalTypeName,
                                   nodeName);
}

void GroupedConvolutionPackingManager::resizeForPacking(SizeArray& maxSizes,
                                                        SizeArray& minSizes,
                                                        unsigned   nGroups,
                                                        unsigned   origGroupsForCurConv)
{
    maxSizes[DIM_C] = (maxSizes[DIM_C] / m_numOriginalGroups) * nGroups * origGroupsForCurConv;
    minSizes[DIM_C] = (minSizes[DIM_C] / m_numOriginalGroups) * nGroups * origGroupsForCurConv;
}

/**
 * @brief This funciton is designed to handle gconv_fwd on the CPU, provided that its input's data
          is known at compile time.
 * @param input input tensor of gconv_fwd, the weights before inflation.
 * @param output output tensor of gconv_fwd, the weights after inflation.
 */
void GroupedConvolutionPackingManager::runGConvFwdOnCpu(const TensorPtr& input, const TensorPtr& output)
{
    LOG_INFO(GC, "{}: input tensor: {}, output tensor: {}", HLLOG_FUNC, input->getName(), output->getName());
    HB_ASSERT(input->getElementType() == output->getElementType(), "input and output data types should be equal");

    switch (input->getElementType())
    {
        case syn_type_float:
            runGConvFwdOnCpuPerType<AsCppType<syn_type_float>>(input, output);
            break;
        case syn_type_fp8_143:
            runGConvFwdOnCpuPerType<AsCppType<syn_type_fp8_143>>(input, output);
            break;
        case syn_type_fp8_152:
            runGConvFwdOnCpuPerType<AsCppType<syn_type_fp8_152>>(input, output);
            break;
        case syn_type_bf16:
            runGConvFwdOnCpuPerType<AsCppType<syn_type_bf16>>(input, output);
            break;
        case syn_type_fp16:
            runGConvFwdOnCpuPerType<AsCppType<syn_type_fp16>>(input, output);
            break;
        case syn_type_int8:
            runGConvFwdOnCpuPerType<AsCppType<syn_type_int8>>(input, output);
            break;
        case syn_type_uint8:
            runGConvFwdOnCpuPerType<AsCppType<syn_type_uint8>>(input, output);
            break;
        case syn_type_int16:
            runGConvFwdOnCpuPerType<AsCppType<syn_type_int16>>(input, output);
            break;
        case syn_type_uint16:
            runGConvFwdOnCpuPerType<AsCppType<syn_type_uint16>>(input, output);
            break;
        case syn_type_int32:
            runGConvFwdOnCpuPerType<AsCppType<syn_type_int32>>(input, output);
            break;
        case syn_type_uint32:
            runGConvFwdOnCpuPerType<AsCppType<syn_type_uint32>>(input, output);
            break;
        case syn_type_int64:
            runGConvFwdOnCpuPerType<AsCppType<syn_type_int64>>(input, output);
            break;
        case syn_type_uint64:
            runGConvFwdOnCpuPerType<AsCppType<syn_type_uint64>>(input, output);
            break;
        default:
            HB_ASSERT(false, "Unexpected syn data type");
    }
}

template<typename T>
void GroupedConvolutionPackingManager::copyElementOfWeightsToInflatedWeights(const TensorPtr&  weights,
                                                                             const CoordArray& weightsCoordinates,
                                                                             const TensorPtr&  inflatedWeights)
{
    const auto element               = TensorElementAccess::getElement<T>(weights, weightsCoordinates);
    const auto kernelsPerPackedGroup = m_groupsPerVector * m_kPerGroup;
    // Calculate the index of element's old group in the packed group.
    const auto oldGroupIndexInPackedGroup = (weightsCoordinates[WEIGHT_DIM_K] % kernelsPerPackedGroup) / m_kPerGroup;

    // Use this index for calculating the kernel offset on the inflated weights channel dim.
    const auto cOut =
        weightsCoordinates[WEIGHT_DIM_C] + oldGroupIndexInPackedGroup * weights->getAllSizesInElements()[WEIGHT_DIM_C];
    CoordArray coordinatesInOutputTensor    = weightsCoordinates;
    coordinatesInOutputTensor[WEIGHT_DIM_C] = cOut;
    TensorElementAccess::setElement<T>(inflatedWeights, coordinatesInOutputTensor, element);
}

/**
 * @brief Runs gconv_fwd core logic.
        It loops over the inputs' elements, for each element it calculates its new coordinate in the inflated weights, then it writes it there.
        The coordinates in the inflated weights are calculated as follows: given input element's k coordinate (tells us to which kernel this element belongs),
        it calculates the index of its group in the packed group.
        Then, this calculated index is used for the calculation of the c coordinate in the inflated weights: cOut = c_coordinate_in_input + group_index * input_c_dim_size.
        Example:
        Given this grouped convolution node:
        number of groups: 2.
            X [4,3,3,1]:                    W [4,2,2,2]
            [1,1,1,1],[2,2,2,2],[3,3,3,3]   Group1: k1: [1,1],[1,1] k2: [2,2],[2,2]
            [4,4,4,4],[5,5,5,5],[6,6,6,6]               [1,1],[1,1]     [2,2],[2,2]
            [7,7,7,7],[8,8,8,8],[9,9,9,9]   Group2: k3: [3,3],[3,3] k4: [4,4],[4,4]
                                                        [3,3],[3,3]     [4,4],[4,4]

        Let's say that the packer packed these 2 groups to one packed group.
        Then: W_inflated [4,2*2,2,2]
          k1: [1,1,0,0],[1,1,0,0] k2: [2,2,0,0],[2,2,0,0] k3: [0,0,3,3],[0,0,3,3] k4: [0,0,4,4],[0,0,4,4]
              [1,1,0,0],[1,1,0,0]     [2,2,0,0],[2,2,0,0]     [0,0,3,3],[0,0,3,3]     [0,0,4,4],[0,0,4,4]
        Then, the gc won't split this grouped conv node (because it has only one group) and the calculation in the img2col will look like that:
                                            0034
                                            0034
                                            1200
                                            1200
                                            0034
                                            0034
                                            1200
                                            1200
                                            0034
                                            0034
                                            1200
                                            1200
                                            0034
                                            0034
                                            1200
                                            1200
                             5555444422221111
                             6666555533332222
                             ...

        This way (inflating the weights) we're keeping the user demand that each group of kernels work on their specific range of channels in the image.

            Pre-assumptions: input and output has the same data type
 * @param input input tensor of gconv_fwd, the weights before inflation.
 * @param output output tensor of gconv_fwd, the weights after inflation.
 */
template<typename T>
void GroupedConvolutionPackingManager::runGConvFwdOnCpuPerType(const TensorPtr& input, const TensorPtr& output)
{
    HB_ASSERT(!m_convNode->is3DConvolution(), "running gconv_fwd on the cpu is not supported in 3d convolution");
    // It is assumed that output tensor is trivial strided (as well as not sparse).
    HB_ASSERT(output->isTrivialStrided(), "Expecting trivially strided output");

    LOG_DEBUG(GC, "{}: inflating {} to this {}", HLLOG_FUNC, input->getName(), output->getName());
    LOG_DEBUG(GC,
              "{}: old groups: {}, new groups: {}",
              HLLOG_FUNC,
              m_groupsPerVector * m_groupsQuotient + m_groupsRemainder,
              m_groupsQuotient + (m_groupsRemainder == 0 ? 0 : 1));

    memset(output->map(), 0, output->getTotalSizeInBytes());
    const auto& sizes = input->getAllSizesInElements();
    for (unsigned r = 0; r < sizes[WEIGHT_DIM_R]; ++r)
    {
        for (unsigned s = 0; s < sizes[WEIGHT_DIM_S]; ++s)
        {
            for (unsigned c = 0; c < sizes[WEIGHT_DIM_C]; ++c)
            {
                for (unsigned k = 0; k < sizes[WEIGHT_DIM_K]; ++k)
                {
                    copyElementOfWeightsToInflatedWeights<T>(input, {k, c, s, r}, output);
                }
            }
        }
    }
}

void GroupedConvolutionPackingManager::handleGroupRemainder(PackedGConvContext& context,
                                                            unsigned           newGConvIdx,
                                                            unsigned           origGroupsForCurConv)
{
    auto& [in0, in1, in2, out]                       = context.operands;
    auto& [splitIn0Node, splitIn1Node, splitIn2Node] = context.splitNodes;
    auto [curIn0MinSizes, curIn0MaxSizes]            = in0->getAllMinMaxSizesInElements();
    auto [curIn1MinSizes, curIn1MaxSizes]            = in1->getAllMinMaxSizesInElements();
    auto [curOutMinSizes, curOutMaxSizes]            = out->getAllMinMaxSizesInElements();
    SizeArray curIn2MaxSizes;
    SizeArray curIn2MinSizes;
    // resize ops on DIM_C (weights are resized on WEIGHT_DIM_K which is also 0 in enum)
    //(op->DIM_C / numOrigGroups) is the cPerGroup or kPerGroup of this tensor
    HB_ASSERT((unsigned)DIM_C == (unsigned)WEIGHT_DIM_K,
              "weights are resized on WEIGHT_DIM_K which is the same as DIM_C in enum");
    resizeForPacking(curIn0MaxSizes, curIn0MinSizes, context.newParams.nGroups, origGroupsForCurConv);
    resizeForPacking(curIn1MaxSizes, curIn1MinSizes, context.newParams.nGroups, origGroupsForCurConv);
    resizeForPacking(curOutMaxSizes, curOutMinSizes, context.newParams.nGroups, origGroupsForCurConv);
    if (in2)
    {
        std::tie(curIn2MinSizes, curIn2MaxSizes) = in2->getAllMinMaxSizesInElements();
        resizeForPacking(curIn2MaxSizes, curIn2MinSizes, context.newParams.nGroups, origGroupsForCurConv);
    }

    out = std::make_shared<Tensor>(out->getDim(), curOutMaxSizes.data(), out->getElementType(), curOutMinSizes.data());
    out->setName(out->getName() + "_subgroup_" + std::to_string(newGConvIdx));
    if (newGConvIdx == 0)
    {
        LOG_DEBUG(GC, "{}: Creating split nodes", HLLOG_FUNC);
        // create split node for the inputs:
        std::string nodeName = fmt::format("split_in0_for_{}", m_convNode->getNodeName());
        splitIn0Node         = createSplitNodeForGConvInput(nodeName, in0, curIn0MaxSizes, curIn0MinSizes);
        nodeName             = fmt::format("split_in1_for_{}", m_convNode->getNodeName());
        splitIn1Node         = createSplitNodeForGConvInput(nodeName, in1, curIn1MaxSizes, curIn1MinSizes);

        context.newNodes.push_back(splitIn0Node);
        context.newNodes.push_back(splitIn1Node);
        if (in2)
        {
            nodeName = fmt::format("split_in2_for_{}", m_convNode->getNodeName());
            splitIn2Node =
                createSplitNodeForGConvInput(nodeName, in2, curIn2MaxSizes, curIn2MinSizes, /* isShape */ true);
            context.newNodes.push_back(splitIn2Node);
        }
    }
    in0 = splitIn0Node->getOutput(newGConvIdx);
    in1 = splitIn1Node->getOutput(newGConvIdx);
    if (in2)
    {
        in2 = splitIn2Node->getOutput(newGConvIdx);
    }
}

void GroupedConvolutionPackingManager::addNewGConvNode(PackedGConvContext&  context,
                                                       unsigned            newGConvIdx,
                                                       const TensorVector& inputs,
                                                       const TensorPtr&    output,
                                                       unsigned            origGroupsForCurConv)
{
    std::string nodeName = fmt::format("{}_packed_gconv_{}", m_convNode->getNodeName(), newGConvIdx);
    auto pNewGConvNode = NodeFactory::createNode(inputs, {output}, &context.newParams, m_convNode->getGUID(), nodeName);
    pNewGConvNode->getNodeAnnotation().mmeMetaData.groupPacking = origGroupsForCurConv > 1;
    context.newNodes.push_back(pNewGConvNode);
    context.pOuts.push_back(std::get<3>(context.operands));  // aggregate outputs for the concat node
}

void GroupedConvolutionPackingManager::addGConvFwdAndNewGConv(PackedGConvContext& context,
                                                              unsigned            newGConvIdx,
                                                              unsigned            totalNewGConvNodes,
                                                              unsigned            origGroupsForCurConv,
                                                              std::string_view    dType)
{
    const auto& [in0, in1, in2, out]                       = context.operands;
    const auto& [splitIn0Node, splitIn1Node, splitIn2Node] = context.splitNodes;

    // Create padding TPC node for the weights:
    auto [paddedWghMinSizes, paddedWghMaxSizes] = in1->getAllMinMaxSizesInElements();
    paddedWghMaxSizes[WEIGHT_DIM_C] =
        in1->getSizeInElements(WEIGHT_DIM_C) *
        origGroupsForCurConv;  // TPC kernel "inflates" the C dimension by t=origGroupsForCurConv
    paddedWghMinSizes[WEIGHT_DIM_C] =
        in1->getMinimalSizeInElements(WEIGHT_DIM_C) *
        origGroupsForCurConv;  // TPC kernel "inflates" the C dimension by t=origGroupsForCurConv
    TensorPtr tpcOut = std::make_shared<Tensor>(in1->getDim(),
                                                paddedWghMaxSizes.data(),
                                                in1->getElementType(),
                                                paddedWghMinSizes.data());
    if (GCFG_ENABLE_CONSTANT_FOLDING_OF_GROUP_CONV_FWD_IN_TRAINING.value() && m_convNode->getInput(1)->inConstSection())
    {
        bool weightOtherUses = m_graph.getTensorConsumers(m_convNode->getInput(1)).size() > 1;
        if (newGConvIdx == 0 && weightOtherUses)
        {
            // There are other consumers to the original weights (except the convolution node),
            // so they are expecting that its data is in the const section.
            context.aggConstSectionTensors.push_back(m_convNode->getInput(1));
        }
        // In case splitIn1Node!=nullptr, in1 is the output of the split and its data need to be calculated.
        // The input of the split is static (the weights of the convolution node) so the node can be calculated on the
        // CPU.
        if (splitIn1Node != nullptr && newGConvIdx == 0)
        {
            // Run split node on cpu happens only once.
            HB_ASSERT(splitIn1Node->RunOnCpu(),
                      "Split run on cpu failed, in this case constant folding of gconv_fwd cannot be done. Try "
                      "to compile with ENABLE_CONSTANT_FOLDING_OF_GROUP_CONV_FWD_IN_TRAINING=false");
            for (const auto& splitOutput : splitIn1Node->getOutputs())
            {
                splitOutput->setAsStaticParam();
            }
            context.newNodes.erase(std::remove(context.newNodes.begin(), context.newNodes.end(), splitIn1Node));
        }
        if (!GCFG_ENABLE_RUN_ON_CPU_DUMMY_MODE.value())
        {
            runGConvFwdOnCpu(in1, tpcOut);
        }

        tpcOut->setAsStaticParam();
        context.aggConstSectionTensors.push_back(tpcOut);
        if (newGConvIdx == totalNewGConvNodes - 1)
        {
            ConstSectionReplacer::replace(m_convNode->getInput(1), context.aggConstSectionTensors, !weightOtherUses);
        }
    }
    else
    {
        context.newNodes.push_back(NodeFactory::createNode({in1},
                                                           {tpcOut},
                                                           &m_kPerGroup,
                                                           fmt::format("gconv_fwd_{}", dType),
                                                           fmt::format("pad_wgh_for_{}", m_convNode->getNodeName())));
    }

    // Create the new GConv node:
    TensorVector inputs = {in0, tpcOut};
    if (in2)
    {
        inputs.push_back(in2);
    }
    addNewGConvNode(context, newGConvIdx, inputs, out, origGroupsForCurConv);
}

void GroupedConvolutionPackingManager::addGConvBwdAndNewGConv(PackedGConvContext& context,
                                                              unsigned            newGConvIdx,
                                                              unsigned            origGroupsForCurConv,
                                                              std::string_view    dType)
{
    const auto& [in0, in1, in2, out] = context.operands;
    // Create the new GConv node:
    auto [gconvOutMinSizes, gconvOutMaxSizes] = out->getAllMinMaxSizesInElements();
    gconvOutMaxSizes[WEIGHT_DIM_C]            = out->getSizeInElements(WEIGHT_DIM_C) * origGroupsForCurConv;
    gconvOutMinSizes[WEIGHT_DIM_C]            = out->getMinimalSizeInElements(WEIGHT_DIM_C) * origGroupsForCurConv;

    TensorPtr gconvOut = std::make_shared<Tensor>(out->getDim(),
                                                  gconvOutMaxSizes.data(),
                                                  out->getElementType(),
                                                  gconvOutMinSizes.data());

    // Create unpadding TPC node for the weights:
    context.newNodes.push_back(NodeFactory::createNode({gconvOut},
                                                       {out},
                                                       &m_kPerGroup,
                                                       fmt::format("gconv_bwd_{}", dType),
                                                       fmt::format("unpad_wgh_for_{}", m_convNode->getNodeName())));

    addNewGConvNode(context, newGConvIdx, {in0, in1}, gconvOut, origGroupsForCurConv);
}

void GroupedConvolutionPackingManager::addConcatNode(PackedGConvContext& context, const TensorPtr& out)
{
    LOG_DEBUG(GC, "{}: Creating concat node", HLLOG_FUNC);
    // add concat node for all gConv outputs (2)
    unsigned concatDim = DIM_C;  // for dedw node it's WEIGHT_DIM_K
    NodePtr  concat    = NodeFactory::createNode(context.pOuts,
                                                 {out},
                                             &concatDim,
                                             NodeFactory::concatenateNodeInternalTypeName,
                                             fmt::format("concat_output_for_{}", m_convNode->getNodeName()));
    context.newNodes.push_back(std::move(concat));
}

/**
 * Each grouped node with m_groupsPerVector > 2 is optimized for better MME utilization.
 * The optimization packs several groups to one packed group, so it reduces the number of total groups in the node.
 * The number of groups is reduced from g before the optimization to ~ g/t, where
 * t=floor(mmeVecorSizeInElements/kPerGroup). After the number of groups reduction, the grouped convolution is split to
 * single group convolution nodes (in handleGroupedConvolutions pass). With this optimization, even after the split to
 * single groups - less convolutions, with larger data, will be performed.
 * */
NodeVector GroupedConvolutionPackingManager::packGroupedConvNode()
{
    // use this optimization only when packing is benefitial
    if (m_groupsPerVector < 2)
    {
        LOG_INFO(GC,
                 "{}: Number of groups that fit in a single mme vector is < 2, optimization skipped. node: {}",
                 HLLOG_FUNC,
                 m_convNode->getNodeName());
        return {};
    }

    if (!isValidGConvForPacking())
    {
        LOG_WARN(GC, "Grouped convolution packing is skipped");
        return {};
    }

    TensorPtr in0 = m_convNode->getInput(0);
    TensorPtr in1 = m_convNode->getInput(1);
    TensorPtr in2 = m_convNode->getInput(2);
    TensorPtr out = m_convNode->getOutput(0);

    PackedGConvContext context;
    // keep 2d node params because conv node creation requires 2d
    context.newParams = MmeNode::convert3DconvTo2DconvStruct(m_convNode->getConvolutionParams());

    unsigned totalNewGConvNodes = (m_groupsRemainder > 0 && m_groupsQuotient > 0) ? 2 : 1;
    LOG_DEBUG(GC, "{}: # of new GConv nodes: {}", HLLOG_FUNC, totalNewGConvNodes);

    for (unsigned newGConvIdx = 0; newGConvIdx < totalNewGConvNodes; newGConvIdx++)
    {
        unsigned origGroupsForCurConv = (newGConvIdx == 0) ? m_groupsPerVector : m_groupsRemainder;
        context.newParams.nGroups     = (newGConvIdx == 0) ? m_groupsQuotient : 1;  // new number of groups
        LOG_DEBUG(GC,
                  "{}: newGConvIdx: {}, origGroupsForCurConv: {}, # of packed groups: {}",
                  HLLOG_FUNC,
                  newGConvIdx,
                  origGroupsForCurConv,
                  context.newParams.nGroups);

        // reset operand state
        context.operands = std::make_tuple(in0, in1, in2, out);

        // in case we have a remainder, curIn0 and curIn1 will be split on dim_C.
        // the new gconv will get the split tensor as input, instead of the original tensor.
        if (m_groupsRemainder > 0)
        {
            handleGroupRemainder(/* INOUT */ context, newGConvIdx, origGroupsForCurConv);
        }

        std::string_view dType = getDtypeSuffixFromSynDataType(m_convNode->getWOperand()->getElementType());
        if (m_convNode->getNodeType() == Node::TYPE_CONVOLUTION || m_convNode->getNodeType() == Node::TYPE_DEDX)
        {
            addGConvFwdAndNewGConv(/* INOUT */ context, newGConvIdx, totalNewGConvNodes, origGroupsForCurConv, dType);
        }
        else  // n->getNodeType() == Node::TYPE_DEDW
        {
            addGConvBwdAndNewGConv(/* INOUT */ context, newGConvIdx, origGroupsForCurConv, dType);
        }
    }
    if (m_groupsRemainder > 0)
    {
        // In this case there were created 2 group convolution nodes in the process above,
        // Therefore their outputs should be concatenated to the original group convolution output.
        addConcatNode(/* INOUT */ context, m_convNode->getOutput(0));
    }
    return context.newNodes;
}

bool GroupedConvolutionPackingManager::isSupportedDataType(synDataType type) const
{
    return type == syn_type_bf16 || type == syn_type_float || type == syn_type_fp8_143 || type == syn_type_fp8_152;
}
