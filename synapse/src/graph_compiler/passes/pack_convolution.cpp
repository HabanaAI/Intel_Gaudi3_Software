#include <exception>
#include "conv_base_node.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "node_factory.h"

#include "pack_convolution.h"
#include <memory>
#include "compilation_hal_reader.h"
#include "node_utils.h"
#include "tensor.h"
#include "types.h"

bool ConvolutionPackingManager::isCandidateForPacking(const NodePtr& node) const
{
    // TODO SW-48366: Enable packing for TYPE_DEDX
    return node->getNodeType() == Node::TYPE_CONVOLUTION || node->getNodeType() == Node::TYPE_DEDX;
}

bool ConvolutionPackingManager::packConvolutionNodes()
{
    if (!packingEnabled())
    {
        return true;
    }
    const NodeVector nodes = m_graph.getExeSortedNodes();
    // Choose packing factor
    for (const NodePtr& node : nodes)
    {
        if (isCandidateForPacking(node))
        {
            unsigned packingFactor                                    = choosePackingFactor(node);
            node->getNodeAnnotation().mmeMetaData.packing[PACKING_X] = packingFactor;
            LOG_TRACE(GC, "{}: Set packing factor {} for {}", HLLOG_FUNC, packingFactor, node->getNodeName());
            if (packingFactor == 1) continue;

            // Deal with tensor weights with multiple users
            handleSameTensorDiffPackingFactor(node, packingFactor);
        }
    }
    // Pack the nodes
    for (const NodePtr& node : nodes)
    {
        if (isCandidateForPacking(node))
        {
            if (node->getNodeAnnotation().mmeMetaData.packing[PACKING_X] == 1) continue;
            packConvNode(node);
        }
    }
    return true;
}

bool ConvolutionPackingManager::packingFactorIsValid(unsigned                  packingFactor,
                                                     const TensorPtr&          inputTensor,
                                                     const TensorPtr&          wTensor,
                                                     const TensorPtr&          outputTensor,
                                                     synConvolution3DParamsV2& convParams,
                                                     bool                      isBwd) const
{
    const unsigned kernelWidth           = convParams.kernel[CONV_KERNEL_WIDTH];
    const unsigned strideWidth           = convParams.stride[CONV_STRIDE_WIDTH];
    const unsigned inputWidth            = inputTensor->getSizeInElements(DIM_W);
    const unsigned outputFirstSpatialDim = outputTensor->getSizeInElements(DIM_W);

    if (outputFirstSpatialDim % packingFactor != 0)
    {
        LOG_TRACE(GC,
                  "{}: Packing factor {} doesn't divide output width {}",
                  HLLOG_FUNC,
                  packingFactor,
                  outputFirstSpatialDim);
        return false;
    }

    // The kernel width after packing cannot exceed the spatial dimension of the input
    const unsigned kernelWidthPacked = getKernelWidthAfterPacking(packingFactor, kernelWidth, strideWidth);
    if (kernelWidthPacked > inputWidth)
    {
        LOG_TRACE(GC,
                  "{}: Packing factor {} causes kernel width {} to be larger than input width {}",
                  HLLOG_FUNC,
                  packingFactor,
                  getKernelWidthAfterPacking(packingFactor, kernelWidth, strideWidth),
                  inputWidth);
        return false;
    }

    return true;
}

// A function to calc MME utilization after applying the packing factor
static float calcMmeUtilization(unsigned packingFactor, unsigned kernelWidth, unsigned outputWidthAfterPacking, unsigned interleavingPortNr)
{
    const float packingAcceleration = float(packingFactor) * (float(kernelWidth) / (kernelWidth + packingFactor - 1));
    const float paddingDeceleration = std::min(1.0f, float(outputWidthAfterPacking) / interleavingPortNr);
    return packingAcceleration * paddingDeceleration;
}

unsigned ConvolutionPackingManager::choosePackingFactor(const NodePtr& node) const
{
    const auto convNode                 = std::static_pointer_cast<ConvBaseNode>(node);
    int        tpcLoweringPackingFactor = getTpcLoweringPackingFactor(convNode);
    if (tpcLoweringPackingFactor)
    {
        return tpcLoweringPackingFactor;
    }

    synConvolution3DParamsV2& convParams   = convNode->getConvolutionParams();
    const TensorPtr&          outputTensor = node->getOutput(0);
    // Early conditions to prevent packing
    const TensorPtr& wTensor = convNode->getWOperand();
    const bool       isBwd   = convNode->getNodeType() != Node::TYPE_CONVOLUTION;
    if (shouldBlockPacking(convParams, wTensor, outputTensor, isBwd))
    {
        return 1;
    }

    const NSizeArray& outputSizes     = outputTensor->getAllNSizesInElements();
    const TSize       mmeWidthInElems = m_graph.getHALReader()->getMmeMinimalWidthInElems(wTensor->getElementType());
    // The amount of packing depends on how many times we can fit the output width in the mme (how many spatial elements
    // fit in MME’s width). If K (output channels) is larger than half of MME's minimal width, packing factor will be 0
    // or 1, meaning no packing.
    const TSize widthPackingFactor = std::max(mmeWidthInElems / outputSizes[DIM_C], static_cast<TSize>(1));
    // Limit the packing factor by dim W of the output height, as it is the outer dimension which is reshaped in
    // packing.
    unsigned packingFactor = std::min(outputSizes[DIM_W], widthPackingFactor);
    if (packingFactor == 1) return 1;

    const TensorPtr& inputTensor = node->getInput(0);

    // Select the highest packing factor such that it prevent padding spatial dim when output[W] < spatial ports number.
    // For more details go to MME stack project and check RecipeGenerator::padSpatialDim().
    // Make sure the packing factor divides dim W of the output height, for a valid reshape, or search for a valid
    // factor. In addition, make sure that packing factor does not make the kernel width larger than the input width.
    if (GCFG_ENABLE_PACKING_FACTOR_COST_FUNCTION.value())
    {
        const HalReader& halReader             = *m_graph.getTraits().getHalReader();
        const unsigned   interleavingPortNr    = halReader.getMmeMaxInterleavedSpatialPortsNr();
        const unsigned   outputFirstSpatialDim = outputTensor->getSizeInElements(DIM_W);
        const unsigned   kernelWidth           = convParams.kernel[CONV_KERNEL_WIDTH];
        unsigned         curPackingFactor      = packingFactor;
        float            prevMmeUtilization    = 0;
        bool             isPackingFactorValid  = false;
        while (curPackingFactor > 1)
        {
            if (packingFactorIsValid(curPackingFactor, inputTensor, wTensor, outputTensor, convParams, isBwd))
            {
                isPackingFactorValid                   = true;
                const unsigned outputWidthAfterPacking = outputFirstSpatialDim / curPackingFactor;
                const float    curMmeUtilization =
                    calcMmeUtilization(curPackingFactor, kernelWidth, outputWidthAfterPacking, interleavingPortNr);
                if (curMmeUtilization <= prevMmeUtilization)
                {
                    break;
                }
                packingFactor      = curPackingFactor;
                prevMmeUtilization = curMmeUtilization;
            }
            curPackingFactor--;
        }
        return isPackingFactorValid ? packingFactor : 1;
    }
    // Else (original logic):
    // Use optimal packing factor without considering any other features (as padding spatial dim)
    while ((packingFactor > 1) &&
           !packingFactorIsValid(packingFactor, inputTensor, wTensor, outputTensor, convParams, isBwd))
    {
        packingFactor--;
    }
    return packingFactor;
}

unsigned ConvolutionPackingManager::getKernelWidthAfterPacking(unsigned packingFactor,
                                                               unsigned kernelWidth,
                                                               unsigned strideWidth) const
{
    return kernelWidth + strideWidth * (packingFactor - 1);
}
unsigned ConvolutionPackingManager::getStrideWidthAfterPacking(unsigned packingFactor, unsigned strideWidth) const
{
    return strideWidth * packingFactor;
}

// Early check version that does not depend on packing factor
bool ConvolutionPackingManager::shouldBlockPacking(const synConvolution3DParamsV2& convParams,
                                                   const TensorPtr&                wTensor,
                                                   const TensorPtr&                outputTensor,
                                                   bool                            isBwd) const
{
    // 4-bit datatype is not relevant for Gaudi, so getAllSizesInElements and getAllSizesInElementsCondensed are
    // equivalent
    if (outputTensor->isType4Bit() || wTensor->isType4Bit())
    {
        return true;
    }

    const NSizeArray& outputSizes                  = outputTensor->getAllNSizesInElements();
    const HalReader&  halReader                    = *m_graph.getTraits().getHalReader();
    const unsigned    maxInterleavedSpatialPortsNr = halReader.getMmeMaxInterleavedSpatialPortsNr();
    // If the original height is smaller than the amount of interleaving ports we are already padding,
    // and packing will increase the amount of padding by 50% while the possible acceleration gains from
    // packing is always less than 50% in this case we cannot pack at all.
    if (outputSizes[DIM_W] < maxInterleavedSpatialPortsNr)
    {
        return true;
    }

    if ((convParams.dilation[CONV_DIL_WIDTH] != 1) || (convParams.dilation[CONV_DIL_HEIGHT] != 1) ||
        (convParams.dilation[CONV_DIL_DEPTH] != 1))
    {
        return true;  // can't pack with dilation
    }

    const unsigned kernelW = convParams.kernel[CONV_KERNEL_WIDTH];
    const unsigned strideW = convParams.stride[CONV_STRIDE_WIDTH];
    // no packing is done if strideS > s, as the data is not contiguous
    if (strideW > kernelW)
    {
        return true;
    }

    return false;
}

void ConvolutionPackingManager::reshapeOutputTensor(const TensorPtr& t, unsigned packingFactor)
{
    SizeArray maxSizes, minSizes;
    t->getAllSizesInElements(maxSizes);
    updatePackedOutputSizes(maxSizes, packingFactor);
    t->getAllMinimalSizesInElements(minSizes);
    updatePackedOutputSizes(minSizes, packingFactor);
    t->reshape(t->getDim(), maxSizes.data(), nullptr, minSizes.data());
}

void ConvolutionPackingManager::updatePackedOutputSizes(SizeArray& sizes, unsigned packingFactor)
{
    HB_ASSERT(sizes[DIM_W] % packingFactor == 0, "tensor W dimension size is not divisible by packing value");
    /* each output line now contains packingX outputs */
    sizes[DIM_C] = sizes[DIM_C] * packingFactor;
    /* the extra outputs per line is reduced from W dim */
    sizes[DIM_W] = sizes[DIM_W] / packingFactor;
}

void ConvolutionPackingManager::packOutput(const NodePtr& node, const TensorPtr& outTensorOrig, unsigned packingFactor)
{
    TensorPtr outTensorPacked = outTensorOrig->clone();
    outTensorPacked->setName(outTensorOrig->getName() + "_packed");

    reshapeOutputTensor(outTensorPacked, packingFactor);
    outTensorPacked->getTensorAnnotation().dataInfo.packing[PACKING_X] = packingFactor;
    // Insert the reshape operations into the graph to connect the original inputs/outputs to the new ones.
    // Using static reshape because the reshaped dimensions are not dynamic (otherwise packing is blocked, by
    // shouldBlockPacking)
    m_reshapeNodeOfm = NodeFactory::createNode({outTensorPacked},
                                               {outTensorOrig},
                                               nullptr,
                                               NodeFactory::staticReshapeNodeTypeName,
                                               outTensorOrig->getName() + "_reshape");
}

void ConvolutionPackingManager::packConvNode(const NodePtr& node)
{
    unsigned packingFactor = node->getNodeAnnotation().mmeMetaData.packing[PACKING_X];
    // pack convolution node - modify the convolution itself to reflect the packing
    auto                      convNode   = std::static_pointer_cast<ConvBaseNode>(node);
    synConvolution3DParamsV2& convParams = convNode->getConvolutionParams();
    unsigned                  convStride = convParams.stride[CONV_STRIDE_WIDTH];
    convParams.kernel[CONV_KERNEL_WIDTH] = getKernelWidthAfterPacking(packingFactor,
                                                                      convParams.kernel[CONV_KERNEL_WIDTH],
                                                                      convParams.stride[CONV_KERNEL_WIDTH]);
    convParams.stride[CONV_KERNEL_WIDTH] =
        getStrideWidthAfterPacking(packingFactor, convParams.stride[CONV_KERNEL_WIDTH]);
    const TensorPtr& wTensor      = convNode->getWOperand();
    const TensorPtr& outputTensor = convNode->getOutput(0);

    packWeights(convNode, wTensor, convStride, packingFactor);
    packOutput(node, outputTensor, packingFactor);
    applyChangeToGraph(node);
    resetTemporaryState();
}
