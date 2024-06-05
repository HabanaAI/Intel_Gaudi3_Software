#include "conv_base_node.h"

#include "access_pattern_generator.h"
#include "habana_graph.h"
#include "hal_reader/hal_reader.h"
#include "infra/cpu_calculator.h"
#include "node_factory.h"
#include "sif/shape_inference_metadata.h"
#include "synapse_types_operators.h"
#include "tensor_shape.h"
#include "utils.h"

ConvBaseNode::ConvBaseNode(const TensorVector& inputs,
                           const TensorVector& outputs,
                           std::string_view    name,
                           Node::eNodeType     type,
                           ShapeFuncID         sifId)
: MmeNode(inputs, outputs, name, type, sifId), m_sifMetadata()
{
}

synConvolution3DParamsV2 ConvBaseNode::convertUserParams(UserParams userParams, size_t userParamsSize) const
{
    if (userParamsSize == sizeof(synConvolutionParams))
    {
        synConvolutionParamsV2 userConvParam = *reinterpret_cast<synConvolutionParams*>(userParams);  // up-convert
        return MmeNode::convert2DconvTo3DconvStruct(userConvParam);
    }
    else if (userParamsSize == sizeof(synConvolutionParamsV2))
    {
        return MmeNode::convert2DconvTo3DconvStruct(*reinterpret_cast<synConvolutionParamsV2*>(userParams));
    }
    else if (userParamsSize == sizeof(synConvolution3DParams))
    {
        return *reinterpret_cast<synConvolution3DParams*>(userParams);  // up-convert
    }
    else if (userParamsSize == sizeof(synConvolution3DParamsV2))
    {
        return *reinterpret_cast<synConvolution3DParamsV2*>(userParams);
    }
    else  // illegal size. TODO - [SW-119095]
    {
        LOG_WARN(HABANA_NODE, "conv node: illegal param for size of {} for node {}", userParamsSize, m_name);
        if (is3DConvolutionGuid())
        {
            return *reinterpret_cast<synConvolution3DParams*>(userParams);
        }
        else  // 2d conv
        {
            return MmeNode::convert2DconvTo3DconvStruct(*reinterpret_cast<synConvolutionParams*>(userParams));
        }
    }
}

void ConvBaseNode::setParams(void* userParams, unsigned userParamsSize)
{
    m_params = convertUserParams(userParams, userParamsSize);
    LOG_TRACE(HABANA_NODE,
              "{} name - {}, params - {}",
              getGUID(),
              m_name,
              MmeNode::synConvolution3DParamsToString(m_params));
    m_originalParams = m_params;
    // nGroups cannot be set to 0
    if (m_params.nGroups == 0)
    {
        m_params.nGroups = 1;
    }
}

bool ConvBaseNode::equalTo(const Node& o) const
{
    if (getNodeType() != o.getNodeType()) return false;
    const ConvBaseNode* pO = dynamic_cast<const ConvBaseNode*>(&o);
    if (nullptr == pO) return false; //Not a convolution
    if (! (getConvolutionParams() == pO->getConvolutionParams())) return false;
    return Node::equalTo(o);
}

void ConvBaseNode::print() const
{
    Node::print();
    MmeNode::printMmeParams(m_params);
}

std::string ConvBaseNode::getNodeParametersStr() const
{
    return MmeNode::synConvolution3DParamsToString(m_params);
}

void ConvBaseNode::printParamsRawData() const
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(GRAPH_DATA)) return;

    if (!is3DConvolution())
    {
        synConvolutionParamsV2 params(m_params.kernel[CONV_KERNEL_WIDTH],
                                      m_params.kernel[CONV_KERNEL_HEIGHT],
                                      m_params.stride[CONV_STRIDE_WIDTH],
                                      m_params.stride[CONV_STRIDE_HEIGHT],
                                      m_params.padding[CONV_PAD_LEFT],
                                      m_params.padding[CONV_PAD_RIGHT],
                                      m_params.padding[CONV_PAD_TOP],
                                      m_params.padding[CONV_PAD_BOTTOM],
                                      m_params.dilation[CONV_DIL_WIDTH],
                                      m_params.dilation[CONV_DIL_HEIGHT],
                                      m_params.paddingType);
        params.nGroups    = m_originalParams.nGroups;
        params.activation = m_originalParams.activation;
        BaseClass::printParamsRawData((void*)&params, sizeof(params));
    }
    else
    {
        BaseClass::printParamsRawData((void*)&m_params, sizeof(m_params));
    }
}

bool ConvBaseNode::validateNode() const
{
    if (getWOperand()->isDynamicShape())
    {
        LOG_ERR(HABANA_NODE, "Dynamic weights are not supported");
        return false;
    }
    return BaseClass::validateNode();
}

bool ConvBaseNode::validateNodeLayout() const
{
    SET_TEMP_LOG_CONTEXT(getNodeName());

    bool ret = MmeNode::validateNodeLayout();

    // Validate convolution size only for non extracted nodes
    // Extracted nodes are created from graph compiler multi nodes
    // If the conv is lowered the validation is incorrect due to
    // reshape params not fully updated in the conv node
    if (ret &&
        ! getNodeAnnotation().isExtracted &&
        ! getWOperand()->isLowered())
    {
        ret = validateConvolutionSize(getXOperand()->getAllSizesInElements(),
                                      getWOperand()->getAllSizesInElements(),
                                      getYOperand()->getAllSizesInElements(),
                                      getYOperand()->getDim(),
                                      m_params);
        ret &= validateConvPadding(getXOperand()->getAllSizesInElements(),
                                   getWOperand()->getAllSizesInElements(),
                                   getYOperand()->getAllSizesInElements(),
                                   getYOperand()->getDim(),
                                   m_params);
    }

    return ret;
}

bool ConvBaseNode::is3DConvolution() const
{
    TensorPtr ifm_tensor = getInput(TENSOR_IFM);
    if (ifm_tensor != nullptr)
    {
        unsigned dims = ifm_tensor->getDim();
        if (dims == CONV_3D_TENSOR_DIM)
        {
            return true;
        }
        return false;
    }
    LOG_ERR(HABANA_NODE, "{} There is no IFM tensor", HLLOG_FUNC);
    return false;

}

bool ConvBaseNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return BaseClass::validateNodeForGraph(g);
}

bool ConvBaseNode::canBeConvertedToGEMM() const
{
    // channels and outer dim can be dynamic.Dynamicity can be checked based on X operand (IFM)
    auto xOperand = getXOperand();
    auto firstDynamicDim = xOperand->getFirstDynamicDimIndex(1);

    if (firstDynamicDim && (*firstDynamicDim) != (xOperand->getDim() - 1))
    {
        LOG_WARN(GC, "Can't flatten MME node {}", getNodeName());
        return false;
    }

    return areConvParamsGEMMConvertible(getConvolutionParams());
}

SifNodeParams ConvBaseNode::getShapeInferenceFunctionUserParams()
{
    m_sifMetadata.params = m_params;
    // SIF should keep the original after-padding
    for(size_t dim = DIM_W; dim <= DIM_D_FOR_5D_TENSOR; dim++)
    {
        ConvParamsIndices convIdx = dimIndexToConvParamsIndices(dim);
        auto paddingAfterIndex = convIdx.paddingAfterIndex;
        m_sifMetadata.params.padding[paddingAfterIndex] = m_originalParams.padding[paddingAfterIndex];
    }
    getOutput(TENSOR_OFM)->getAllSizesInElements(m_sifMetadata.maxOutputSizes, SYN_MAX_TENSOR_DIM);
    m_sifMetadata.maxOutputSizesKnown = !requiresOutputMaxDimInfer();
    return reinterpret_cast<SifNodeParams>(&m_sifMetadata);
}

size_t ConvBaseNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(SifConvolutionMetadata);
}

// Update convolution params according to the slice position in the original tensor
// Set the middle slices padding to be the value requested by the user
void ConvBaseNode::updateConvForXOperandROI(ROIPosition& roiPos, const OffsetArray& padBefore, const OffsetArray& padAfter)
{
    updateDimPaddingByPosition(DIM_W, roiPos, padBefore, padAfter);
    updateDimPaddingByPosition(DIM_H, roiPos, padBefore, padAfter);
    if (is3DConvolution())
    {
        updateDimPaddingByPosition(DIM_D_FOR_5D_TENSOR, roiPos, padBefore, padAfter);
    }
}

void ConvBaseNode::updateDimPaddingByPosition(unsigned dim, const ROIPosition& pos, const OffsetArray& padBefore, const OffsetArray& padAfter)
{
    if (pos.isFirst[dim] && pos.isLast[dim])
    {
        //the slice is the entire dim - do nothing.
        return;
    }

    ConvParamsIndices convIdx = dimIndexToConvParamsIndices(dim);

    if (!pos.isFirst[dim])
    {
        // remove before padding, and set to the value requested by the caller
        m_params.padding[convIdx.paddingBeforeIndex] = padBefore[dim];
    }
    if (!pos.isLast[dim])
    {
        // remove after padding, and set to the value requested by the caller
        m_params.padding[convIdx.paddingAfterIndex] = padAfter[dim];
    }
}

TensorShape ConvBaseNode::getXOperandShape(const TensorShape& yOperandShape) const
{
    const unsigned int dim = yOperandShape.getDim();
    TensorShape xOperandShape;
    xOperandShape.setDim(dim);

    // Calculate Base coordinate
    CoordArray baseCoord = yOperandShape.getBases();
    CoordArray endCoord;
    for (unsigned dim=0; dim<SYN_MAX_TENSOR_DIM; dim++)
    {
        endCoord[dim] = baseCoord[dim] + yOperandShape.getSize(dim) - 1;
    }

    int startKernelIndex[] = {0, 0, 0};
    convertToInputCoord(baseCoord.data(), m_params, startKernelIndex, dim);
    // Return the base not clipped to input tensor size
    // In case of padding in the boundaries of the tensor convolution params need to be changed
    xOperandShape.setBase(baseCoord);

    // Calculate sizes
    int endKernelIndex[] = {m_params.kernel[CONV_KERNEL_WIDTH] - 1, m_params.kernel[CONV_KERNEL_HEIGHT] - 1,
                            m_params.kernel[CONV_KERNEL_DEPTH] - 1};
    convertToInputCoord(endCoord.data(), m_params, endKernelIndex, dim);

    SizeArray sizes;
    for (unsigned dim=0; dim<SYN_MAX_TENSOR_DIM; dim++)
    {
        sizes[dim] = endCoord[dim] + 1 - baseCoord[dim];
    }
    sizes[0] = getXOperand()->getSizeInElements(0);
    // Return sizes not clipped
    xOperandShape.setSize(sizes);

    return xOperandShape;
}

TensorShape ConvBaseNode::getYOperandShape(const TensorShape& xOperandShape) const
{
    const unsigned int numDims = xOperandShape.getDim();
    TensorShape yOperandShape;
    yOperandShape.setDim(numDims);

    // Calculate Base coordinate
    CoordArray baseCoordX = xOperandShape.getBases();
    CoordArray endCoordX;
    for (unsigned dim=0; dim<SYN_MAX_TENSOR_DIM; dim++)
    {
        endCoordX[dim] = baseCoordX[dim] + xOperandShape.getSize(dim) - 1;
    }
    CoordArray baseCoordY = baseCoordX;
    CoordArray endCoordY = endCoordX;

    // To get the full Y slice, which is affected by the given X slice - the filter should be positioned
    // with max valid index on the first X coord, and with min valid index on the last X coord.
    // A valid index means that the kernel positioning, due to stride and dilation spacing, aligns this
    // kernel index with the X coord, and produces integer Y coordinate when mapping to output coord.
    int endKernelIndex[] = {m_params.kernel[CONV_KERNEL_WIDTH] - 1, m_params.kernel[CONV_KERNEL_HEIGHT] - 1,
                            m_params.kernel[CONV_KERNEL_DEPTH] - 1};
    convertToOutputCoord(baseCoordY.data(), m_params, endKernelIndex, numDims, true /*ceil*/);
    int startKernelIndex[] = {0, 0, 0};
    convertToOutputCoord(endCoordY.data(), m_params, startKernelIndex, numDims, false /*floor*/);

    yOperandShape.setBase(baseCoordY);
    SizeArray sizes;
    for (unsigned dim=0; dim<SYN_MAX_TENSOR_DIM; dim++)
    {
        sizes[dim] = endCoordY[dim] + 1 - baseCoordY[dim];
    }

    // Return sizes not clipped
    sizes[0] = getYOperand()->getSizeInElements(0);
    yOperandShape.setSize(sizes);
    return yOperandShape;
}

int ConvBaseNode::getInputROIOverlapForDim(uint32_t inputIdx, unsigned dim) const
{
    int overlap = 0;
    if (getInput(inputIdx) == getXOperand())
    {
        overlap = getXOverlapForDim(dim);
    }
    else if (getInput(inputIdx) == getYOperand())
    {
        overlap = getYOverlapForDim(dim);
    }
    return overlap;
}

int ConvBaseNode::getXOverlapForDim(unsigned dim) const
{
        ConvParamsIndices convId = dimIndexToConvParamsIndices(dim);
        // overlap = filter last line offset - (stride-1) [the last line already belongs to the current slice, stride-1 line are skipped between kernel activations]
        //           (kernel_size-1) * dilation - (stride-1)
        int overlap = static_cast<int>((m_params.kernel[convId.spatialIndex] - 1) * m_params.dilation[convId.spatialIndex] - (m_params.stride[convId.spatialIndex] - 1));
        return overlap;
}

int ConvBaseNode::getYOverlapForDim(unsigned dim) const
{
        ConvParamsIndices convId = dimIndexToConvParamsIndices(dim);
        // Try to position the kernel as far "back" as possible on the first computed x coordinate.
        // The only slice-size independent coordinate is the first slice first coord, which is 0 or the first padding line if exists.
        // Check which largest kernel index provides a valid (integer) y coordinate.
        // This is the way to position the kernel on the first x slice coord such that maximal number of y pixels are influenced.
        bool yFound = false;
        int minCoordX = -m_params.padding[convId.paddingBeforeIndex];
        int minCoordY = 0, kernelIdx = 0, dilOffset = 0;
        yFound = getMinValidYCoordForDim(dim, minCoordX, minCoordY, kernelIdx, dilOffset);
        HB_ASSERT(yFound, "start of tensor didn't find a valid kernel activation, though aligned with stride");

        // kernelIdx is the kernel index which provides the earliest influenced y coordinate by the beginning of the slice.
        // calculate the overlap accordingly.
        // overlap =  ((kernel_index) * dilation + stride - (1 + dil_offset)) / stride
        int overlap = static_cast<int>((kernelIdx * m_params.dilation[convId.spatialIndex] + m_params.stride[convId.spatialIndex] - (1 + dilOffset)) /
                                        static_cast<double>(m_params.stride[convId.spatialIndex]));
        return overlap;
}

// Find the min yCoord, for which xCoord is in the bounds of a kernel activation. xCoors is not necessarily aligned to a kernel
// index in this minimal y calc, due to stride and dilation gaps. However, this is a lower bound to an ROI starting with this xCoord.
// Return true if a valid yCoord was found, or false if no valid coord can be found.
// No valid yCoord means xCoord is in a gap between kernel activations due to large stride.
// If xCoord is a multiplication of stride - it is promised that at least kernel index 0 is positioned on it.
// Return by ref the kernel index and the dilation offset that were used to get this y coord.
bool ConvBaseNode::getMinValidYCoordForDim(unsigned dim, int xCoord, int& yCoord, int& validKernelIndex, int& dilOffset) const
{
    // Try to position the kernel as far "back" as possible on xCoord: Find the highest kernel index, which is a valid
    // application of the kernel on xCoord. xCoord might not be positioned directly on a kernel index, due to dilation.
    // In this case find the dilation offset from xCoord to the first valid coord.
    ConvParamsIndices convId = dimIndexToConvParamsIndices(dim);

    // Check which largest kernel index provides a valid (integer) y coordinate.
    for (int kernelIdx = m_params.kernel[convId.spatialIndex] - 1; kernelIdx >= 0; kernelIdx--)
    {
        // In case of dilation > 1 - the coord might fall on the skipped lines.
        // For each kernel index (kernel positioning) - try to shift xCoord in the range of coord..coord+dil-1
        // to find coord which is not skipped, thus producing a valid y.
        // Block dilation search in case of kernel size 1, as dilation is meaningless
        // Block dilation search if reached kernel index 0, as shifting the kernel forward no longer includes the original xCoord.
        int maxDilation = (m_params.kernel[convId.spatialIndex] == 1 || kernelIdx == 0) ? 1 : m_params.dilation[convId.spatialIndex];
        for (int dil = 0; dil < maxDilation; dil++)
        {
            int coord = xCoord + dil;
            bool isIntOutputCoord = false;
            yCoord = convertToOutputCoordForDim(coord, m_params, kernelIdx, dim, true /*ceil*/, &isIntOutputCoord);
            if (isIntOutputCoord) // y coord is integer, thus this is a valid positioning of the kernel
            {
                validKernelIndex = kernelIdx;
                dilOffset = dil;
                return true;
            }
        }
    }
    // xCoord is not included in any kernel activation.
    return false;
}

// Find the max yCoord, for which xCoord is in the bounds of a kernel activation. xCoord is not necessarily aligned to a kernel
// index due to stride and dilation gaps. However, this is an upper bound to an ROI ending with this xCoord.// Return true if a valid yCoord was found, or false if no valid coord can be found.
// No valid yCoord means xCoord is in a gap between kernel activations due to large stride.
// If xCoord is a multiplication of stride - it is promised that at least kernel index 0 is positioned on it.
// Return by ref the kernel index and the dilation offset that were used to get this y coord.
bool ConvBaseNode::getMaxValidYCoordForDim(unsigned dim, int xCoord, int& yCoord, int& validKernelIndex, int& dilOffset) const
{
    // Try to position the kernel as far "forward" as possible on xCoord: Find the smallest kernel index on xCoord, which is a valid
    // application of the kernel on xCoord. xCoord might not be positioned directly on a kernel index, due to dilation.
    // In this case find the dilation offset from xCoord to the previous valid coord.
    ConvParamsIndices convId = dimIndexToConvParamsIndices(dim);

    // Check which smallest kernel index provides a valid (integer) y coordinate.
    for (int kernelIdx = 0;  kernelIdx < m_params.kernel[convId.spatialIndex]; kernelIdx++)
    {
        // In case of dilation > 1 - the coord might fall on the skipped lines.
        // For each checked kernel index (kernel positioning) - try to shift xCoord in the range of coord -> coord-(dil-1)
        // to find coord which is not skipped, thus producing a valid y.
        // Block dilation search if we reached the last kernel index, as shifting the kernel backwards no longer includes the original xCoord.
        // This also holds for kernel size 1.
        int maxDilation = (kernelIdx == m_params.kernel[convId.spatialIndex] - 1) ? 1 : m_params.dilation[convId.spatialIndex];
        for (int dil = 0; dil < maxDilation; dil++)
        {
            int coord = xCoord - dil;
            bool isIntOutputCoord = false;
            yCoord = convertToOutputCoordForDim(coord, m_params, kernelIdx, dim, false /*ceil*/, &isIntOutputCoord);
            if (isIntOutputCoord) // y coord is integer, thus this is a valid positioning of the kernel
            {
                validKernelIndex = kernelIdx;
                dilOffset = dil;
                return true;
            }
        }
    }
    // xCoord is not included in any kernel activation.
    return false;
}

// Return the padding for ROI of size xRoiSize, which begins in a stride aligned coordinate.
// Y operand ROI, which has the input shape derived from X operand ROI, includes all coords required to calculate
// the X ROI, but influences more X coordinates. To make the operation correct between the two ROIs, the MME has
// to get the influenced X coords external to the ROI as padding, and ignore their values eventually.
// Note: this function can be generalized to non stride aligned ROI by taking the start coordinate.
void ConvBaseNode::getXStrideAlignedROIPaddingForDim(unsigned dim, unsigned xRoiSize, int& padBefore, int& padAfter) const
{
    ConvParamsIndices convId = dimIndexToConvParamsIndices(dim);
    // Calc the index of the line in the kernel that is positioned on the beginning of the slice.
    // The index is equal to the num of lines before it in the kernel. This is the required padding before.
    // padBefore = kernelIdxStart * m_params.dilation - dilOffsetStart;
    int baseCoordX = -m_params.padding[convId.paddingBeforeIndex];
    int kernelIdxStart = 0, dilOffsetStart = 0, baseCoordY = 0;
    bool yFound = false;
    yFound = getMinValidYCoordForDim(dim, baseCoordX, baseCoordY, kernelIdxStart, dilOffsetStart);
    HB_ASSERT(yFound, "Invalid shape start - need to align shape base with X line that is computed");
    padBefore = kernelIdxStart * m_params.dilation[convId.spatialIndex] - dilOffsetStart;

    // Calc the index of the line in the kernel that is positioned on the last coord of the slice.
    // The index + 1 is the number of lines from the kernel that are part of this slice.
    // The total kernel size minus those lines is the required padding after.
    // padAfter = getDimActualKernelSize() - (kernelIdxEnd * m_params.dilation + dilOffsetEnd + 1);
    int endCoordX = baseCoordX + xRoiSize - 1;
    int kernelIdxEnd = 0, dilOffsetEnd = 0, endCoordY = 0;
    yFound = getMaxValidYCoordForDim(dim, endCoordX, endCoordY, kernelIdxEnd, dilOffsetEnd);
    if (yFound)
    {
        padAfter = getDimActualKernelSize(dim) - (kernelIdxEnd * m_params.dilation[convId.spatialIndex] + dilOffsetEnd + 1);
    }
    else
    {
        // There are coordinates at the end of the slice which aren't computed (due to stride).
        // Instead of negative padding keep 0 padding, to give a chance to the MME to zero them.
        padAfter = 0;
    }
}

ConvParamsIndices ConvBaseNode::dimIndexToConvParamsIndices(unsigned tensorDim)
{
    HB_ASSERT((tensorDim >= DIM_W && tensorDim <= DIM_D_FOR_5D_TENSOR), "Invalid spatial dim");
    ConvParamsIndices ret = {CONV_MAX_SPATIAL_INDEX, 0, 0};
    // tensor dim starts counting from FCD, and conv params array are only for the spatial dims.
    // set the offset between the 2 enums
    ret.spatialIndex = static_cast<eConvSpatialIndex>(tensorDim - DIM_W); // DIM_W = 1, DIM_H = 2, DIM_D_FOR_5D_TENSOR = 3
    // the padding array is set as padding before, padding after per spatial index,
    // so the matching padding before is in index 2*spatialIndex and the padding after is in index 2*spatialIndex+1
    ret.paddingBeforeIndex = 2 * ret.spatialIndex; // CONV_PAD_LEFT = 0, CONV_PAD_TOP = 2, CONV_PAD_FRONT = 4
    ret.paddingAfterIndex = 2 * ret.spatialIndex + 1; // CONV_PAD_RIGHT = 1, CONV_PAD_BOTTOM = 3, CONV_PAD_BACK = 5

    return ret;
}

// input_idx = output_idx * stride + kernel_idx * dilation - padding
void ConvBaseNode::convertToInputCoord(int*                            coord,
                                       const synConvolution3DParamsV2& convParams,
                                       int                             kernelIndex[MAX_CONV_DIMS],
                                       unsigned int                    numDims)
{
    coord[DIM_W] = convertToInputCoordForDim(coord[DIM_W], convParams, kernelIndex[dimIndexToConvParamsIndices(DIM_W).spatialIndex], DIM_W);
    coord[DIM_H] = convertToInputCoordForDim(coord[DIM_H], convParams, kernelIndex[dimIndexToConvParamsIndices(DIM_H).spatialIndex], DIM_H);
    if (numDims == CONV_3D_TENSOR_DIM)
    {
        coord[DIM_D_FOR_5D_TENSOR] =
            convertToInputCoordForDim(coord[DIM_D_FOR_5D_TENSOR], convParams, kernelIndex[dimIndexToConvParamsIndices(DIM_D_FOR_5D_TENSOR).spatialIndex], DIM_D_FOR_5D_TENSOR);
    }
    coord[DIM_C] = 0;
}

int ConvBaseNode::convertToInputCoordForDim(int                             coord,
                                            const synConvolution3DParamsV2& convParams,
                                            int                             kernelIndex,
                                            unsigned int                    tensorDim)
{
    ConvParamsIndices convId = dimIndexToConvParamsIndices(tensorDim);
    // input_idx = output_idx * stride + kernel_idx * dilation - padding
    int inputCoord = coord * convParams.stride[convId.spatialIndex] +
                     kernelIndex * convParams.dilation[convId.spatialIndex] - convParams.padding[convId.paddingBeforeIndex];
    return inputCoord;
}

// output_index = (input_index - kernel_index * dilation + padding) / stride
// returns the floor value in case of non integer ouput index
void ConvBaseNode::convertToOutputCoord(int*                            coord,
                                        const synConvolution3DParamsV2& convParams,
                                        int                             kernelIndex[MAX_CONV_DIMS],
                                        unsigned int                    numDims,
                                        bool                            ceil)
{
    coord[DIM_W] = convertToOutputCoordForDim(coord[DIM_W], convParams, kernelIndex[dimIndexToConvParamsIndices(DIM_W).spatialIndex], DIM_W, ceil);
    coord[DIM_H] = convertToOutputCoordForDim(coord[DIM_H], convParams, kernelIndex[dimIndexToConvParamsIndices(DIM_H).spatialIndex], DIM_H, ceil);
    if (numDims == CONV_3D_TENSOR_DIM)
    {
        coord[DIM_D_FOR_5D_TENSOR] =
            convertToOutputCoordForDim(coord[DIM_D_FOR_5D_TENSOR], convParams, kernelIndex[dimIndexToConvParamsIndices(DIM_D_FOR_5D_TENSOR).spatialIndex], DIM_D_FOR_5D_TENSOR, ceil);
    }
    coord[DIM_C] = 0;
}

// Return the floor or ceil value of the requested coordinate, according to ceil input param.
// If it's not int it's an invalid kernel index and input index combination based on the conv params.
// The caller can use pIsIntOutputCoord to get indication if the coord was int and is valid.
// TODO: handle paddingType here! <===== PADDING_TYPE

int ConvBaseNode::convertToOutputCoordForDim(int                             coord,
                                             const synConvolution3DParamsV2& convParams,
                                             int                             kernelIndex,
                                             unsigned int                    tensorDim,
                                             bool                            ceil,
                                             bool*                           pIsIntOutputCoord)
{
    ConvParamsIndices convId = dimIndexToConvParamsIndices(tensorDim);
    // output_index = (input_index - kernel_index * dilation + padding) / stride
    // if the coordinate isn't int, it's floored. If the user requested ceil - the remainder is checked to apply ceil.
    double realOutputCoord = (static_cast<int>((coord - kernelIndex * convParams.dilation[convId.spatialIndex] + convParams.padding[convId.paddingBeforeIndex])) /
                              static_cast<double>(convParams.stride[convId.spatialIndex]));
    int outputCoord = std::floor(realOutputCoord);
    bool isIntOutputCoord = (outputCoord == realOutputCoord);
    if (ceil && !isIntOutputCoord)
    {
        outputCoord++;
    }
    if (pIsIntOutputCoord)
    {
        *pIsIntOutputCoord = isIntOutputCoord;
    }
    return outputCoord;
}

// Get actual kernel size including dilation
unsigned ConvBaseNode::getDimActualKernelSize(unsigned dim) const
{
    ConvParamsIndices convIdx = ConvBaseNode::dimIndexToConvParamsIndices(dim);
    unsigned kernelSize = m_params.kernel[convIdx.spatialIndex];
    unsigned dilation = m_params.dilation[convIdx.spatialIndex];
    return (kernelSize + (kernelSize - 1) * (dilation - 1));
}

bool ConvBaseNode::isDynamicPaddingConvolution() const
{
    if (!isDynamicShape()) return false;
    if (m_params.paddingType != PADDING_SAME) return false;

    if (m_params.stride[CONV_STRIDE_WIDTH ] != 1 && m_params.kernel[CONV_KERNEL_WIDTH ] != 1) return true;
    if (m_params.stride[CONV_STRIDE_HEIGHT] != 1 && m_params.kernel[CONV_KERNEL_HEIGHT] != 1) return true;
    if (m_params.stride[CONV_STRIDE_DEPTH ] != 1 && m_params.kernel[CONV_KERNEL_DEPTH ] != 1) return true;

    return false;
}

DimVector ConvBaseNode::getSpatialDims() const
{
    DimVector ret;
    ret.push_back(DIM_W);
    ret.push_back(DIM_H);
    if (is3DConvolution())
    {
        ret.push_back(DIM_D_FOR_5D_TENSOR);
    }
    return ret;
}

gc::access_pattern::NodeAccessPatternPtr ConvBaseNode::generateNodeAccessPattern() const
{
    return gc::access_pattern::AccessPatternMmeNodeGenerator::generate(this);
}

bool ConvBaseNode::isSpatialSlicingSupported(unsigned dim) const
{
    ConvParamsIndices convIdx = ConvBaseNode::dimIndexToConvParamsIndices(dim);

    if (m_params.paddingType == PADDING_SAME)
    {
        // TODO Handle this case, JIRA SW-88249
        LOG_TRACE(SRAM_SLICE, "{}: Spatial slicing is required, but padding type is SAME", HLLOG_FUNC);
        return false;
    }

    if (m_params.padding[convIdx.paddingAfterIndex] >= getDimActualKernelSize(dim))
    {
        LOG_TRACE(SRAM_SLICE, "{}: Spatial slicing is required, but padding after is too large", HLLOG_FUNC);
        return false;
    }

    if (m_params.padding[convIdx.paddingBeforeIndex] < 0 || m_params.padding[convIdx.paddingAfterIndex] < 0)
    {
        // TODO - SW-25558 - handle negative padding
        LOG_TRACE(SRAM_SLICE, "{}: Spatial slicing is required, but padding is negative", HLLOG_FUNC);
        return false;
    }
    return true;
}
