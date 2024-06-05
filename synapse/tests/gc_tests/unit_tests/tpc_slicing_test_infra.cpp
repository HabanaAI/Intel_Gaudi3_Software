#include "tpc_slicing_test_infra.h"
#include <string>

//
// DimSliceRange
//

DimSliceRange::Iterator& DimSliceRange::Iterator::operator++()
{
    advance();
    return *this;
}

DimSliceRange::Iterator DimSliceRange::Iterator::operator++(int)
{
    Iterator retVal {*this};
    advance();
    return retVal;
}

bool DimSliceRange::Iterator::operator==(const Iterator& other) const
{
    if (ended() && other.ended()) return true;
    return m_currDimSlice == other.m_currDimSlice;
}

DimSliceRange::Iterator DimSliceRange::Iterator::endIterator(unsigned int dimSize, unsigned int granularity)
{
    Iterator endIter(dimSize, granularity);
    endIter.m_currDimSlice.offset = dimSize;
    return endIter;
}

// Iterator advance algorithm:
// 1. Try to increase slice size
// 2. If slice size cannot be increased for the current offset, then
//       i. increase offset
//      ii. set smallest valid slice size for the new offset.
void DimSliceRange::Iterator::advance()
{
    if (ended()) return;  // Nothing to do

    if (advanceSliceSize()) return;

    advanceOffset();
    setInitialSliceSizeForOffset();
}

// Try to increase the slice size (possible if curOffset + curSliceSize < dimSize) and return true if successful.
bool DimSliceRange::Iterator::advanceSliceSize()
{
    unsigned curSliceEnd = m_currDimSlice.offset + m_currDimSlice.sliceSize;

    HB_ASSERT(m_dimSize >= curSliceEnd,
              "Trying to advance the slice size of an ended DimSliceRange iterator (dim size: {}, granularity: {}, "
              "current offset: {}, current slice size: {})",
              m_dimSize,
              m_granularity,
              m_currDimSlice.offset,
              m_currDimSlice.sliceSize);

    unsigned remainder = m_dimSize - curSliceEnd;  // Asserted above >= 0
    m_currDimSlice.sliceSize += std::min(remainder, m_granularity);
    return remainder > 0;
}

void DimSliceRange::Iterator::advanceOffset()
{
    m_currDimSlice.offset += m_granularity;
}

void DimSliceRange::Iterator::setInitialSliceSizeForOffset()
{
    int remainder = m_dimSize - m_currDimSlice.offset;
    // if remainder <= 0, the slice size would be 1. that's ok, since the iterator has ended (offset + size > dimSize).
    m_currDimSlice.sliceSize = clip(int(m_granularity), 1, remainder);
}

//
// TPCCustomIndexSpaceNode
//

NodePtr TPCCustomIndexSpaceNode::create(const TPCCustomIndexSpaceNode::Params& params,
                                        TensorPtr                              userInput,
                                        TensorPtr                              userOutput)
{
    HB_ASSERT(params.dims.size() <= 4, "Up to 4 dimensions are currently supported");

    // Currently creates a tensor with a single input and a single output.
    // Example for 4 dims:
    // Input dimensions are always {dim[0].size, dims[1].size, dims[2].size, dims[3].size}
    // Output may have the same dimension, or {dims[1].size, dim[0].size, dims[2].size, dims[3].size}
    // in case of transpose==true in the test params (transpose is relevant for the 2 inner dims).
    const auto         numDims = params.dims.size();
    std::vector<TSize> inputSizes(numDims);
    for (auto i = 0; i < numDims; i++)
    {
        inputSizes[i] = params.dims[i].size;
    }
    std::vector<TSize> outputSizes(inputSizes.begin(), inputSizes.end());
    if (params.transpose)
    {
        HB_ASSERT(numDims >= 2, "At least 2 dims are required when transpose is set");
        std::swap(outputSizes[0], outputSizes[1]);
    }

    TensorPtr input  = userInput ? userInput : TensorPtr(new Tensor(numDims, inputSizes.data(), syn_type_float));
    TensorPtr output = userOutput ? userOutput : TensorPtr(new Tensor(numDims, outputSizes.data(), syn_type_float));

    return NodePtr(new TPCCustomIndexSpaceNode({input}, {output}, params));
}

NodePtr TPCCustomIndexSpaceNode::createSliceableNode(TensorPtr userInput, TensorPtr userOutput)
{
    HB_ASSERT_PTR(userInput);
    HB_ASSERT_PTR(userOutput);
    HB_ASSERT(userInput->getDim() == userOutput->getDim(), "Input and output should have the same number of dims");
    HB_ASSERT(userInput->getAllSizesInElements() == userOutput->getAllSizesInElements(),
              "Input and output sizes should be identical");
    const auto&                     sizes = userInput->getAllSizesInElements();
    TPCCustomIndexSpaceNode::Params nodeParams {};
    for (auto i = 0; i < userInput->getDim(); i++)
    {
        nodeParams.dims.emplace_back(sizes[i], /*granularity*/ 1, /*overlap*/ 0);
    }
    nodeParams.transpose = false;

    NodePtr newNode = TPCCustomIndexSpaceNode::create(nodeParams, userInput, userOutput);

    // Required for Gaudi1 tests without big/small flags
    static_cast<TPCNode*>(newNode.get())->setAllowedForStitching(true);

    return newNode;
}

unsigned TPCCustomIndexSpaceNode::m_nodeIndex = 0;

TPCCustomIndexSpaceNode::TPCCustomIndexSpaceNode(const TensorVector& inputs,
                                                 const TensorVector& outputs,
                                                 const Params&       params)
: TPCNode(inputs, outputs, params.name.empty() ? "CustomIndexSpaceTPC" + std::to_string(m_nodeIndex) : params.name),
  m_params(params)
{
    initIndexSpace();
    setGUID(NOP_KERNEL_NAME);
    m_nodeIndex++;
}

void TPCCustomIndexSpaceNode::initIndexSpace()
{
    m_instanceWrapper.updateGlueCodeParamsAndTensorAccessPatternPointers(*this);

    initIndexSpaceGeometry();
    initInputAccessPattern();
    initOutputAccessPattern();

    m_instanceWrapper.setInstantiated(true);
}

void TPCCustomIndexSpaceNode::initIndexSpaceGeometry()
{
    auto& instance = m_instanceWrapper.getInstance();

    // Index space would be a grid where each element represent a dim[0].granularity x dims[1].granularity x ... x
    // dims[n].granularity chunk of the input, and the same chunk or a transpose of it in the output (depends on the
    // test params)
    instance.indexSpaceRank = 5;
    std::fill(instance.indexSpaceGeometry, instance.indexSpaceGeometry + ARRAY_SIZE(instance.indexSpaceGeometry), 1);
    for (auto i = 0; i < m_params.dims.size(); i++)
    {
        instance.indexSpaceGeometry[i] = div_round_up(m_params.dims[i].size, m_params.dims[i].granularity);
    }
}

void TPCCustomIndexSpaceNode::initInputAccessPattern()
{
    auto& tensorAP = m_instanceWrapper.getInstance().inputTensorAccessPattern[0];
    tensorAP.Value = 0;

    for (auto i = 0; i < m_params.dims.size(); i++)
    {
        tensorAP.mapping[i].indexSpaceDim = i;
        tensorAP.mapping[i].a             = m_params.dims[i].granularity;
        tensorAP.mapping[i].start_b       = m_params.dims[i].inputOffset;
        tensorAP.mapping[i].end_b =
            (int)m_params.dims[i].granularity - 1 + m_params.dims[i].inputOverlap + m_params.dims[i].inputOffset;
    }
}

void TPCCustomIndexSpaceNode::initOutputAccessPattern()
{
    auto& tensorAP = m_instanceWrapper.getInstance().outputTensorAccessPattern[0];
    tensorAP.Value = 0;

    for (auto i = 0; i < m_params.dims.size(); i++)
    {
        // transpose is relevant for the first 2 dims.
        unsigned idxSpaceDim                = (m_params.transpose && (i <= 1)) ? (1 - i) : i;
        tensorAP.mapping[i].indexSpaceDim   = idxSpaceDim;
        tensorAP.mapping[i].a               = m_params.dims[idxSpaceDim].granularity;
        tensorAP.mapping[i].end_b           = m_params.dims[idxSpaceDim].granularity - 1;
    }
}

//
// TPCCustomIndexSpaceMappingNode
//

NodePtr TPCCustomIndexSpaceMappingNode::create(const TPCCustomIndexSpaceMappingNode::Params& params)
{
    std::vector<TSize> tensorSizes(params.tensorRank, TENSOR_DIM_SIZE);
    TensorVector       inputs;
    for (auto i = 0; i < params.dimMappingForInputs.size(); i++)
    {
        inputs.push_back(std::make_shared<Tensor>(tensorSizes.size(), tensorSizes.data(), syn_type_float));
    }
    TensorVector outputs;
    for (auto i = 0; i < params.dimMappingForOutputs.size(); i++)
    {
        outputs.push_back(std::make_shared<Tensor>(tensorSizes.size(), tensorSizes.data(), syn_type_float));
    }
    return NodePtr(new TPCCustomIndexSpaceMappingNode(inputs, outputs, params));
}

unsigned TPCCustomIndexSpaceMappingNode::m_nodeIndex = 0;

TPCCustomIndexSpaceMappingNode::TPCCustomIndexSpaceMappingNode(const TensorVector& inputs,
                                                               const TensorVector& outputs,
                                                               const Params&       params)
: TPCNode(inputs, outputs, "TPCCustomIndexSpaceMappingNode_" + std::to_string(m_nodeIndex)), m_params(params)
{
    initIndexSpace();
    setGUID(NOP_KERNEL_NAME);
    m_nodeIndex++;
}

void TPCCustomIndexSpaceMappingNode::initTensorAccessPattern(tpc_lib_api::TensorAccessPattern&      tensorAccessPattern,
                                                             const TensorDimToIndexSpaceDimMapping& dimMapping)
{
    HB_ASSERT(dimMapping.empty() || (dimMapping.size() == m_params.tensorRank), "Invalid dim mapping params");

    if (dimMapping.empty())  // All dims are all required
    {
        tensorAccessPattern.allRequired = true;
    }
    else
    {
        tensorAccessPattern.allRequired = false;
        for (Dim tensorDim = 0; tensorDim < m_params.tensorRank; tensorDim++)
        {
            const auto& [indexSpaceDim, allRequiredDim] = dimMapping[tensorDim];
            HB_ASSERT(indexSpaceDim < m_params.nodeResolutionRank, "Invalid dim mapping params");
            tensorAccessPattern.mapping[tensorDim].indexSpaceDim = indexSpaceDim;
            if (allRequiredDim)
            {
                // a < 1
                tensorAccessPattern.mapping[tensorDim].a       = 0;
                tensorAccessPattern.mapping[tensorDim].start_b = 0;
                tensorAccessPattern.mapping[tensorDim].end_b   = 63;
            }
            else
            {
                // a >= 1
                tensorAccessPattern.mapping[tensorDim].a       = 64;
                tensorAccessPattern.mapping[tensorDim].start_b = 0;
                tensorAccessPattern.mapping[tensorDim].end_b   = 63;
            }
        }
    }
}

void TPCCustomIndexSpaceMappingNode::initIndexSpace()
{
    auto& instance = m_instanceWrapper.getInstance();

    m_instanceWrapper.updateGlueCodeParamsAndTensorAccessPatternPointers(*this);

    instance.indexSpaceRank = m_params.nodeResolutionRank;
    for (auto i = 0; i < m_params.nodeResolutionRank; i++)
    {
        instance.indexSpaceGeometry[i] = RESOLUTION_DIM_SIZE;
    }

    HB_ASSERT(getNumInputs() == m_params.dimMappingForInputs.size(), "Invalid dim mapping params for inputs");
    for (auto i = 0; i < getNumInputs(); i++)
    {
        initTensorAccessPattern(m_instanceWrapper.getInstance().inputTensorAccessPattern[i],
                                m_params.dimMappingForInputs[i]);
    }

    HB_ASSERT(getNumOutputs() == m_params.dimMappingForOutputs.size(), "Invalid dim mapping params for outputs");
    for (auto i = 0; i < getNumOutputs(); i++)
    {
        initTensorAccessPattern(m_instanceWrapper.getInstance().outputTensorAccessPattern[i],
                                m_params.dimMappingForOutputs[i]);
    }

    m_instanceWrapper.setInstantiated(true);
}