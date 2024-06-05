#include "transpose_node.h"

#include "access_pattern_generator.h"
#include "data_type_4bit.h"
#include "data_type_utils.h"
#include "defs.h"
#include "engine_selector.h"
#include "fcd_ops_utils.h"
#include "graph_traits.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "hal_reader/hal_reader.h"
#include "huge_tensor_transpose_slicer.h"
#include "layout.h"
#include "node_factory.h"
#include "sif/shape_inference_metadata.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"
#include "transpose_nodes_creator.h"
#include "transpose_permutation_handler_base.h"
#include "transpose_splitter.h"
#include "transpose_utils.h"
#include "transpose_via_mme.h"
#include "two_dim_transpose_reshaper.h"
#include "types.h"
#include "types_exception.h"
#include "utils.h"

#include <cstring>
#include <deque>
#include <memory>
#include <sstream>

using namespace gc;

/**
 * Extract each dimension index by its dim position
 *                                    C,W,H,B
 * Ex. for 4 dimensions with sizes - [2,1,3,4], index [1,0,2,1]
 * means, the offset from the base ptr is
 *  C     W         H                 B
 *  1 + 0 * (2) + 2 * (2 * 1) + 1 * (2 * 1 * 3) = 11
 */
template<typename A>
static void extractRealIndex(const uint64_t* strides,
                             unsigned int    index,
                             A&              realIndex)
{
    for (int dimIndex = realIndex.size() - 1; dimIndex >=0; --dimIndex)
    {
        realIndex[dimIndex] = index / strides[dimIndex];
        index = index % strides[dimIndex];
    }
}

template<typename A>
static void reorganizeIndex(const TransposePermutationArray& permutation,
                            const A& realIndex,
                            A& outputIndex)
{
    for (unsigned int i = 0; i < realIndex.size(); ++i)
    {
        outputIndex[i] = realIndex[permutation[i]];
    }
}

template<unsigned int N>
struct NotBelow1
{
    static const int value = N == 0 ? 1 : N;
};

// specialize for possible dimensions
template<size_t ElementSize, unsigned int Dim = HABANA_DIM_MAX>
static void runTransposeOnCpu(size_t                           inDim,
                              const void*                      inData,
                              const uint64_t*                  inStrides,
                              uint64_t                         numOfElements,
                              const TransposePermutationArray& permutation,
                              const uint64_t*                  outStrides,
                              void*                            outData)
{
    // Recursive call to find the matching dimension at runtime, without going below 1. assert protects infinite loop.
    HB_ASSERT(Dim != 1, "Can't transpose a dim 1 tensor");
    if (Dim != inDim)
    {
        runTransposeOnCpu<ElementSize, NotBelow1<Dim - 1>::value>(inDim,
                                                                  inData,
                                                                  inStrides,
                                                                  numOfElements,
                                                                  permutation,
                                                                  outStrides,
                                                                  outData);
        return;
    }

#pragma omp parallel for
    for (size_t index = 0; index < numOfElements; ++index)
    {
        std::array<unsigned int, Dim> realIndex;
        extractRealIndex(inStrides, index * ElementSize, realIndex);

        std::array<unsigned int, Dim> outputIndex;
        reorganizeIndex(permutation, realIndex, outputIndex);

        size_t poutIndex = 0;
        for (unsigned int i = 0; i < outputIndex.size(); ++i)
        {
            poutIndex += (outStrides[i] / ElementSize) * outputIndex[i];
        }

        std::memcpy(static_cast<char*>(outData) + ElementSize * poutIndex,
                    static_cast<const char*>(inData) + ElementSize * index,
                    ElementSize);
    }
}

// a wrapper function that doesn't deal with u/int4_t datatypes
static void transposeOnCpuImpl(const TensorPtr& in, const TensorPtr& out, const TransposePermutationArray& permutation)
{
    synDataType dataType = in->getElementType();

    const size_t elementSize = in->getElementSizeInBytes();
    const size_t dim         = in->getDim();
    HB_ASSERT(dim != 1, "Can't transpose a dim 1 tensor");

    const void* pIn  = in->map();
    void*       pOut = out->getAddress();
    if (pOut == nullptr)
    {
        LOG_TRACE(GC, "Allocating new buffer to transpose output {} according to input buffer type", out->getName());
        pOut = allocateBufferForSynType(in->getBufferDataType(), in->getTotalElements());
        out->setTensorBuffer(pOut, in->getTotalElements() * elementSize, in->getBufferDataType(), false);
        out->setShouldFreeBuffer(true);
    }

    uint64_t IfmStrides[Tensor::c_numOfNStrides];
    // cache hot constants
    in->getNStridesInBytes(IfmStrides);
    HB_ASSERT(IfmStrides[0] == elementSize, "detected strided fcd");

    uint64_t OfmStrides[Tensor::c_numOfNStrides];
    out->getNStridesInBytes(OfmStrides);
    if (OfmStrides[0] == 0)
    {
        LOG_DEBUG(GC, "Transpose output strides not set, calculating them using input element type");
        // Temporarily set output type as the input type, this will calculate the strides.
        // Then restore element type as before.
        synDataType origOutputType = out->getElementType();
        out->changeDefaultElementType(in->getElementType());
        out->getNStridesInBytes(OfmStrides);
        out->changeDefaultElementType(origOutputType);
    }
    // Allow invalid strides in case of invalid element type for output tensors
    HB_ASSERT(OfmStrides[0] == elementSize || out->getElementType() == syn_type_na,
              "detected strided fcd on output tensor with valid data type");

    if (GCFG_ENABLE_RUN_ON_CPU_DUMMY_MODE.value()) return;
    auto         OfmSizes      = out->getNSizesInElements();
    unsigned int numOfElements = 1;
    for (unsigned int index = 0; index < dim; ++index)
    {
        numOfElements *= OfmSizes[index];
    }

    HB_ASSERT(out->getBufferSizeInBytes() >= numOfElements * elementSize,
              "node {}: trying to calculate cpu transpose without enough space in output buffer!",
              out->getName());

    switch (dataType)
    {
        case syn_type_fixed:
        case syn_type_uint8:
        case syn_type_fp8_143:
        case syn_type_fp8_152:
            return runTransposeOnCpu<sizeof(int8_t)>(dim,
                                                     pIn,
                                                     IfmStrides,
                                                     numOfElements,
                                                     permutation,
                                                     OfmStrides,
                                                     pOut);

        case syn_type_int16:
        case syn_type_uint16:
        case syn_type_bf16:
        case syn_type_fp16:
            return runTransposeOnCpu<sizeof(int16_t)>(dim,
                                                      pIn,
                                                      IfmStrides,
                                                      numOfElements,
                                                      permutation,
                                                      OfmStrides,
                                                      pOut);
        case syn_type_single:
        case syn_type_int32:
        case syn_type_uint32:
            return runTransposeOnCpu<sizeof(int32_t)>(dim,
                                                      pIn,
                                                      IfmStrides,
                                                      numOfElements,
                                                      permutation,
                                                      OfmStrides,
                                                      pOut);
        case syn_type_int64:
        case syn_type_uint64:
            return runTransposeOnCpu<sizeof(int64_t)>(dim,
                                                      pIn,
                                                      IfmStrides,
                                                      numOfElements,
                                                      permutation,
                                                      OfmStrides,
                                                      pOut);
        case syn_type_int4:
        case syn_type_uint4:
            // fallthrough: The caller (TransposeNode::transposeOnCpu) should've already transformed them to temp
            // u/int8_t
        default:
            HB_ASSERT(false, "Unsupported tensor type");
    }
}

template <typename T>
static void cast4BitTo8Bit(const TensorPtr& in, const TensorPtr& out)
{
    HB_ASSERT(in->isType4Bit(), "Tensor {} must be of type u/int4", in->getName());
    auto pIn  = static_cast<T*>(in->map());
    auto pOut = static_cast<T*>(out->map());

    if (in->isCondensed4Bit())
    {
        HB_ASSERT(in->getTensorAnnotation().info4Bit.condensedDim != MME_STATIC_WEIGHTS_CONDENSE_DIM,
                  "TransposeOnCpu of MME static weights still not supported");
        expand4BitBufferTo8BitBuffer(pIn, in->getDenseSizeInElements(), pOut);
    }
    else
    {
        // 4bit tensors occupy same memory as 8bit tensors if not condensed yet.
        memcpy(pOut, pIn, in->getTotalSizeInBytes());
    }
}

template <typename T>
static void cast8BitTo4Bit(const TensorPtr& in, const TensorPtr& out)
{
    HB_ASSERT(out->isType4Bit(), "Tensor {} must be of type u/int4", out->getName());

    auto pIn  = static_cast<T*>(in->map());
    auto pOut = static_cast<T*>(out->map());

    if (out->isCondensed4Bit())
    {
        HB_ASSERT(out->getTensorAnnotation().info4Bit.condensedDim != MME_STATIC_WEIGHTS_CONDENSE_DIM,
                  "TransposeOnCpu of MME static weights still not supported");
        condense8BitBufferTo4BitBuffer(pIn, in->getDenseSizeInElements(), pOut);
    }
    else
    {
        // 4bit tensors occupy same memory as 8bit tensors if not condensed yet.
        memcpy(pOut, pIn, in->getTotalSizeInBytes());
    }
}

void TransposeNode::transposeOnCpu(const TensorPtr&                 in,
                                   const TensorPtr&                 out,
                                   const TransposePermutationArray& permutation)
{
    HB_ASSERT(in->getDim() == out->getDim(), "Expected same dimension for input and output");

    // when the input tensor is a static param and unquantized, the data is in float32,
    // even though the tensor's type may not be float32.
    // because of that we scale the data size to match float32 size, and rebind the output tensor to match float32 type.
    synDataType dataType = in->getElementType();
    if (in->isStaticParam() && (!in->isDataTypeMatchData()))
    {
        dataType = synDataType::syn_type_single;
        if (out->isBound())
        {
            out->rebind(in->getTotalElements() * sizeof(float));
        }
    }

    switch (dataType)
    {
    case syn_type_int4:
    {
        auto castedIn  = std::make_shared<Tensor>(in->getDim(), in->getAllSizesInElements().data(), syn_type_int8);
        auto castedOut = std::make_shared<Tensor>(out->getDim(), out->getAllSizesInElements().data(), syn_type_int8);
        cast4BitTo8Bit<int8_t>(in, castedIn);
        transposeOnCpuImpl(castedIn, castedOut, permutation);
        cast8BitTo4Bit<int8_t>(castedOut, out);
        break;
    }
    case syn_type_uint4:
    {
        auto castedIn  = std::make_shared<Tensor>(in->getDim(), in->getAllSizesInElements().data(), syn_type_uint8);
        auto castedOut = std::make_shared<Tensor>(out->getDim(), out->getAllSizesInElements().data(), syn_type_uint8);
        cast4BitTo8Bit<uint8_t>(in, castedIn);
        transposeOnCpuImpl(castedIn, castedOut, permutation);
        cast8BitTo4Bit<uint8_t>(castedOut, out);
        break;
    }
    default:
        transposeOnCpuImpl(in, out, permutation);
    }

    if (in->isDataTypeMatchData())
    {
        LOG_DEBUG(GC,
                  "Transpose input tensor had its buffer data type same as element data type,"
                  "setting for transpose output tensor");
        out->setAsDataTypeMatchData();
    }
}

template<class T>
T* getUnitMatrix(unsigned matriWidthSize)
{
    T* matrix = new T[matriWidthSize * matriWidthSize];

    memset(matrix, 0, matriWidthSize * matriWidthSize * sizeof(T));

    for (unsigned i = 0; i < matriWidthSize; ++i)
    {
        matrix[i*matriWidthSize + i] = (T)1;
    }

    return matrix;
}

template<>
int32_t* getUnitMatrix<int32_t>(unsigned matriWidthSize)
{
    HB_ASSERT(matriWidthSize == 64, "Expected 64x64 for 4-byte data type");
    int16_t* matrix = new int16_t[(matriWidthSize * 2) * (matriWidthSize * 2)];

    memset(matrix, 0, (matriWidthSize * 2) * (matriWidthSize * 2) * sizeof(int16_t));

    for (unsigned i = 0; i < matriWidthSize; ++i)
    {
        matrix[(i * matriWidthSize * 2) + (i * 2)] = (int16_t)1;
    }
    for (unsigned i = 0; i < matriWidthSize; ++i)
    {
        matrix[(matriWidthSize * 2 * matriWidthSize) + (i * matriWidthSize * 2) + (i * 2) + 1] = (int16_t)1;
    }

    return reinterpret_cast<int32_t*>(matrix);
}

template<>
float_t* getUnitMatrix<float_t>(unsigned matrixWidthSize)
{

    if (matrixWidthSize > SHRT_MAX)
    {
        LOG_ERR(HABANA_NODE, "Physical Transpose of fp32 tensor is limited to SHRT_MAX");
        HB_ASSERT(false, "Physical Transpose of fp32 tensor is limited to SHRT_MAX");
        return nullptr;
    }

    float_t* matrix = new float_t[matrixWidthSize * matrixWidthSize];

    memset(matrix, 0, matrixWidthSize * matrixWidthSize * sizeof(float));

    for (unsigned i = 0; i < matrixWidthSize; ++i)
    {
        matrix[i*matrixWidthSize + i] = 1.0;
    }

    return matrix;
}


template<>
bfloat16* getUnitMatrix<bfloat16>(unsigned matrixWidthSize)
{
    int16_t* matrix = new int16_t[matrixWidthSize * matrixWidthSize];

    memset(matrix, 0, matrixWidthSize * matrixWidthSize * sizeof(int16_t));

    for (unsigned i = 0; i < matrixWidthSize; ++i)
    {
        matrix[i * matrixWidthSize + i] = bfloat16(1.0f);
    }

    return reinterpret_cast<bfloat16*>(matrix);
}

void* generateUnitMatrixFromTensorType(synDataType tensorType,
                                       unsigned matrixWidthSize)
{
    switch (tensorType)
    {
    case syn_type_fixed:
        return (void*)getUnitMatrix<int8_t>(matrixWidthSize);
    case syn_type_uint8:
        return (void*)getUnitMatrix<uint8_t>(matrixWidthSize);
    case syn_type_int16:
        return (void*)getUnitMatrix<int16_t>(matrixWidthSize);
    case syn_type_uint16:
        return (void*)getUnitMatrix<uint16_t>(matrixWidthSize);
    case syn_type_single:
        return (void*)getUnitMatrix<int32_t>(matrixWidthSize);
    case syn_type_int32:
        return (void*)getUnitMatrix<int32_t>(matrixWidthSize);
    case syn_type_bf16:
    case syn_type_int4:
    case syn_type_uint4:
    case syn_type_fp16:
    case syn_type_uint32:
    case syn_type_tf32:
    case syn_type_hb_float:
    case syn_type_fp8_143:
    case syn_type_fp8_152:
    case syn_type_int64:
    case syn_type_uint64:
    case syn_type_ufp16:
        HB_ASSERT(
            false,
            "Can't get unit matrix for int4/uint4/fp16/uint32/bf16/fp8_143/fp8_152/tf32/hb_float/int64/uint64/ufp16");
        break;
    case syn_type_na:
    case syn_type_max:
        HB_ASSERT(false, "Can't get unit matrix with no type");
        break;
    }

    return nullptr;
}

TensorPtr createUnitTensor(unsigned int dim, synDataType type, unsigned vectorSizeInElements)
{
    TSize sizes[Tensor::c_tensorMaxDim];
    memset (sizes, 0, sizeof(sizes));

    sizes[0] = sizes[1] = vectorSizeInElements;
    for (unsigned i = 2; i < dim; ++i)
    {
        sizes[i] = 1;
    }

    void* unitMatrix = generateUnitMatrixFromTensorType(type, vectorSizeInElements);

    if (dataTypeToSizeInBits(type) == BITS_PER_BYTE * sizeof(float))
    {
        sizes[0] = 128;
        sizes[1] = 64;
        sizes[2] = 2;
        type     = syn_type_int16;
    }

    // We should multiply the transposed tensor by the unit matrix to keep it the same
    TensorPtr unitTensor = std::make_shared<Tensor>(dim, sizes, type);
    unitTensor->setName("UnitMatrix");

    unitTensor->bind(unitMatrix, true);

    unitTensor->setUnitTensor();

    return unitTensor;
}

void layoutPermutation(const TransposePermutationArray& permutation, char* dimStr, std::ostringstream& output)
{
    for (unsigned int i = 0; i < permutation.size(); ++i)
    {
        output << dimStr[permutation[i]] << " ";
    }
}

std::string TransposeNode::getPermutationString(const TransposePermutationArray& permutation)
{
    return Permutation(permutation).toString();
}

TransposeNode::TransposeNode(const TensorVector& inputs,
                             const TensorVector& outputs,
                             UserParams          params,
                             unsigned            paramsSize,
                             std::string_view    name)
: MultiNode(inputs, outputs, name, Node::TYPE_INTERNAL_TRANSPOSE, SIF_TRANSPOSE)
{
    setParams(params, paramsSize);
}

TransposeNode::TransposeNode(const TensorPtr& input,
                             const TensorPtr& output,
                             UserParams       params,
                             unsigned         paramsSize,
                             std::string_view name)
: TransposeNode(TensorVector {std::move(input)}, TensorVector {std::move(output)}, params, paramsSize, name)
{
}

void TransposeNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParams == nullptr)
    {
        LOG_ERR(HABANA_NODE, "TransposeNode userParams is null");
        throw InvalidNodeParamsException(m_name, "userParams");
    }
    TransposePermutationArray permutation;
    unsigned                  tensorDim = 0;
    if (userParamsSize != sizeof(synTransposeParamsNDims))
    {
        if (userParamsSize != sizeof(synTransposeParams))
        {
            LOG_ERR(HABANA_NODE, "TransposeNode userParams size is incorrect");
            throw InvalidNodeParamsSizeException(m_name);
        }
        synTransposeParams transposeParams = *(synTransposeParams*)userParams;
        tensorDim                          = transposeParams.tensorDim;

        permutation = {transposeParams.permutation, transposeParams.permutation + transposeParams.tensorDim};
    }
    else
    {
        synTransposeParamsNDims transposeParams = *(synTransposeParamsNDims*)userParams;
        tensorDim                               = transposeParams.tensorDim;

        for (int i = 0; i < transposeParams.tensorDim; i++)
        {
            permutation.push_back((TransposePermutationDim)transposeParams.permutation[i]);
        }
    }

    LOG_TRACE(HABANA_NODE,
              "TransposeNode name - {}, params - tensorDim={}, permutation={}, in sizes={}",
              m_name,
              tensorDim,
              toString(permutation, ','),
              toString(m_inputs[0]->getNSizesInElements(), ','));
    m_permutation = permutation;
    memcpy(m_sifMetadata.permutation, permutation.data(), permutation.size() * sizeof(permutation[0]));
}

NodePtr TransposeNode::createNode(const TensorVector& inputs,
                                  const TensorVector& outputs,
                                  UserParams          userParams,
                                  unsigned            userParamsSize,
                                  std::string_view    guid,
                                  std::string_view    name)
{
    return NodePtr(new TransposeNode(inputs, outputs, userParams, userParamsSize, name));
}

bool TransposeNode::validateNode() const
{
    if (m_inputs.size() != 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting 1 input and 1 output)");
        return false;
    }

    if (getInput(TENSOR_IFM)->getDim() != getOutput(TENSOR_OFM)->getDim())
    {
        LOG_ERR(HABANA_NODE, "Invalid operand shapes (expecting to have the same dimension");
        return false;
    }

    if (!getInput(TENSOR_IFM)->compareGeometryWithTranspose(*getOutput(TENSOR_OFM), m_permutation))
    {
        LOG_ERR(HABANA_NODE, "Invalid operand shapes (expecting reflect the transpose permutation");
        return false;
    }

    return MultiNode::validateNode();
}

NodePtr TransposeNode::clone() const
{
    return std::make_shared<TransposeNode>(*this);
}

void TransposeNode::setPermutation(TransposePermutationArray array)
{
    m_permutation = std::move(array);
    memcpy(m_sifMetadata.permutation, m_permutation.data(), m_permutation.size() * sizeof(m_permutation[0]));
}

bool TransposeNode::RunOnCpu()
{
    TensorPtr in  = getInput(TENSOR_IFM);
    TensorPtr out = getOutput(TENSOR_OFM);
    if (in->getElementType() != out->getElementType() && out->getElementType() != syn_type_na)
    {
        LOG_WARN(GC, "Cannot perform transpose on cpu for input with type {} and output with type {}",
                 in->getElementType(), out->getElementType());
        return false;
    }
    transposeOnCpu(in, out, m_permutation);
    return true;
}

std::optional<NodeVector> TransposeNode::tryToReshapeTwoDimTranspose(const uint64_t twoDimTransposeCost) const
{
    if (twoDimTransposeCost <= FcdOpsUtils::getOptimalCost(*m_inputs[0])) return std::nullopt;
    LOG_TRACE(GC, "try to reshape {} to 3 dim transpose", getNodeName());

    TwoDimTransposeReshaper reshaper(*this);
    if (!reshaper.isValid()) return std::nullopt;

    auto [newNodes, threeDimTransposeCost] =
        TransposeSplitter(reshaper.get3DimTranspose(), true /* skip perf report */).splitTransposeViaCostModel();
    if (threeDimTransposeCost >= twoDimTransposeCost) return std::nullopt;

    LOG_DEBUG(GC,
              "reshape {} to 3 dim transpose, old sizes: [{}], new sizes: [{}]",
              getNodeName(),
              m_inputs[0]->getDimSizesStr(),
              reshaper.get3DimTranspose().getInput(0)->getDimSizesStr());
    const auto& wrappingNodes = reshaper.getWrappingNodes();
    newNodes.insert(newNodes.end(), wrappingNodes.begin(), wrappingNodes.end());
    return newNodes;
}

NodeList TransposeNode::handle64BitTranspose() const
{
    NodeList ret;
    // arbitrary choice of 32u since mem move ops are agnostic to signedness
    constexpr synDataType dtype = syn_type_uint32;
    // cast to 32b + reshape data tensors
    auto [newInput, reinterpretIn, reshapeIn]    = reinterpret64BitTensor(getInput(0), true, dtype);
    auto [newOutput, reinterpretOut, reshapeOut] = reinterpret64BitTensor(getOutput(0), false, dtype);
    ret.emplace_back(std::move(reinterpretIn));
    ret.emplace_back(std::move(reinterpretOut));
    ret.emplace_back(std::move(reshapeIn));
    ret.emplace_back(std::move(reshapeOut));

    // catch corner case with informative error message.
    constexpr unsigned maxStaticReshapeDim = ARRAY_SIZE(synStaticReshapeSifParams::outputMaxSizes);
    HB_ASSERT(getOutput(0)->getDim() != maxStaticReshapeDim || !getOutput(0)->isDynamicDim(maxStaticReshapeDim - 1),
              "Oops, missing support for {}D dynamic transpose with 64bit operands. node name: {}",
              maxStaticReshapeDim,
              m_name);

    // create new expanded params
    synTransposeParamsNDims newParams = {0};
    newParams.tensorDim               = m_permutation.size() + 1;
    newParams.permutation[0]          = 0;
    for (unsigned dim = 0; dim < m_permutation.size(); dim++)
    {
        newParams.permutation[dim + 1] = m_permutation[dim] + 1;
    }
    auto  n = NodeFactory::createNode({newInput}, {newOutput}, &newParams, NodeFactory::transposeNodeTypeName, m_name);
    auto* transposeNode = dynamic_cast<TransposeNode*>(n.get());
    HB_ASSERT_PTR(transposeNode);
    transposeNode->setPreferTransposeOnlyOnce(getPreferTransposeOnlyOnce());
    transposeNode->setPreferLogicalBeforePhysical(getPreferLogicalBeforePhysical());
    ret.emplace_back(std::move(n));
    return ret;
}

NodeList TransposeNode::extract(const HabanaGraph& g)
{
    if (GCFG_ENABLE_HUGE_TENSOR_SLICING.value() && HugeTensorTransposeSlicer::doesRequireSlicing(this))
    {
        // [CID: 86457] False positive - Uninitialized scalar variable defects caused by usage of std::optional,
        // https://community.synopsys.com/s/article/FP-Uninitialized-scalar-variable-defects-caused-by-usage-of-std-optional
        auto ret = HugeTensorTransposeSlicer(this, std::nullopt).slice();
        return {ret.begin(), ret.end()};
    }

    if (is64BitOperands())
    {
        return handle64BitTranspose();
    }
    // TODO: SW-111336: Optimize transpose multi node extraction with cost mode

    // legacy mode
    if (likely(g.getCompilationMode() == CompilationMode::Eager || GCFG_TRANSPOSE_SPLITTING_THRESHOLD.value() == 1.0))
    {
        const auto& nodeVec = TransposeNodesCreator().getTransposeNodes(*this);
        return NodeList(nodeVec.begin(), nodeVec.end());
    }

    auto [nodeVec, cost] = TransposeSplitter(*this).splitTransposeViaCostModel();
    // In case that transpose is two dim transpose there is no possible split.
    // However, we can try to separates the larger dim, so it increase the transpose dim to 3, which may
    // open possible splits that improves utilization.
    if (m_permutation.size() == 2 && GCFG_ENABLE_TWO_DIM_TRANSPOSE_RESHAPER.value())
    {
        auto res = tryToReshapeTwoDimTranspose(cost);
        if (res.has_value())
        {
            nodeVec = std::move(res.value());
        }
    }

    return NodeList(nodeVec.begin(), nodeVec.end());
}

bool TransposeNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return true;
}

void TransposeNode::printParamsRawData() const
{
    if (m_permutation.size() > 5)
    {
        synTransposeParamsNDims transposeParams;
        memset(&transposeParams, 0, sizeof(transposeParams));
        transposeParams.tensorDim = m_permutation.size();
        std::copy(m_permutation.begin(), m_permutation.end(), transposeParams.permutation);
        BaseClass::printParamsRawData((void*)&transposeParams, sizeof(transposeParams));
    }
    else
    {
        synTransposeParams transposeParams;
        memset(&transposeParams, 0, sizeof(transposeParams));
        transposeParams.tensorDim = m_permutation.size();
        std::copy(m_permutation.begin(), m_permutation.end(), transposeParams.permutation);
        BaseClass::printParamsRawData((void*)&transposeParams, sizeof(transposeParams));
    }
}

SifNodeParams TransposeNode::getShapeInferenceFunctionUserParams()
{
    return reinterpret_cast<SifNodeParams>(&m_sifMetadata);
}
size_t TransposeNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(SifTransposeMetadata);
}

void TransposeNode::permuteParams(const PermutationVector& inputPermutations)
{
    if (inputPermutations.size() > 1)
    {
        LOG_WARN(GC, "'transpose' has more than one input permutation, taking the first one to convert params");
    }
    auto& inputPerm = inputPermutations[0];
    auto  inverseIn = inputPerm.getInversePermutation();
    inverseIn.permute(gc::Permutation(m_permutation));
    inverseIn.permute(inputPerm);
    inverseIn.getValues(m_permutation.data(), inverseIn.size());

    memcpy(m_sifMetadata.permutation, m_permutation.data(), m_permutation.size() * sizeof(m_permutation[0]));
}

gc::access_pattern::NodeAccessPatternPtr TransposeNode::generateNodeAccessPattern() const
{
    return gc::access_pattern::AccessPatternTransposeGenerator::generate(this, m_permutation);
}

// LogicalTransposeNode

LogicalTransposeNode::LogicalTransposeNode(const TensorPtr& IFM,
                                           const TensorPtr& OFM,
                                           std::string_view name,
                                           Node::eNodeType  type)
: LogicalOpNode(TensorVector({IFM}), TensorVector({OFM}), name, OUTPUT_TO_INPUT, type, SIF_TRANSPOSE),
  m_isUserPermutationTranspose(false)
{
}

NodePtr LogicalTransposeNode::createNode(const TensorVector& inputs,
                                         const TensorVector& outputs,
                                         UserParams          userParams,
                                         std::string_view    guid,
                                         std::string_view    name)
{
    LogicalTransposeNode* logicalTransposeNode =
        new LogicalTransposeNode(inputs[TENSOR_IFM], outputs[TENSOR_OFM], name);
    logicalTransposeNode->setParams(userParams, sizeof(TransposePermutationArray));
    return (NodePtr(logicalTransposeNode));
}

void LogicalTransposeNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParamsSize != sizeof(TransposePermutationArray))
    {
        LOG_ERR(HABANA_NODE, "LogicalTransposeNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name);
    }
    m_permutation = *(TransposePermutationArray*)userParams;
    memcpy(m_sifMetadata.permutation, m_permutation.data(), m_permutation.size() * sizeof(m_permutation[0]));
    LOG_TRACE(HABANA_NODE,
              "LogicalTransposeNode name - {}, params - permutation={}, in sizes={}",
              getNodeName(),
              toString(m_permutation, ','),
              toString(m_inputs[0]->getNSizesInElements(), ','));
}

bool LogicalTransposeNode::isSupportedPermutation(const Tensor&                    in,
                                                  const Tensor&                    out,
                                                  const TransposePermutationArray& permutation)
{
    if (out.isZeroSizedDataTensor()) return true;  // in case of zero sized tensor, we wish to use logical transpose.
    return permutation[0] == 0;                    //  check that we will not get FCD stride
}

bool LogicalTransposeNode::validateNode() const
{
    if (m_inputs.size() != 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting 1 input and 1 output)");
        return false;
    }
    return LogicalOpNode::validateNode();
}

bool LogicalTransposeNode::RunOnCpu()
{
    TensorPtr in  = getInput(TENSOR_IFM);
    TensorPtr out = getOutput(TENSOR_OFM);
    TransposeNode::transposeOnCpu(in, out, m_permutation);
    return true;
}

NStrideArray LogicalTransposeNode::calculateAliasStrides(unsigned idx) const
{
    bool                      swapPermutation = getAliasDirection() == INPUT_TO_OUTPUT;
    TensorPtr                 real            = !swapPermutation ? getInput(TENSOR_IFM) : getOutput(TENSOR_OFM);
    TensorPtr                 alias           = !swapPermutation ? getOutput(TENSOR_IFM) : getInput(TENSOR_OFM);
    gc::Permutation           permutation(m_permutation);

    if (swapPermutation)
    {
        /*
         * We need to change to backward permutation.
         * For example: original permutation from input to output CWHB->CBWH
         * should be changed to output to input permutation CWHB->CHBW
         */
        permutation = permutation.getInversePermutation();
    }

    const TStride* oldStrides = real->getNStridesInBytes();
    NStrideArray   newStrides = {1};
    real->getNStridesInBytes(newStrides.data());
    permutation.permute(oldStrides, newStrides.data(), real->getDim());
    return newStrides;
}

void LogicalTransposeNode::runLogicalOperation() const
{
    LOG_DEBUG(OPT_LOGICAL_OPS,
              "Run transpose as a logical operation. permutation: {}",
              TransposeNode::getPermutationString(m_permutation));

    bool         swapPermutation = getAliasDirection() == INPUT_TO_OUTPUT;
    TensorPtr    realTensor      = !swapPermutation ? getInput(TENSOR_IFM) : getOutput(TENSOR_OFM);
    TensorPtr    aliasTensor     = !swapPermutation ? getOutput(TENSOR_IFM) : getInput(TENSOR_OFM);
    NStrideArray newStrides      = calculateAliasStrides(0);

    // Set alias tensor
    aliasTensor->setAsAliasSubTensor(realTensor);

    auto sizes = aliasTensor->getNSizesInElements();
    aliasTensor->reshape(aliasTensor->getDim(), sizes.data(), newStrides.data());

    HB_ASSERT(!aliasTensor->isStridedOnFCD(), "logical transpose {} created fcd strides", m_name);
}

NodePtr LogicalTransposeNode::clone() const
{
    return NodePtr(new LogicalTransposeNode(*this));
}

bool LogicalTransposeNode::isRedundantNode() const
{
    for (int i = 0; i < this->m_permutation.size(); i++)
    {
        /* if the transpose node does change the input tensor permutation */
        if (i != this->m_permutation[i])
        {
            return false;
        }
    }

    return true;
}

bool LogicalTransposeNode::isAliasStrided() const
{
    bool         swapPermutation = getAliasDirection() == INPUT_TO_OUTPUT;
    TensorPtr    aliasTensor     = !swapPermutation ? getOutput(TENSOR_IFM) : getInput(TENSOR_OFM);
    NStrideArray newStrides      = calculateAliasStrides(0);

    uint64_t expectedStride = aliasTensor->getElementSizeInBytes();
    for (unsigned dim = 0; dim < aliasTensor->getDim(); dim++)
    {
        if (expectedStride != newStrides[dim]) return true;
        expectedStride *= aliasTensor->getSizeInElements(dim);
    }
    return false;
}

SifNodeParams LogicalTransposeNode::getShapeInferenceFunctionUserParams()
{
    return reinterpret_cast<SifNodeParams>(&m_sifMetadata);
}

size_t LogicalTransposeNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(SifTransposeMetadata);
}

std::string TransposeNode::getNodeParametersStr() const
{
    return fmt::format("permutation: {}", getPermutationString(m_permutation));
}

std::string LogicalTransposeNode::getNodeParametersStr() const
{
    return fmt::format("permutation: {}", TransposeNode::getPermutationString(m_permutation));
}

gc::access_pattern::NodeAccessPatternPtr LogicalTransposeNode::generateNodeAccessPattern() const
{
    return gc::access_pattern::AccessPatternTransposeGenerator::generate(this, m_permutation);
}

MmeTransposeNode::MmeTransposeNode(const TensorPtr&                 IFM,
                                   const TensorPtr&                 OFM,
                                   const TransposePermutationArray& permutation,
                                   std::string_view                 name)
: MmeNode(TensorVector({IFM}), TensorVector({OFM}), name, Node::TYPE_INTERNAL_TRANSPOSE, SIF_TRANSPOSE),
  m_permutation(permutation)
{
    memcpy(m_sifMetadata.permutation, permutation.data(), permutation.size() * sizeof(permutation[0]));
}

bool MmeTransposeNode::validateNode() const
{
    if (m_inputs.size() != 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "MmeTransposeNode Invalid number of operands (expecting 1 input and 1 output)");
        return false;
    }
    return getInput(TENSOR_IFM)->compareGeometryWithTranspose(*getOutput(TENSOR_OFM), m_permutation) && Node::validateNode();
}

NodePtr MmeTransposeNode::clone() const
{
    return NodePtr(new MmeTransposeNode(*this));
}

const TransposePermutationArray& MmeTransposeNode::permutation() const
{
    return m_permutation;
}

bool MmeTransposeNode::RunOnCpu()
{
    TensorPtr in  = getInput(TENSOR_IFM);
    TensorPtr out = getOutput(TENSOR_OFM);
    TransposeNode::transposeOnCpu(in, out, m_permutation);
    return true;
}

synDataType MmeTransposeNode::getRequiredInputType(uint32_t tensorIdx) const
{
    return Node::getRequiredInputType(tensorIdx);
}

synDataType MmeTransposeNode::getRequiredOutputType(uint32_t tensorIdx) const
{
    return Node::getRequiredOutputType(tensorIdx);
}

NodeROI MmeTransposeNode::generateRoi() const
{
    TransposePermutationHandlerBase handler(m_permutation);
    NodeROI                         fullRoi;
    fullRoi.size[0] = handler.OfmFcdSize(*getOutput(TENSOR_OFM));
    fullRoi.size[1] = handler.OfmSpatialSize(*getOutput(TENSOR_OFM));
    for (unsigned int idx = 2; idx < ARRAY_SIZE(fullRoi.size); ++idx)
    {
        fullRoi.size[idx] = 1;
    }
    fullRoi.vectorSize =
        m_graphTraits->getHalReader()->getMmeVectorSize() / getInput(TENSOR_IFM)->getElementSizeInBytes();
    fullRoi.numIterations     = div_round_up(fullRoi.size[1], fullRoi.vectorSize);
    fullRoi.spatialSizeMinus1 = (fullRoi.size[1] - 1) % fullRoi.vectorSize;

    return fullRoi;
}

bool MmeTransposeNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return TransposeViaNativeMME::supported(permutation());
}

bool MmeTransposeNode::isOperandTransposed(const TensorPtr& t) const
{
    // Despite the name, the operand is read straight-up and written transposed
    return false;
}

std::string MmeTransposeNode::getNodeParametersStr() const
{
    return fmt::format("permutation: {}", TransposeNode::getPermutationString(m_permutation));
}

gc::access_pattern::NodeAccessPatternPtr MmeTransposeNode::generateNodeAccessPattern() const
{
    return gc::access_pattern::AccessPatternTransposeGenerator::generate(this, m_permutation);
}

bool TransposeNode::isPhysicalTranspose(const TensorPtr& input, const TransposePermutationArray& permutation)
{
    HB_ASSERT_PTR(input);
    HB_ASSERT(!permutation.empty(), "permutation rank is 0, which is undefined");
    return permutation[0] != 0 && !isSameDataMemOrder(*input, permutation);
}

bool TransposeNode::isPhysicalTranspose(const TransposeNodeParams& transpose)
{
    return isPhysicalTranspose(transpose.input, transpose.permutation);
}

SifNodeParams MmeTransposeNode::getShapeInferenceFunctionUserParams()
{
    return &m_sifMetadata;
}

size_t MmeTransposeNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(m_sifMetadata);
}

NodePtr MmeTransposeNode::createNode(const TensorVector& inputs,
                                     const TensorVector& outputs,
                                     UserParams          userParams,
                                     std::string_view    guid,
                                     std::string_view    name)
{
    auto params = reinterpret_cast<TransposePermutationArray*>(userParams);
    HB_ASSERT_PTR(params);
    return NodePtr(new MmeTransposeNode(inputs[0], outputs[0], *params, name));
}