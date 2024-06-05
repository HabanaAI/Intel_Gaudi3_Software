#include "strided_op_node_utils.h"
#include "defs.h"
#include "handle_memory_reuse.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"
#include "h2d_tensors.h"
#include "node_factory.h"
#include <sstream>

namespace StridedOpUtils
{
// return a string describing synStridedOpParams
std::string stridedOpParamsString(const synStridedOpParams& params, unsigned dim)
{
    std::stringstream ss;
    ss << "base offset: " << params.baseOffset << ", strides: [";
    for (unsigned i = 0; i < dim; i++)
    {
        ss << params.strides[i] << ", ";
    }
    ss << "]";
    return ss.str();
}

bool verifyStridedAccess(const TensorPtr& real, const TensorPtr& alias, const synStridedOpParams& params)
{
    // verify that the last strided element does not exceed the original tensor size
    if (real->isZeroSizedDataTensor() || alias->isZeroSizedDataTensor()) return true;
    uint64_t numOfInputElements = real->getDenseSizeInElements();
    uint64_t lastElementOffset  = 0;
    for (unsigned d = 0; d < alias->getDim(); d++)
    {
        lastElementOffset += params.strides[d] * (alias->getSizeInElements(d) - 1);
    }
    if (params.baseOffset + lastElementOffset >= numOfInputElements)
    {
        LOG_ERR(GC,
                "illegal memory access for strided op. real tensor {} sizes {}. alias tensor {} sizes {}. params: {}",
                real->getName(),
                real->getDimSizesStr(),
                alias->getName(),
                alias->getDimSizesStr(),
                stridedOpParamsString(params, alias->getDim()));
        return false;
    }
    return true;
}

synStridedOpParams createExpandedParams(const synStridedOpParams& params, unsigned dim)
{
    // create expanded view params
    HB_ASSERT(dim < HABANA_DIM_MAX, "cannot expand params");
    synStridedOpParams newParams = {0};
    newParams.baseOffset         = params.baseOffset;
    newParams.strides[0]         = 1;
    for (unsigned i = 0; i < dim; i++)
    {
        newParams.strides[i + 1] = params.strides[i];
    }
    return newParams;
}

synStridedOpParams createReinterpretedParams(const synStridedOpParams& params, unsigned originalDim)
{
    static constexpr unsigned NUM_STRIDES = ARRAY_SIZE(params.strides);
    HB_ASSERT(originalDim < NUM_STRIDES, "cannot expand params!");
    synStridedOpParams ret = {0};
    ret.baseOffset         = params.baseOffset * 2;
    ret.strides[0]         = 1;
    for (unsigned i = 0; i < NUM_STRIDES - 1; ++i)
    {
        ret.strides[i + 1] = params.strides[i] * 2;
    }
    return ret;
}

synStridedOpParams createParamsFromShapeTensors(const TensorPtr& strides, const TensorPtr& offset)
{
    HB_ASSERT_PTR(strides);

    synStridedOpParams params = {0};

    if (strides->isHost2DeviceTensor())
    {
        synDynamicStridedDmaH2dTensor* dynStridesData =
            reinterpret_cast<synDynamicStridedDmaH2dTensor*>(strides->getHostMaxData());

        params.baseOffset = dynStridesData->offset;
        HB_ASSERT(dynStridesData->num_strides <= sizeof(params.strides) / sizeof(params.strides[0]),
                  "num_strides is greater than num of supported strides");
        for (unsigned i = 0; i < dynStridesData->num_strides; i++)
        {
            params.strides[i] = dynStridesData->strides[i];
        }
    }
    else
    {
        HB_ASSERT_PTR(offset);
        HB_ASSERT(strides->isShapeTensor(), "expected dynamic strides tensor to be a shape tensor!");
        HB_ASSERT(offset->isShapeTensor(), "expected dynamic offset tensor to be a shape tensor!");

        params.baseOffset = offset->getSizeInElements(0);
        for (unsigned i = 0; i < strides->getDim(); i++)
        {
            params.strides[i] = strides->getSizeInElements(i);
        }
    }
    return params;
}

uint64_t getLastIndex(const TensorPtr& t, const synStridedOpParams& p)
{
    uint64_t lastIdx = 0;
    for (unsigned i = 0; i < t->getDim(); i++)
    {
        lastIdx += (t->getSizeInElements(i) - 1) * p.strides[i];
    }
    return lastIdx;
}

static std::pair<std::vector<uint64_t>, std::vector<uint64_t>> getShapeAndStrides(const TensorPtr&          t,
                                                                                  const synStridedOpParams& p)
{
    std::vector<uint64_t> shape(t->getDim()), strides(t->getDim());
    for (unsigned dim = 0; dim < t->getDim(); dim++)
    {
        shape[dim]   = t->getSizeInElements(dim);
        strides[dim] = p.strides[dim];
    }
    return std::make_pair(shape, strides);
}

bool isOverlap(const TensorPtr& t1, const TensorPtr& t2, const synStridedOpParams& p1, const synStridedOpParams& p2)
{
    DataRange<uint64_t> r1(p1.baseOffset, p1.baseOffset + getLastIndex(t1, p1) + 1);
    DataRange<uint64_t> r2(p2.baseOffset, p2.baseOffset + getLastIndex(t2, p2) + 1);
    if (!r1.isOverlap(r2)) return false;

    auto [sizes1, strides1] = getShapeAndStrides(t1, p1);
    auto [sizes2, strides2] = getShapeAndStrides(t2, p2);
    return MemoryReuseHandler::isStridedOverlap(sizes1, sizes2, strides1, strides2, p1.baseOffset, p2.baseOffset);
}

bool compareParams(const synStridedOpParams& p1, const synStridedOpParams& p2, unsigned dim)
{
    if (p1.baseOffset != p2.baseOffset) return false;
    for (unsigned i = 0; i < dim; i++)
    {
        if (p1.strides[i] != p2.strides[i]) return false;
    }
    return true;
}

bool isDenseStridedOpParams(const synStridedOpParams& params, const TensorPtr& view)
{
    uint64_t stride = 1;
    for (unsigned d = 0; d < view->getDim(); d++)
    {
        if (stride != params.strides[d]) return false;
        stride *= view->getSizeInElements(d);
    }
    return true;
}

void convertShapeToH2D(NodeList&           nodes,
                       TensorVector&       inputs,
                       TensorVector&       outputs,
                       synStridedOpParams& params,
                       const std::string&  name)
{
    TSize     size     = sizeof(synDynamicStridedDmaH2dTensor) / sizeof(int32_t);
    TensorPtr newInput = createHost2DeviceTensor(syn_type_int32, size, name);
    newInput->setHostOnly();

    TensorPtr stridesTensor = inputs[2];
    TensorPtr offsetTensor  = inputs[3];
    inputs.pop_back();  // remove strides and offset tensors from view input
    inputs.pop_back();

    NodePtr convert = NodeFactory::createNode({stridesTensor, offsetTensor},
                                              {newInput},
                                              nullptr,
                                              NodeFactory::stridedOpsConversionNodeTypeName,
                                              name + "_convert");
    convert->inferOutputsSizes(synDeviceTypeInvalid, true);   // calculate min and max data
    convert->inferOutputsSizes(synDeviceTypeInvalid, false);  // device type not needed
    inputs.push_back(newInput);                               // replace strides shape tensor with H2D tensor
    nodes.push_back(convert);
}

std::tuple<TensorPtr, NodePtr> expandH2DTensor(const TensorPtr& tensor, unsigned dim)
{
    TensorPtr expanded = tensor->clone(false, false, false);
    expanded->setHostOnly();
    expanded->setTensorBuffer(tensor->getData(), tensor->getBufferSizeInBytes(), tensor->getBufferDataType(), true);
    NodePtr expand = NodeFactory::createNode({tensor},
                                             {expanded},
                                             &dim,
                                             NodeFactory::dynamicStridedDmaExpandH2DNodeTypeName,
                                             tensor->getName() + "_expand_dims");
    expand->inferOutputsSizes(synDeviceTypeInvalid, true);   // calculate min and max data
    expand->inferOutputsSizes(synDeviceTypeInvalid, false);  // device type not needed

    return std::tie(expanded, expand);
}

std::tuple<TensorPtr, NodePtr> reinterpretH2DTensor(const TensorPtr& tensor, unsigned factor)
{
    TensorPtr expanded = tensor->clone(false, false, false);
    expanded->setHostOnly();
    expanded->setTensorBuffer(tensor->getData(), tensor->getBufferSizeInBytes(), tensor->getBufferDataType(), true);
    NodePtr expand = NodeFactory::createNode({tensor},
                                             {expanded},
                                             &factor,
                                             NodeFactory::dynamicStridedDmaReinterpretH2DNodeTypeName,
                                             tensor->getName() + "_reinterpret_64_bit");
    expand->inferOutputsSizes(synDeviceTypeInvalid, true);   // calculate min and max data
    expand->inferOutputsSizes(synDeviceTypeInvalid, false);  // device type not needed

    return std::tie(expanded, expand);
}

};  // namespace StridedOpUtils
