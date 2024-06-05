#include "gc_interface_utils.hpp"
#include "habana_graph.h"
#include "data_type_utils.h"
#include "node_utils.h"
#include "transpose_node.h"

void createFuserTensorAttributes(pFuserTensorAttributes attributes, TensorPtr gcTensor)
{
    attributes->isNotNeeded    = gcTensor->isNotNeeded();
    attributes->isInitialized  = gcTensor->isStaticParam() && gcTensor->isBound();
    synTensorType gcTensorType = gcTensor->getTensorType();
    // synTensorType enum values are equal to FuserTensorType values
    attributes->type = (gcapi::FuserTensorType)gcTensorType;
}

void createFuserSection(pFuserSection section, TensorPtr gcTensor)
{
    if (gcTensor->isPersistent())
    {
        section->type   = gcapi::SECTION_PERSISTENT;
        section->offset = gcTensor->getMemorySectionOffset();
        section->id     = gcTensor->getMemorySectionID();
    }
    else if (gcTensor->isPartOfRMWSection())
    {
        section->type   = gcapi::SECTION_RMW;
        section->offset = gcTensor->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.value();
        section->id     = gcTensor->getTensorAnnotation().nonPersistentSectionInfo.sectionId.value();
    }
    else
    {
        section->type = gcapi::SECTION_WORKSPACE;
        section->offset = 0;  // for workspace sections offset is 0;
        section->id     = MEMORY_ID_RESERVED_FOR_WORKSPACE;
    }
}

template<typename TensorVersion>
void internalCreateFuserTensor(std::shared_ptr<TensorVersion> fuserTensor, TensorPtr gcTensor)
{
    fuserTensor->uniqueIdentifier = gcTensor->getId();
    fuserTensor->dataType         = toGlueCodeDataType(gcTensor->getElementType());
    fuserTensor->geometry.dims    = gcTensor->getDim();

    auto sizesArrayLength = sizeof(fuserTensor->geometry.maxSizes) / sizeof(fuserTensor->geometry.maxSizes[0]);
    gcTensor->getAllSizesInElements(fuserTensor->geometry.sizes, sizesArrayLength);
    gcTensor->getAllMinimalSizesInElements(fuserTensor->geometry.minSizes, sizesArrayLength);
    gcTensor->getAllStridesInBytes(fuserTensor->strides, sizesArrayLength);

    QuantizationData quantizationParams      = gcTensor->getQuantizationParams();
    fuserTensor->quantizationParam.scale     = quantizationParams.scale();
    fuserTensor->quantizationParam.zeroPoint = quantizationParams.zp();
    if (gcTensor->getElementType() == syn_type_fp8_143)
    {
        fuserTensor->quantizationParam.fp8bias = quantizationParams.expBias();
    }
    fuserTensor->Reducible = gcTensor->isReductionEnabled();

    if (gcTensor->isStaticParam())
    {
        fuserTensor->pData = gcTensor->getAddress();
    }
    else
    {
        fuserTensor->pData = nullptr;
    }

    pFuserSection section = std::make_shared<FuserSectionType>();
    createFuserSection(section, gcTensor);
    fuserTensor->section = *section;

    pFuserTensorAttributes attributes = std::make_shared<FuserTensorAttributesType>();
    createFuserTensorAttributes(attributes, gcTensor);
    fuserTensor->attributes = *attributes;
}
// template specialization for above template method - allowing it to be defined in cpp
template void internalCreateFuserTensor<FuserTensorTypeV4>(FuserTensorPtrV4 fuserTensor, TensorPtr gcTensor);

template<typename NodeVersion, typename TensorVersion, typename EdgeVersion>
void internalCreateFuserNodeEdgesAndTensors(std::shared_ptr<NodeVersion> fuserNode,
                                            TensorVector&                tensors,
                                            bool                         isInputEdge,
                                            PermutationsVector&          permutations)
{
    unsigned tensorIndex     = 0;
    bool     gotPermutations = !permutations.empty();
    std::vector<std::shared_ptr<TensorVersion>> createdFuserTensors;
    createdFuserTensors.reserve(tensors.size());
    for (auto& tensor : tensors)
    {
        std::shared_ptr<TensorVersion> fuserTensor = std::shared_ptr<TensorVersion>(nullptr);
        if (tensor != nullptr)
        {
            // first check if tensor was already created
            auto iter = std::find_if(
                createdFuserTensors.begin(),
                createdFuserTensors.end(),
                [&](const std::shared_ptr<TensorVersion>& t) { return tensor->getId() == t->uniqueIdentifier; });
            if (iter == createdFuserTensors.end())
            {
                // create new tensor
                fuserTensor = std::make_shared<TensorVersion>();
                createFuserTensor<TensorVersion>(fuserTensor, tensor);
                /*
                 * set permutations.
                 * don't set permutations for aux and output shape tensors which are inputs.
                 * This is same as glue code Tensor creation logic in tpc node.
                 */
                if (gotPermutations && !(isInputEdge && tensor->isTensorAuxOrShapeOutput()))
                {
                    setCommonTensorPermutations<TensorVersion>(fuserTensor, permutations[tensorIndex]);
                }
                createdFuserTensors.push_back(fuserTensor);
            }
            else
            {
                // tensor already created
                fuserTensor = *iter;
            }
        }
        EdgeVersion edge;
        edge.tensor     = fuserTensor;
        edge.targetNode = std::weak_ptr<NodeVersion>();  // set to empty since the target node is not known or exists
        if (isInputEdge)
        {
            fuserNode->inputEdges.push_back(edge);
        }
        else
        {
            fuserNode->outputEdges.push_back(edge);
        }
        tensorIndex++;
    }
}
// template specialization for above template method - allowing it to be defined in cpp
template void internalCreateFuserNodeEdgesAndTensors<FuserNodeTypeV4, FuserTensorTypeV4, FuserEdgeTypeV4>(
    FuserNodePtrV4        fuserNode,
    TensorVector&       tensors,
    bool                isInputEdge,
    PermutationsVector& permutations);

template<typename NodeVersion>
void internalCreateFuserNode(const HabanaGraph& g, std::shared_ptr<NodeVersion> fuserNode, const NodePtr node)
{
    std::string         guid              = node->getGUID();
    UserParams          params            = nullptr;
    unsigned            paramsSize        = 0;
    const std::string   nodeName          = node->getNodeName();
    const unsigned      maxNodeNameLength = sizeof(fuserNode->nodeName);

    if (g.runsOnTPC(node))
    {
        std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
        HB_ASSERT_PTR(tpcNode);

        params     = tpcNode->getParams();
        paramsSize = tpcNode->getParamsSize();
    }
    else
    {
        params     = (UserParams)node->getParamsRawData().data();
        paramsSize = node->getParamsRawData().size();
    }

    // TODO [SW-124551] - remove this temp WA for sending transpose node params to fuser
    if (node->getNodeType() == Node::TYPE_INTERNAL_TRANSPOSE)
    {
        std::shared_ptr<TransposeNode> internalTransposeNode = std::dynamic_pointer_cast<TransposeNode>(node);

        if (internalTransposeNode)
        {
            auto tensorDim = internalTransposeNode->permutation().size();

            if (tensorDim < 5)
            {
                synTransposeParams transposeParams;
                memset(&transposeParams, 0, sizeof(transposeParams));
                transposeParams.tensorDim = tensorDim;
                std::copy(internalTransposeNode->permutation().begin(),
                          internalTransposeNode->permutation().end(),
                          transposeParams.permutation);
                paramsSize = sizeof(synTransposeParams);
                params     = (UserParams)internalTransposeNode->getParamsRawData().data();
                memcpy(params,(void*) &transposeParams ,paramsSize );
            }
        }
    }

    // Filling all node params
    fuserNode->nodeParams = params;
    fuserNode->paramsSize = paramsSize;
    copyString(guid, fuserNode->guid, guid.length() + 1);
    synNodeId id = node->getId();
    HB_ASSERT(id <= std::numeric_limits<uint32_t>::max(),
              "node {} id exceeds 32 bit - can't send to CommonIR",
              nodeName);
    fuserNode->uniqueIdentifier      = id;
    fuserNode->controlEdgesToNode    = std::set<unsigned> {};
    fuserNode->useDeterministic      = node->getDeterministic();
    fuserNode->isShapeManipulationOp = false;
    copyStringSafe(nodeName, fuserNode->nodeName, nodeName.length() + 1, maxNodeNameLength);
    fuserNode->originalComplexGuidId = node->getNodeAnnotation().originalComplexGuidId;
    copyString(node->getNodeAnnotation().originalComplexGuid,
               fuserNode->originalComplexGuid,
               tpc_lib_api::MAX_NODE_NAME);
}
// template specialization for above template method - allowing it to be defined in cpp
template void internalCreateFuserNode<FuserNodeTypeV4>(const HabanaGraph& g, FuserNodePtrV4 fuserNode, const NodePtr n);

template<typename TensorVersion>
bool areStridesValid(std::shared_ptr<TensorVersion> fuserTensor)
{
    // strides are valid if they are non-zero
    unsigned stridesArraySize = sizeof(fuserTensor->geometry.maxSizes);
    auto     zeroStrides      = std::vector<unsigned>(stridesArraySize / sizeof(unsigned), 0);
    return memcmp(zeroStrides.data(), fuserTensor->strides, stridesArraySize) != 0;
}

template<typename TensorVersion>
void internalCreateGCTensor(TensorPtr gcTensor, std::shared_ptr<TensorVersion> fuserTensor)
{
    synDataType tensorDataType = translateTensorDataType(fuserTensor->dataType);
    gcTensor->setElementType(tensorDataType);

    std::vector<TStride> strides;
    TStride*             stridesPtr = nullptr;
    if (areStridesValid(fuserTensor))
    {
        // set strides if they are valid (non-zero)
        auto strideNum = sizeof(fuserTensor->geometry.maxSizes) / sizeof(fuserTensor->geometry.maxSizes[0]);
        strides.assign(fuserTensor->strides, fuserTensor->strides + strideNum);
        stridesPtr = strides.data();
    }

    gcTensor->reshape(fuserTensor->geometry.dims,
                      fuserTensor->geometry.maxSizes,
                      stridesPtr,
                      fuserTensor->geometry.minSizes);
    gcTensor->setProp(synTensorPropGeometryMin);
    gcTensor->setProp(synTensorPropGeometryMax);
    gcTensor->setProp(synTensorPropGeometryDim);
    gcTensor->setProp(synTensorPropDeviceLayout);

    // Note - currently doesn't support per-channel
    QuantizationData quantizationParams(tensorDataType);
    quantizationParams.setScale(fuserTensor->quantizationParam.scale);
    quantizationParams.setExpBias(fuserTensor->quantizationParam.fp8bias);
    gcTensor->setQuantizationParams(quantizationParams);
    gcTensor->setProp(synTensorPropFpQuantMetadata);

    gcTensor->getTensorAnnotation().tensorReductionInfo.isReductionEnabled = fuserTensor->Reducible;
    // get tensor data if exist
    if (fuserTensor->attributes.isInitialized)
    {
        HB_ASSERT(fuserTensor->pData != nullptr, "tensor data is null");
        // copy the data since it will be deallocated by the user (complex GUID lib for example)
        gcTensor->setTensorBuffer(const_cast<void*>(fuserTensor->pData),
                                  gcTensor->getTotalSizeInBytes(),
                                  tensorDataType,
                                  true);
        gcTensor->setAsStaticParam();
        gcTensor->setProp(synTensorPropHostPtr);
    }
    else if (fuserTensor->attributes.type == gcapi::HOST_TO_DEVICE_TENSOR)
    {
        // reshape above sets device size according to # of elements,
        // but we need to allocate buffer for two sets of data (max/min)
        gcTensor->setDeviceSizeInBytes(gcTensor->getTotalSizeInBytes() * 2);
        gcTensor->bind(new char[gcTensor->getTotalSizeInBytes()], true);
        gcTensor->setAsDataTypeMatchData();
        gcTensor->setProp(synTensorPropHostPtr);
    }

    gcTensor->setTensorType(synTensorType(fuserTensor->attributes.type));
}
// template specialization for above template method - allowing it to be defined in cpp
template void internalCreateGCTensor<FuserTensorTypeV4>(TensorPtr gcTensor, FuserTensorPtrV4 fuserTensor);

template<typename TensorVersion>
uint64_t internalGetCommonTensorSizeInBytes(std::shared_ptr<TensorVersion> fuserTensor)
{
    uint64_t sizeInBytes      = 0;
    auto     sizesArrayLength = sizeof(fuserTensor->geometry.maxSizes) / sizeof(fuserTensor->geometry.maxSizes[0]);
    if (areStridesValid(fuserTensor))
    {  // strides are non-zero
        sizeInBytes = fuserTensor->strides[sizesArrayLength - 1];
    }
    else
    {  // get size in bytes according to actual sizes
        synDataType tensorDtype = translateTensorDataType(fuserTensor->dataType);
        sizeInBytes             = dataTypeSizeInBytes(tensorDtype);  // Doesn't support 4 bit data types
        for (unsigned i = 0; i < sizesArrayLength; i++)
        {
            sizeInBytes *= fuserTensor->geometry.maxSizes[i];
        }
    }
    return sizeInBytes;
}
template uint64_t internalGetCommonTensorSizeInBytes<FuserTensorTypeV4>(FuserTensorPtrV4 fuserTensor);

template<typename TensorVersion>
void internalSetCommonTensorPermutation(std::shared_ptr<TensorVersion> fuserTensor, gc::Permutation& permutation)
{
    auto sizesArrayLength        = sizeof(fuserTensor->geometry.maxSizes) / sizeof(fuserTensor->geometry.maxSizes[0]);
    auto permutationValuesVector = permutation.getValues();
    for (unsigned dim = 0; dim < sizesArrayLength; ++dim)
    {
        // treat dims outside the tensor size as identity as well
        if (permutation.isIdentity() || dim >= fuserTensor->geometry.dims)
        {
            fuserTensor->geometry.permutation[dim] = dim;
        }
        else
        {
            fuserTensor->geometry.permutation[dim] = (unsigned)(permutationValuesVector[dim]);
        }
    }
}
template void internalSetCommonTensorPermutation<FuserTensorTypeV4>(FuserTensorPtrV4   fuserTensor,
                                                                    gc::Permutation& permutation);