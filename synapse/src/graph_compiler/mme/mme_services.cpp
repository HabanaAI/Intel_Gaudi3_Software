#include "mme_services.h"
#include "mme_brain_ifc.h"
#include "compilation_hal_reader.h"
#include "node.h"
#include "synapse_common_types.h"
#include "tensor_shape.h"
#include "transpose_node.h"
#include "types.h"


namespace MmeCommon
{
void MmeAuxTensorHandler::addAuxTensorsForCdParallel(MMENodePtr& mmeNode, unsigned concurrencyLevel)
{
    auto        auxTensorDType  = mmeNode->getOutput(0)->getElementType();
    const auto& outputShape     = mmeNode->getOutput(0)->getShape();
    TensorShape scratchPadShape = getScratchPadShape(mmeNode, outputShape, concurrencyLevel);

    LOG_DEBUG(MME_STACK,
              "Creating scratchpad aux tensor for node {} with size: [{}], dtype: {}",
              mmeNode->getNodeName(),
              toString(scratchPadShape.getMaxSizes(), ','),
              getStringFromSynDataType(auxTensorDType));
    TensorPtr scratchPadAux = std::make_shared<Tensor>(scratchPadShape, auxTensorDType);
    scratchPadAux->setName(mmeNode->getNodeName() + "_aux_scratchpad");
    scratchPadAux->setAsAuxTensor(true);
    scratchPadAux->setTensorInWorkspace();

    TensorShape reduceTensorShape = getReduceShape(concurrencyLevel);
    LOG_DEBUG(MME_STACK,
              "Creating reduce aux tensor for node {} with size: [{}], dtype: {}",
              mmeNode->getNodeName(),
              toString(reduceTensorShape.getMaxSizes(), ','),
              getStringFromSynDataType(auxTensorDType));
    TensorPtr reduceAux = std::make_shared<Tensor>(reduceTensorShape, auxTensorDType);
    reduceAux->setName(mmeNode->getNodeName() + "_aux_reduce");
    reduceAux->setAsAuxTensor(false);
    addDataToReduceTensor(reduceAux, auxTensorDType, concurrencyLevel);

    mmeNode->addInput(TENSOR_AUX_CD_SCRATCHPAD, scratchPadAux, Node::TENSOR_TYPE_DATA, true);
    mmeNode->addInput(TENSOR_AUX_CD_REDUCTION, reduceAux, Node::TENSOR_TYPE_DATA, true);
}

TensorShape MmeAuxTensorHandler::getScratchPadShape(const MMENodePtr&  mmeNode,
                                                    const TensorShape& outputShape,
                                                    unsigned           concurrencyLevel)
{
    auto     maxSizes = outputShape.getMaxSizes();
    auto     minSizes = outputShape.getMinSizes();
    unsigned rank     = mmeNode->getOutput(0)->getDim();
    HB_ASSERT(rank < SYN_MAX_TENSOR_DIM,
              "Tensor of more than 4 dims is not supported yet");  // TODO: support tensors of 5 dims
    maxSizes.at(rank)           = concurrencyLevel;
    minSizes.at(rank)           = concurrencyLevel;
    TensorShape scratchPadShape = outputShape;
    scratchPadShape.setDim(rank + 1);
    scratchPadShape.setMinSize(minSizes.data(), false);
    scratchPadShape.setMaxSize(maxSizes.data());
    return scratchPadShape;
}

TensorShape MmeAuxTensorHandler::getReduceShape(unsigned int concurrencyLevel)
{
    ::SizeArray reduceTensorSize {concurrencyLevel};
    return {1, reduceTensorSize};
}

void MmeAuxTensorHandler::addDataToReduceTensor(TensorPtr&  reduceTensor,
                                                synDataType dataType,
                                                unsigned    concurrencyLevel)
{
    unsigned sizeInBytes = reduceTensor->getTotalSizeInBytes();
    switch (dataType)
    {
        case syn_type_float:
        {
            float* fpBuffer = new float[concurrencyLevel];
            for (int i = 0; i < concurrencyLevel; i++)
            {
                fpBuffer[i] = 1.0f;
            }
            HB_ASSERT(sizeInBytes == concurrencyLevel * sizeof(fpBuffer[0]),
                      "fpBuffer size is different then reduceTensor total size in bytes");
            reduceTensor->setTensorBuffer(fpBuffer, sizeInBytes, dataType, true);
            delete[] fpBuffer;
            break;
        }
        case syn_type_bf16:
        {
            bfloat16* buffer = new bfloat16[concurrencyLevel];
            for (int i = 0; i < concurrencyLevel; i++)
            {
                buffer[i] = 1.0f;
            }
            HB_ASSERT(sizeInBytes == concurrencyLevel * sizeof(buffer[0]),
                      "fpBuffer size is different then reduceTensor total size in bytes");
            reduceTensor->setTensorBuffer(buffer, sizeInBytes, dataType, true);
            delete[] buffer;
            break;
        }
        default:
            HB_ASSERT(0, "DType is not supported for reduce tensor yet.");
    }
}

MmeServices::ePattern MmeServices::matchPattern(const MMENodePtr& mmeNode, const MmeStrategy& strategy)
{
    auto& perforationDim               = mmeNode->getNodeAnnotation().perforationDim;
    bool  cdConcurrencyEn =
        strategy.cdConcurrencyEn == TurnedOn;  // TODO [SW-144531]: integrate cd concurrency with cd perforation

    if (checkTransposeViaGemmPattern(mmeNode))
    {
        return TRANSPOSE_VIA_GEMM;
    }
    else if (GCFG_ENABLE_CD_PARALLEL.value() && perforationDim.has_value() &&
        mmeNode->getMmeBrainIfc()->isCdDim(perforationDim.value()))
    {
        if (!cdConcurrencyEn)
        {
            return CD_PARALLEL;
        }
    }
    else if (cdConcurrencyEn)
    {
        return CD_CONCURRENCY;
    }

    return PATTERNS_NR;
}

void MmeServices::addAuxTensorToNode(MMENodePtr& mmeNode, const MmeStrategy& strategy)
{
    switch (matchPattern(mmeNode, strategy))
    {
        case CD_PARALLEL:
            getAuxHandler().addAuxTensorsForCdParallel(mmeNode, CompilationHalReader::getHalReader()->getNumDcores());
            break;
        case TRANSPOSE_VIA_GEMM:
            getAuxHandler().addUnitMatrixToNode(mmeNode, getDtypeForTranspose(*mmeNode->getInput(TENSOR_IFM)));
        default:
            break;
    }
}

void MmeServices::adjustDcoreRoisForCdParallel(MMENodePtr& mmeNode, const MmeStrategy& strategy)
{
    if (matchPattern(mmeNode, strategy) == CD_PARALLEL)
    {
        auto&   dcoreROIs   = mmeNode->getNodeAnnotation().m_dcoreROIs;
        auto    cdDim       = mmeNode->getMmeBrainIfc()->getCDDims().back();
        TOffset dcoreOffset = 0;
        for (auto& dcoreRoi : dcoreROIs)
        {
            dcoreRoi.size[cdDim]       = 1;
            dcoreRoi.baseOffset[cdDim] = dcoreOffset;
            dcoreOffset += 1;
        }
    }
}

// Returns if transpose via gemm (unit-matrix transpose) is needed.
// transpose via gemm is limited to 4D operation by design.
// in addition we will benefit from it over native transpose on specific dtypes & fcd size is large enough.
bool MmeServices::checkTransposeViaGemmPattern(const MMENodePtr& mmeNode)
{
    const TensorPtr& outputTensor    = mmeNode->getOutput(0);
    synDataType      elementDataType = outputTensor->getElementType();
    bool typeIsSupportedForTranspose = elementDataType == syn_type_fp8_143 || elementDataType == syn_type_fp8_152 ||
                                       elementDataType == syn_type_int8 || elementDataType == syn_type_uint8 ||
                                       elementDataType == syn_type_bf16 || elementDataType == syn_type_fp16 ||
                                       elementDataType == syn_type_int16 || elementDataType == syn_type_uint16;

    bool tensorDimSupported = outputTensor->getDim() <= 4;
    bool fcdSizeLargeEnough =
        outputTensor->getSizeInBytes(0) > CompilationHalReader::getHalReader()->getCacheLineSizeInBytes();
    return CompilationHalReader::getHalReader()->getDeviceType() == synDeviceGaudi3 && GCFG_ENABLE_TRANSPOSE_VIA_GEMM.value() && 
           typeIsSupportedForTranspose && tensorDimSupported && fcdSizeLargeEnough;
}
void MmeAuxTensorHandler::addUnitMatrixToNode(MMENodePtr& mmeNode, synDataType dtype)
{
    std::shared_ptr<MmeTransposeNode> xposeNode = std::static_pointer_cast<MmeTransposeNode>(mmeNode);
    xposeNode->addInput(TENSOR_UNIT_MATRIX, createSparseUnitTensor(dtype));
    xposeNode->setTransposeViaGemm(true);
}
TensorPtr MmeAuxTensorHandler::createSparseUnitTensor(synDataType dtype)
{
    char*              oneUntyped = getOneByDtype(dtype);
    static const TSize sizes      = {1};
    TensorPtr          unitTensor = std::make_shared<Tensor>(1, &sizes, dtype, oneUntyped);
    unitTensor->setShouldFreeBuffer(true);
    unitTensor->setMemorySectionID(MEMORY_ID_RESERVED_FOR_PROGRAM_DATA);
    unitTensor->setName("UnitTensorForGemmXpose");
    unitTensor->setUnitTensor();
    return unitTensor;
}
char* MmeAuxTensorHandler::getOneByDtype(synDataType dataType)
{
    static const float oneFloat   = 1.0;
    char*              oneUntyped = nullptr;
    if (dataType == syn_type_fp16)
    {
        fp16_t* oneTyped = new fp16_t[1];
        *oneTyped        = fp16_t(oneFloat,
                           MmeCommon::RoundingMode::RoundToNearest,
                           EXPONENT_BIAS_FP16_15,
                           0,
                           false,
                           false,
                           false,
                           MmeCommon::InfNanMode::e_mme_no_inf_nan);
        oneUntyped       = (char*)oneTyped;
    }
    else if (dataType == syn_type_ufp16)
    {
        ufp16_t* oneTyped = new ufp16_t[1];
        *oneTyped         = ufp16_t(oneFloat,
                            MmeCommon::RoundingMode::RoundToNearest,
                            EXPONENT_BIAS_UFP16_31,
                            0,
                            false,
                            false,
                            false,
                            MmeCommon::InfNanMode::e_mme_no_inf_nan);
        oneUntyped        = (char*)oneTyped;
    }
    else if (dataType == syn_type_bf16)
    {
        bf16_t* oneTyped = new bf16_t[1];
        *oneTyped        = bf16_t(oneFloat);
        oneUntyped       = (char*)oneTyped;
    }
    else if (dataType == syn_type_fp8_152)
    {
        fp8_152_t* oneTyped = new fp8_152_t[1];
        *oneTyped           = fp8_152_t(oneFloat,
                              MmeCommon::RoundingMode::RoundToNearest,
                              EXPONENT_BIAS_FP8_152_15,
                              0,
                              false,
                              false,
                              false,
                              false,
                              MmeCommon::InfNanMode::e_mme_no_inf_nan);
        oneUntyped          = (char*)oneTyped;
    }
    else if (dataType == syn_type_fp8_143)
    {
        fp8_143_t* oneTyped = new fp8_143_t[1];
        *oneTyped           = fp8_143_t(oneFloat,
                              MmeCommon::RoundingMode::RoundToNearest,
                              EXPONENT_BIAS_FP8_152_15,
                              0,
                              false,
                              false,
                              false,
                              false,
                              MmeCommon::InfNanMode::e_mme_no_inf_nan);
        oneUntyped          = (char*)oneTyped;
    }
    else if (dataType == syn_type_float)
    {
        oneUntyped = (char*)(&oneFloat);
    }
    else
    {
        HB_ASSERT(0, "createSparseUnitTensor got unsupported data-type");
        return nullptr;
    }
    return oneUntyped;
}

// MME cannot process u\int tensors , return the equivalent dtype in fp.
// use FP8_152 for u\int8 tensors, fp16 for u\int16 tensors, and float for u\int32
synDataType MmeServices::getDtypeForTranspose(const Tensor& tensor)
{
    synDataType dataType = tensor.getElementType();
    if (dataType == syn_type_uint8 || dataType == syn_type_int8)
    {
        return syn_type_fp8_152;
    }
    
    else if (dataType == syn_type_uint16 || dataType == syn_type_int16)
    {
        return syn_type_fp16;
    }
    else if (dataType == syn_type_uint32 || dataType == syn_type_int32)
    {
        return syn_type_float;
    }
    return dataType;
}
}  // namespace MmeCommon
