#include "graph_compiler/utils.h"
#include "gaudi/mme.h"
#include "synapse_common_types.h"
#include "test_utils.h"
#include "cpu_calculator.h"
#include "infra/defs.h"
#include "data_type_utils.h"
#include "tensor_validator.inl"

#include "mme_reference/tensor_comparator.h"
#include "mme_reference/mme_reference.h"
#include "mme_reference/data_types/non_standard_dtypes.h"

static bool diffTensors(const MmeSimTensor* t0,
                        const MmeSimTensor* t1,
                        int*                diffIdx,
                        bool                fp,
                        float               absTol,
                        float               relTol,
                        bool                skipFirst)
{
    bool ret = false;
    assert(t0->getDim() == t1->getDim());
    assert(t0->getElementSize() == t1->getElementSize());
    for (int i = 0; i < t0->getDim(); i++)
    {
        assert(t0->getSize(i) == t1->getSize(i));
    }

    int idx[MmeSimTensor::c_tensorMaxDim] = {0};
    if (diffIdx)
    {
        memcpy(idx, diffIdx, sizeof(int) * t0->getDim());
    }
    int      dim  = -1;
    bool     diff = false;
    unsigned es   = t0->getElementSize();
    assert(es == 2 || es == 4);

    while (dim < t0->getDim())
    {
        if (dim < 0)
        {
            if (!skipFirst)
            {
                char* e0 = t0->getElementAt(idx);
                char* e1 = t1->getElementAt(idx);
                if (fp)
                {
                    union
                    {
                        float    f;
                        uint16_t w[2];
                        uint32_t dw;
                    } f0, f1;
                    if (es == 2)
                    {
                        f0.w[0] = 0;
                        f0.w[1] = *(uint16_t*)e0;
                        f1.w[0] = 0;
                        f1.w[1] = *(uint16_t*)e1;
                    }
                    else
                    {
                        f0.dw = *(uint32_t*)e0;
                        f1.dw = *(uint32_t*)e1;
                    }

                    if (!std::isnan(f0.f) && !std::isnan(f1.f))
                    {
                        float epsilon = std::abs(f0.f - f1.f);
                        diff          = (epsilon > (absTol + (relTol * std::min(std::abs(f0.f), std::abs(f1.f)))));
                    }
                    else
                    {
                        diff = (std::isnan(f0.f) != std::isnan(f1.f));
                    }
                }
                else
                {
                    diff = (((es == 2) && ((*(uint16_t*)e0) != (*(uint16_t*)e1))) ||
                            ((es == 4) && ((*(uint32_t*)e0) != (*(uint32_t*)e1))));
                }

                if (diff)
                {
                    if (diffIdx)
                    {
                        memcpy(diffIdx, idx, t0->getDim() * sizeof(int));
                    }
                    return true;
                }
            }

            skipFirst = false;
            dim++;
        }
        else
        {
            idx[dim]++;
            if (idx[dim] >= t0->getSize(dim))
            {
                idx[dim] = 0;
                dim++;
            }
            else
            {
                dim = -1;
            }
        }
    }

    return ret;
}

float getIndexValue(const TSize* sizes, const CoordArray& wrongIdx, synDataType type, void* buffer)
{
    if (type == syn_type_bf16)
    {
        return (float)(getValueFromBuffer<uint16_t>(sizes, wrongIdx.data(), 4, buffer));
    }
    else
    {
        return getValueFromBuffer<float>(sizes, wrongIdx.data(), 4, buffer);
    }
}

static MmeCommon::EMmeDataType
getDataType(synDataType m_dataType, bool isInput = false, synDeviceType deviceType = synDeviceGaudi)
{
    switch (m_dataType)
    {
        case syn_type_float:
            return (isInput && deviceType == synDeviceGaudi2) ? MmeCommon::e_type_fp32_ieee : MmeCommon::e_type_fp32;
        case syn_type_hb_float:
            return MmeCommon::e_type_fp32;
        case syn_type_bf16:
            return MmeCommon::e_type_bf16;
        case syn_type_fp16:
            return MmeCommon::e_type_fp16;
        case syn_type_fp8_143:
            return MmeCommon::e_type_fp8_143;
        case syn_type_fp8_152:
            return MmeCommon::e_type_fp8_152;
        case syn_type_tf32:
            return MmeCommon::e_type_tf32;
        default:
            assert(0 && "Unsupported reference code type");
            return MmeCommon::EMmeDataType::e_types_nr;
    }
}

static MmeCommon::EMmeOpType convertOpType(ERepefenceOp op, bool isBatchGemm)
{
    switch (op)
    {
        case REFERENCE_OP_FWD:
            return MmeCommon::e_mme_fwd;
        case REFERENCE_OP_DEDX:
            return isBatchGemm ? MmeCommon::e_mme_abt : MmeCommon::e_mme_dedx;
        case REFERENCE_OP_DEDW:
            return isBatchGemm ? MmeCommon::e_mme_atb : MmeCommon::e_mme_dedw;
        case REFERENCE_OP_AB:
            return MmeCommon::e_mme_ab;
        case REFERENCE_OP_ABT:
            return MmeCommon::e_mme_abt;
        case REFERENCE_OP_ATB:
            return MmeCommon::e_mme_atb;
        case REFERENCE_OP_ATBT:
            return MmeCommon::e_mme_atbt;
        case REFERENCE_OP_TRANSPOSED_DEDX:
            return MmeCommon::e_mme_transposed_dedx;
        default:
            assert(0 && "Unsupported reference op type");
            return MmeCommon::e_mme_fwd;
    }
}

static MmeCommon::ChipType convertDeviceType(synDeviceType deviceType)
{
    switch (deviceType)
    {
        case synDeviceGaudi:
            return MmeCommon::e_mme_Gaudi;
        case synDeviceGaudi2:
            return MmeCommon::e_mme_Gaudi2;
        case synDeviceGaudi3:
            return MmeCommon::e_mme_Gaudi3;
        default:
            assert(0 && "Unsupported device type");
            return MmeCommon::e_mme_Gaudi;
    }
}

static void
synToMmeParams(const synConvolution3DParams& convParams, const synTensorDescriptor& wDesc, ConvolutionParams& params)
{
    params.relu          = convParams.activation.reluEnable;
    params.dim           = wDesc.m_dims == MAX_DIMENSIONS_NUM ? 3 : 2;
    params.convStride[0] = convParams.stride[CONV_STRIDE_WIDTH];
    params.convStride[1] = convParams.stride[CONV_STRIDE_HEIGHT];
    params.convStride[2] = convParams.stride[CONV_STRIDE_DEPTH];
    params.dilation[0]   = convParams.dilation[CONV_DIL_WIDTH];
    params.dilation[1]   = convParams.dilation[CONV_DIL_HEIGHT];
    params.dilation[2]   = convParams.dilation[CONV_DIL_DEPTH];
    params.padding[0]    = convParams.padding[CONV_PAD_LEFT];
    params.padding[1]    = convParams.padding[CONV_PAD_TOP];
    params.padding[2]    = convParams.padding[CONV_PAD_FRONT];
}

static void getTestTolerance(int type, float& abstol, float& reltol)
{
    if (type & syn_type_bf16)
    {
        abstol = 0.000001f;
        reltol = 0.015f;
    }
    else
    {
        abstol = 0.0000001f;
        reltol = 0.0015f;
    }
}

static bool innerCheckResult(const synTensorDescriptor& desc, char* firstData, char* secondData, CoordArray& outIndex)
{
    MmeSimTensor t0((int*)desc.m_sizes, desc.m_dims, getDataType(desc.m_dataType), firstData);
    MmeSimTensor t1((int*)desc.m_sizes, desc.m_dims, getDataType(desc.m_dataType), secondData);

    float abstol = 0, reltol = 0;
    getTestTolerance(desc.m_dataType, abstol, reltol);

    return !diffTensors(&t0, &t1, outIndex.data(), true, abstol, reltol, false);
}

static bool isDedxOperation(ERepefenceOp op)
{
    return op == REFERENCE_OP_DEDX || op == REFERENCE_OP_TRANSPOSED_DEDX;
}

static void calculateMmeOp(const synTensorDescriptor&    xDesc,
                           char*                         xData,
                           const synTensorDescriptor&    wDesc,
                           char*                         wData,
                           const synTensorDescriptor&    yDesc,
                           char*                         yData,
                           const synConvolution3DParams& convParams,
                           ERepefenceOp                  op,
                           synDeviceType                 deviceType,
                           MmeCommon::RoundingMode       roundingMode)
{
    ConvolutionParams params;
    synToMmeParams(convParams, wDesc, params);

    MmeSimTensor x((MmeCommon::SizeArray&)xDesc.m_sizes,
                   xDesc.m_dims,
                   getDataType(xDesc.m_dataType, !isDedxOperation(op), deviceType),
                   xData,
                   QuantizationData::getDefaultExpBias(xDesc.m_dataType));
    MmeSimTensor w((MmeCommon::SizeArray&)wDesc.m_sizes,
                   wDesc.m_dims,
                   getDataType(wDesc.m_dataType, (op != REFERENCE_OP_DEDW), deviceType),
                   wData,
                   QuantizationData::getDefaultExpBias(wDesc.m_dataType));
    MmeSimTensor y((MmeCommon::SizeArray&)yDesc.m_sizes,
                   yDesc.m_dims,
                   getDataType(yDesc.m_dataType, (isDedxOperation(op) || op == REFERENCE_OP_DEDW), deviceType),
                   yData,
                   QuantizationData::getDefaultExpBias(yDesc.m_dataType));

    MmeSimTensor* output = isDedxOperation(op) ? &x : (op == REFERENCE_OP_DEDW ? &w : &y);
    MmeSimTensor  acc(output->getSizes(), output->getDim(), MmeCommon::e_type_fp32);

    CPUCalculator calculator(convertDeviceType(deviceType), Mme::c_mme_max_tensor_dims, Mme::c_mme_max_conv_dims);
    calculator.doConvolution(acc, x, w, y, params, convertOpType(op, false), roundingMode);

    pMMESimTensor                 pOutRefTensor = std::make_shared<MmeSimTensor>(*output);
    std::shared_ptr<MmeSimTensor> sharedAcc     = std::make_shared<MmeSimTensor>(acc);
    calculator.doActivation(pOutRefTensor, sharedAcc, nullptr, nullptr, convParams.activation.reluEnable, roundingMode);
}

static void calculateMmeBatchGemmOp(const synTensorDescriptor& xDesc,
                                    char*                      xData,
                                    const synTensorDescriptor& wDesc,
                                    char*                      wData,
                                    const synTensorDescriptor& yDesc,
                                    char*                      yData,
                                    ERepefenceOp               op,
                                    synDeviceType              deviceType,
                                    MmeCommon::RoundingMode    roundingMode)
{
    MmeSimTensor  x((MmeCommon::SizeArray&)xDesc.m_sizes,
                   xDesc.m_dims,
                   getDataType(xDesc.m_dataType, !isDedxOperation(op), deviceType),
                   xData,
                   QuantizationData::getDefaultExpBias(xDesc.m_dataType));
    MmeSimTensor  w((MmeCommon::SizeArray&)wDesc.m_sizes,
                   wDesc.m_dims,
                   getDataType(wDesc.m_dataType, (op != REFERENCE_OP_DEDW), deviceType),
                   wData,
                   QuantizationData::getDefaultExpBias(wDesc.m_dataType));
    MmeSimTensor  y((MmeCommon::SizeArray&)yDesc.m_sizes,
                   yDesc.m_dims,
                   getDataType(yDesc.m_dataType, (isDedxOperation(op) || op == REFERENCE_OP_DEDW), deviceType),
                   yData,
                   QuantizationData::getDefaultExpBias(yDesc.m_dataType));
    MmeSimTensor* output = isDedxOperation(op) ? &x : (op == REFERENCE_OP_DEDW ? &w : &y);

    std::shared_ptr<MmeSimTensor> acc =
        std::make_shared<MmeSimTensor>(output->getSizes(), output->getDim(), MmeCommon::EMmeDataType::e_type_fp32);

    CPUCalculator calculator(convertDeviceType(deviceType), Mme::c_mme_max_tensor_dims, Mme::c_mme_max_conv_dims);
    calculator.doBatchGemm(*acc,
                           (op == REFERENCE_OP_DEDX) ? y : x,
                           (op == REFERENCE_OP_DEDW) ? y : w,
                           convertOpType(op, true),
                           roundingMode);

    pMMESimTensor pOutRefTensor = std::make_shared<MmeSimTensor>(*output);
    calculator.doActivation(pOutRefTensor, acc, nullptr, nullptr, false, roundingMode);
}

static char* allocateByDesc(const synTensorDescriptor& desc)
{
    uint32_t tensorSize = multiplyElements(desc.m_sizes, desc.m_sizes + desc.m_dims);

    tensorSize *= getElementSizeInBytes(desc.m_dataType);

    return new char[tensorSize];
}

static unsigned getCDSize(const MmeCommon::EMmeOpType op, MmeSimTensor& x, MmeSimTensor& w, MmeSimTensor& y)
{
    const MmeCommon::SizeArray& wSizes      = w.getSizes();
    unsigned                    filterSizes = multiplyElements(std::next(std::next(wSizes.begin())), wSizes.end());
    switch (op)
    {
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_abt:
            return x.getSize(0);
            break;
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_atbt:
            return x.getSize(1);
            break;
        case MmeCommon::e_mme_fwd:
            return x.getSize(0) * filterSizes;
        case MmeCommon::e_mme_dedx:
        case MmeCommon::e_mme_transposed_dedx:
            return y.getSize(0) * filterSizes;
            break;
        case MmeCommon::e_mme_dedw:
        {
            const MmeCommon::SizeArray& ySizes = y.getSizes();
            return multiplyElements(std::next(ySizes.begin()), ySizes.end());
            break;
        }
        default:
            MME_ASSERT(0, "invalid op");
    }
    return 0;
}

bool applyPearsonCompare(void* result, void* refResult, synDataType dataType, uint64_t numElements)
{
    switch (dataType)
    {
        case syn_type_bf16:
            validateResult(static_cast<bfloat16*>(refResult), static_cast<bfloat16*>(result), numElements);
            break;
        case syn_type_fp16:
            validateResult(static_cast<fp16_t*>(refResult), static_cast<fp16_t*>(result), numElements);
            break;
        case syn_type_float:
            validateResult(static_cast<float*>(refResult), static_cast<float*>(result), numElements);
            break;
        default:
            MME_ASSERT(0, "Data type is unsupported for Pearson comparison");
    }
    return true;
}

bool checkMmeOp(const synTensorDescriptor&    xDesc,
                char*                         xData,
                const synTensorDescriptor&    wDesc,
                char*                         wData,
                const synTensorDescriptor&    yDesc,
                char*                         yData,
                const synConvolution3DParams& convParams,
                ERepefenceOp                  op,
                CoordArray&                   outIndex,
                synDeviceType                 deviceType,
                float*                        expectedResult,
                bool                          usePearsonCompare,
                MmeCommon::RoundingMode       roundingMode)
{
    bool ret = false;

    MmeSimTensor x((int*)xDesc.m_sizes,
                   xDesc.m_dims,
                   getDataType(xDesc.m_dataType, !isDedxOperation(op), deviceType),
                   xData);
    MmeSimTensor w((int*)wDesc.m_sizes,
                   wDesc.m_dims,
                   getDataType(wDesc.m_dataType, op != REFERENCE_OP_DEDW, deviceType),
                   wData);
    MmeSimTensor y((int*)yDesc.m_sizes,
                   yDesc.m_dims,
                   getDataType(yDesc.m_dataType, (isDedxOperation(op) || op == REFERENCE_OP_DEDW), deviceType),
                   yData);

    char*                      result    = nullptr;
    char*                      refResult = nullptr;
    const synTensorDescriptor* outDesc   = nullptr;
    if (op == REFERENCE_OP_FWD)
    {
        result    = yData;
        refResult = allocateByDesc(yDesc);
        yData     = refResult;
        outDesc   = &yDesc;
    }
    else if (isDedxOperation(op))
    {
        result    = xData;
        refResult = allocateByDesc(xDesc);
        xData     = refResult;
        outDesc   = &xDesc;
    }
    else if (op == REFERENCE_OP_DEDW)
    {
        result    = wData;
        refResult = allocateByDesc(wDesc);
        wData     = refResult;
        outDesc   = &wDesc;
    }

    calculateMmeOp(xDesc, xData, wDesc, wData, yDesc, yData, convParams, op, deviceType, roundingMode);

    MmeSimTensor  res((int*)outDesc->m_sizes,
                     outDesc->m_dims,
                     getDataType(outDesc->m_dataType, false, deviceType),
                     result);
    pMMESimTensor pResult = std::make_shared<MmeSimTensor>(res);

    MmeSimTensor  refRes((int*)outDesc->m_sizes,
                        outDesc->m_dims,
                        getDataType(outDesc->m_dataType, false, deviceType),
                        refResult);
    pMMESimTensor pRefResult = std::make_shared<MmeSimTensor>(refRes);

    if (usePearsonCompare)
    {
        ret = applyPearsonCompare(result, refResult, outDesc->m_dataType, pResult->getElementsCount());
    }
    else
    {
        unsigned         expBias = QuantizationData::getDefaultExpBias(outDesc->m_dataType);
        TensorComparator comparator(getCDSize(convertOpType(op, false), x, w, y),
                                    getDataType(outDesc->m_dataType, false, deviceType),
                                    expBias);
        ret = comparator.doCompare(pResult, "result", pRefResult, "refResult");

        if (!ret)
        {
            MmeCommon::SizeArray idxArray = comparator.getDiffElement().value();

            for (int i = 0; i < outDesc->m_dims; i++)
            {
                outIndex[i] = idxArray[i];
            }

            char* pElement = pRefResult->getElementAt(idxArray);
            if ((pElement != nullptr) && (expectedResult != nullptr))
            {
                *expectedResult = *((float*)(pElement));
            }
        }
        else
        {
            outIndex.fill(0);
        }
    }
    delete[] refResult;
    return ret;
}

bool checkBatchGemmOp(const synTensorDescriptor& xDesc,
                      char*                      xData,
                      const synTensorDescriptor& wDesc,
                      char*                      wData,
                      const synTensorDescriptor& yDesc,
                      char*                      yData,
                      ERepefenceOp               op,
                      CoordArray&                outIndex,
                      float*                     expectedResult,
                      synDeviceType              deviceType,
                      MmeCommon::RoundingMode    roundingMode)

{
    bool ret = false;

    MmeSimTensor x((int*)xDesc.m_sizes,
                   xDesc.m_dims,
                   getDataType(xDesc.m_dataType, !isDedxOperation(op), deviceType),
                   xData);
    MmeSimTensor w((int*)wDesc.m_sizes,
                   wDesc.m_dims,
                   getDataType(wDesc.m_dataType, op != REFERENCE_OP_DEDW, deviceType),
                   wData);
    MmeSimTensor y((int*)yDesc.m_sizes,
                   yDesc.m_dims,
                   getDataType(yDesc.m_dataType, (isDedxOperation(op) || op == REFERENCE_OP_DEDW), deviceType),
                   yData);

    char*                      result    = nullptr;
    char*                      refResult = nullptr;
    const synTensorDescriptor* outDesc   = nullptr;
    switch (op)
    {
        case REFERENCE_OP_FWD:
        case REFERENCE_OP_AB:
        case REFERENCE_OP_ATB:
        case REFERENCE_OP_ABT:
        case REFERENCE_OP_ATBT:
            result    = yData;
            refResult = allocateByDesc(yDesc);
            yData     = refResult;
            outDesc   = &yDesc;
            break;
        case REFERENCE_OP_DEDX:
        case REFERENCE_OP_TRANSPOSED_DEDX:
            result    = xData;
            refResult = allocateByDesc(xDesc);
            xData     = refResult;
            outDesc   = &xDesc;
            break;
        case REFERENCE_OP_DEDW:
            result    = wData;
            refResult = allocateByDesc(wDesc);
            wData     = refResult;
            outDesc   = &wDesc;
            break;
        default:
            delete[] refResult;
            assert(0 && "unsupported operation type");
    }

    calculateMmeBatchGemmOp(xDesc, xData, wDesc, wData, yDesc, yData, op, deviceType, roundingMode);

    MmeSimTensor  res((int*)outDesc->m_sizes,
                     outDesc->m_dims,
                     getDataType(outDesc->m_dataType, false, deviceType),
                     result);
    pMMESimTensor pResult = std::make_shared<MmeSimTensor>(res);

    MmeSimTensor  refRes((int*)outDesc->m_sizes,
                        outDesc->m_dims,
                        getDataType(outDesc->m_dataType, false, deviceType),
                        refResult);
    pMMESimTensor pRefResult = std::make_shared<MmeSimTensor>(refRes);

    unsigned         expBias = QuantizationData::getDefaultExpBias(outDesc->m_dataType);
    TensorComparator comparator(getCDSize(convertOpType(op, true), x, w, y),
                                getDataType(outDesc->m_dataType, false, deviceType),
                                expBias);
    ret = comparator.doCompare(pResult, "result", pRefResult, "refResult");

    if (!ret)
    {
        MmeCommon::SizeArray idxArray = comparator.getDiffElement().value();

        for (int i = 0; i < outDesc->m_dims; i++)
        {
            outIndex[i] = idxArray[i];
        }
        *expectedResult = *((float*)(pRefResult->getElementAt(idxArray)));
    }
    else
    {
        outIndex.fill(0);
    }

    delete[] refResult;
    return ret;
}

bool checkMaskedBatchGemmOp(const synTensorDescriptor& xDesc,
                            char*                      xData,
                            const synTensorDescriptor& wDesc,
                            char*                      wData,
                            const synTensorDescriptor& xMaskDesc,
                            char*                      xMaskData,
                            const synTensorDescriptor& wMaskDesc,
                            char*                      wMaskData,
                            const synTensorDescriptor& yDesc,
                            char*                      yData,
                            ERepefenceOp               op,
                            CoordArray&                outIndex,
                            float*                     expectedResult,
                            synDeviceType              deviceType,
                            MmeCommon::RoundingMode    roundingMode)
{
    bool          ret = false;
    CPUCalculator calculator(convertDeviceType(deviceType), Mme::c_mme_max_tensor_dims, Mme::c_mme_max_conv_dims);

    //  mask tensor
    MmeSimTensor xMask((int*)xMaskDesc.m_sizes,
                       xMaskDesc.m_dims,
                       getDataType(xMaskDesc.m_dataType, true, deviceType),
                       xMaskData);
    MmeSimTensor wMask((int*)wMaskDesc.m_sizes,
                       wMaskDesc.m_dims,
                       getDataType(wMaskDesc.m_dataType, true, deviceType),
                       wMaskData);
    MmeSimTensor yMaskAcc((int*)yDesc.m_sizes, yDesc.m_dims, getDataType(syn_type_float, false, deviceType));

    calculator.doBatchGemm(yMaskAcc, xMask, wMask, convertOpType(op, true), roundingMode);

    //  bgemm tensors
    MmeSimTensor x((int*)xDesc.m_sizes, xDesc.m_dims, getDataType(xDesc.m_dataType, true, deviceType), xData);
    MmeSimTensor w((int*)wDesc.m_sizes, wDesc.m_dims, getDataType(wDesc.m_dataType, true, deviceType), wData);
    MmeSimTensor yAcc((int*)yDesc.m_sizes, yDesc.m_dims, getDataType(syn_type_float, false, deviceType));

    calculator.doBatchGemm(yAcc, x, w, convertOpType(op, true), roundingMode);

    //  actual test result
    MmeSimTensor res((int*)yDesc.m_sizes, yDesc.m_dims, getDataType(yDesc.m_dataType, false, deviceType), yData);
    //  reference final result
    MmeSimTensor acc((int*)yDesc.m_sizes, yDesc.m_dims, getDataType(syn_type_float, false, deviceType));
    MmeSimTensor refRes((int*)yDesc.m_sizes, yDesc.m_dims, getDataType(yDesc.m_dataType, false, deviceType));

    synTensorDescriptor accDesc = yDesc;
    accDesc.m_dataType          = syn_type_float;
    calculateAdd(accDesc, yAcc.data(), yMaskAcc.data(), acc.data());

    pMMESimTensor pAcc       = std::make_shared<MmeSimTensor>(acc);
    pMMESimTensor pRefResult = std::make_shared<MmeSimTensor>(refRes);
    pMMESimTensor pResult    = std::make_shared<MmeSimTensor>(res);

    calculator.doActivation(pRefResult, pAcc, nullptr, nullptr, false, roundingMode);

    unsigned         expBias = QuantizationData::getDefaultExpBias(yDesc.m_dataType);
    TensorComparator comparator(getCDSize(convertOpType(op, true), x, w, res),
                                getDataType(yDesc.m_dataType, false, deviceType),
                                expBias);
    ret = comparator.doCompare(pResult, "result", pRefResult, "refResult");

    if (!ret)
    {
        MmeCommon::SizeArray idxArray = comparator.getDiffElement().value();

        for (int i = 0; i < yDesc.m_dims; i++)
        {
            outIndex[i] = idxArray[i];
        }
        *expectedResult = *((float*)(pRefResult->getElementAt(idxArray)));
    }
    else
    {
        outIndex.fill(0);
    }

    return ret;
}

bool checkResults(const synTensorDescriptor& desc, char* firstData, char* secondData, CoordArray& outIndex)
{
    outIndex.fill(0);

    return innerCheckResult(desc, firstData, secondData, outIndex);
}

bool checkFwdConvolution(const synTensorDescriptor&    x,
                         char*                         xData,
                         const synTensorDescriptor&    w,
                         char*                         wData,
                         const synTensorDescriptor&    y,
                         char*                         yData,
                         const synConvolution3DParams& convParams,
                         CoordArray&                   outIndex,
                         synDeviceType                 deviceType,
                         MmeCommon::RoundingMode       roundingMode)
{
    return checkMmeOp(x,
                      xData,
                      w,
                      wData,
                      y,
                      yData,
                      convParams,
                      REFERENCE_OP_FWD,
                      outIndex,
                      deviceType,
                      nullptr,
                      false,
                      roundingMode);
}

bool checkDEDX(const synTensorDescriptor&    y,
               char*                         yData,
               const synTensorDescriptor&    w,
               char*                         wData,
               const synTensorDescriptor&    x,
               char*                         xData,
               const synConvolution3DParams& convParams,
               CoordArray&                   outIndex,
               synDeviceType                 deviceType,
               MmeCommon::RoundingMode       roundingMode)
{
    return checkMmeOp(x,
                      xData,
                      w,
                      wData,
                      y,
                      yData,
                      convParams,
                      REFERENCE_OP_DEDX,
                      outIndex,
                      deviceType,
                      nullptr,
                      false,
                      roundingMode);
}

bool checkDEDW(const synTensorDescriptor&    y,
               char*                         yData,
               const synTensorDescriptor&    x,
               char*                         xData,
               const synTensorDescriptor&    w,
               char*                         wData,
               const synConvolution3DParams& convParams,
               CoordArray&                   outIndex,
               synDeviceType                 deviceType,
               bool                          usePearsonCompare,
               MmeCommon::RoundingMode       roundingMode)
{
    return checkMmeOp(x,
                      xData,
                      w,
                      wData,
                      y,
                      yData,
                      convParams,
                      REFERENCE_OP_DEDW,
                      outIndex,
                      deviceType,
                      nullptr,
                      usePearsonCompare,
                      roundingMode);
}

void calculateFwdConvolution(const synTensorDescriptor&    x,
                             char*                         xData,
                             const synTensorDescriptor&    w,
                             char*                         wData,
                             const synTensorDescriptor&    y,
                             char*                         result,
                             const synConvolution3DParams& convParams,
                             synDeviceType                 deviceType,
                             MmeCommon::RoundingMode       roundingMode)
{
    calculateMmeOp(x, xData, w, wData, y, result, convParams, REFERENCE_OP_FWD, deviceType, roundingMode);
}

void calculateDEDX(const synTensorDescriptor&    y,
                   char*                         yData,
                   const synTensorDescriptor&    w,
                   char*                         wData,
                   const synTensorDescriptor&    x,
                   char*                         result,
                   const synConvolution3DParams& convParams,
                   synDeviceType                 deviceType,
                   MmeCommon::RoundingMode       roundingMode)
{
    calculateMmeOp(x, result, w, wData, y, yData, convParams, REFERENCE_OP_DEDX, deviceType, roundingMode);
}

void calculateDEDW(const synTensorDescriptor&    y,
                   char*                         yData,
                   const synTensorDescriptor&    x,
                   char*                         xData,
                   const synTensorDescriptor&    w,
                   char*                         result,
                   const synConvolution3DParams& convParams,
                   synDeviceType                 deviceType,
                   MmeCommon::RoundingMode       roundingMode)
{
    calculateMmeOp(x, xData, w, result, y, yData, convParams, REFERENCE_OP_DEDW, deviceType, roundingMode);
}

void calculateRelu(const synTensorDescriptor& inDesc, void* inData, const synTensorDescriptor& outDesc, void* result)
{
    unsigned inputSize = 1;
    assert(outDesc.m_sizes == inDesc.m_sizes && outDesc.m_dims == inDesc.m_dims &&
           "input and output should have the same shape !");
    inputSize = multiplyElements(&inDesc.m_sizes[0], &inDesc.m_sizes[inDesc.m_dims]);
    memcpy(result, inData, inputSize * getElementSizeInBytes(inDesc.m_dataType));
    switch (inDesc.m_dataType)
    {
        case syn_type_single:
        case syn_type_fixed:
            calculateRelu((float*)result, inputSize);
            break;
        case syn_type_int32:
            calculateRelu((int32_t*)result, inputSize);
            break;
        case syn_type_bf16:
            calculateRelu((bfloat16*)result, inputSize);
            break;
        case syn_type_int16:
            calculateRelu((int16_t*)result, inputSize);
            break;
        case syn_type_uint8:
            calculateRelu((int8_t*)result, inputSize);
            break;
        default:
            assert(0 && "unknown conversion type");
    }
}

void calculateGemm(const synTensorDescriptor& xDesc,
                   char*                      xData,
                   const synTensorDescriptor& wDesc,
                   char*                      wData,
                   const synTensorDescriptor& yDesc,
                   char*                      yData,
                   const synGEMMParams&       params,
                   ERepefenceOp               op,
                   synDeviceType              deviceType,
                   MmeCommon::RoundingMode    roundingMode)
{
    calculateMmeBatchGemmOp(xDesc, xData, wDesc, wData, yDesc, yData, op, deviceType, roundingMode);
}

template<class T>
static void addition(char* inA, char* inB, char* out, uint64_t numElements)
{
    T* inAPtr = (T*)inA;
    T* inBPtr = (T*)inB;
    T* outPtr = (T*)out;

    for (uint64_t i = 0; i < numElements; ++i)
    {
        outPtr[i] = float(inAPtr[i]) + float(inBPtr[i]);
    }
}

void calculateAdd(const synTensorDescriptor& desc, char* inA, char* inB, char* out)
{
    uint64_t numElements = multiplyElements(desc.m_sizes, desc.m_sizes + desc.m_dims);
    if (desc.m_dataType == syn_type_bf16)
    {
        addition<bfloat16>(inA, inB, out, numElements);
    }
    else if (desc.m_dataType == syn_type_float)
    {
        addition<float>(inA, inB, out, numElements);
    }
    else
    {
        assert(0 && "Not supported type");
    }
}
