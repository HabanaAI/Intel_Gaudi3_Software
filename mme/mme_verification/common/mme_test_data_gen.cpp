#include "mme_test_data_gen.h"
#include "general_utils.h"
#include "mme_assert.h"
#include "include/mme_common/mme_common_enum.h"
#include "print_utils.h"
#include "mme_common_utils.h"

using namespace MmeCommon;

unsigned DataGenerator::getFpBias(const EMmeDataType type, const unsigned fpBias) const
{
    return getFpBias(type, fpBias, m_dataParams.m_chipType);
}

unsigned DataGenerator::getFpBias(const EMmeDataType type, const unsigned fpBias, const ChipType chipType)
{
    switch (type)
    {
        case EMmeDataType::e_type_ufp16:
            return EXPONENT_BIAS_UFP16_31;
        case EMmeDataType::e_type_fp16:
            if (chipType == e_mme_Gaudi2)
            {
                //  Gaudi2 only supports a single a single bias for fp16.
                //  untill now this json field wasnt used in the case of fp16.
                //  once all jsons are fixed this can be removed
                return EXPONENT_BIAS_FP16_15;
            }
            //  for Gaudi3 fallthrough
        case EMmeDataType::e_type_fp8_143:
        case EMmeDataType::e_type_fp8_152:
        {
            return fpBias;
        }
        default:
            return 0;
    }
}

bool DataGenerator::createAndInitDataParams(const nlohmann::json& testJson,
                                            bool runOnSim,
                                            unsigned driverDevicesNr)
{
    if (!createParams(testJson, testJson["secondaryOutput"].get<bool>(), runOnSim, driverDevicesNr))
    {
        return false;
    }
    if (testJson["skipRun"].get<bool>())
    {
        return true;
    }
    bool hasReduction = testJson["reductionOp"].get<EMmeReductionOp>() != MmeCommon::e_mme_reduction_none;
    generateData(testJson, testJson["secondaryOutput"].get<bool>(), hasReduction);
    return true;
}

SizeArray DataGenerator::getSizeArrayFromVector(const std::vector<unsigned int>& configArray) const
{
    // in case config is smaller then sizeArray - fill with 1's
    SizeArray sizes = {1, 1, 1, 1, 1};
    unsigned dim = std::min(configArray.size(), sizes.size());
    std::copy_n(configArray.begin(), dim, sizes.begin());
    return sizes;
}

// This function will be used when non-flattened reductionAdd is supported
static void checkReductionAddSizes(const SizeArray& xSizes,
                                   const SizeArray& wSizes,
                                   const SizeArray& ySizes,
                                   unsigned packingFactor,
                                   unsigned reductionLevel,
                                   bool& packingFactorIsOk,
                                   bool& reductionLevelIsOk)
{
    if (packingFactor != xSizes[1])
    {
        packingFactorIsOk = false;
        return;
    }
    if (reductionLevel != xSizes[0] / packingFactor)
    {
        reductionLevelIsOk = false;
        return;
    }
    // Verify that reductionPacking is valid. It must be one of: {1, n4, n4*n3, n4*n3*n2, ...}, where n values taken from w
    packingFactorIsOk = false;
    unsigned legalPackingValue = 1;
    for (int i=MME_MAX_TENSOR_DIMS-1; i>=0; i--)
    {
        if (packingFactor == legalPackingValue)
        {
            packingFactorIsOk = true;
            break;
        }
        legalPackingValue *= ySizes[i];
    }

    // Check that all sizes of y and w are identical, except that wSizes[4] = ySizes[4] * reductionLevel
    reductionLevelIsOk = true;
    for (int i=MME_MAX_TENSOR_DIMS-1; i>0; i--)
    {
        if (i == MME_MAX_TENSOR_DIMS-1)
        {
            if (wSizes[4] != ySizes[4] * reductionLevel)
            {
                reductionLevelIsOk = false;
                return;
            }
        }
        else
        {
            if (ySizes[i] != wSizes[i])
            {
                reductionLevelIsOk = false;
                return;
            }
        }
    }
}

// verify that the dimensions can produce a proper convolution,
bool DataGenerator::verifyTensorSizes(const SizeArray& xSizes,
                                      const SizeArray& wSizes,
                                      const SizeArray& ySizes,
                                      const EMmeOpType& op,
                                      unsigned packingFactor,
                                      unsigned reductionLevel) const
{
    // check common dim
    bool commonDimOK = false, outputWidthOK = false, packingFactorOk = true, reductionLevelOk = true, batchSizesAreOK = true;
    switch (op)
    {
        case e_mme_memcpy:
            commonDimOK = (ySizes[1] == xSizes[1]);
            outputWidthOK = (ySizes[0] == xSizes[0]);
            break;
        case e_mme_gemm_transpose:
        case e_mme_trans:
            commonDimOK = true;
            outputWidthOK = true;
            break;
        case e_mme_fwd:
        case e_mme_ab:
            commonDimOK = (xSizes[0] == wSizes[1]);
            outputWidthOK = (ySizes[0] == wSizes[0]);
            break;
        case e_mme_transposed_dedx:
            commonDimOK = (ySizes[0] == wSizes[1]);
            outputWidthOK = (xSizes[0] == wSizes[0]);
            break;
        case e_mme_dedx:
            commonDimOK = (ySizes[0] == wSizes[0]);
            outputWidthOK = (xSizes[0] == wSizes[1]);
            break;
        case e_mme_dedw:
            // TODO
            commonDimOK = true;
            outputWidthOK = (wSizes[0] == ySizes[0]);
            break;
        case e_mme_abt:
            commonDimOK = (xSizes[0] == wSizes[0]);
            outputWidthOK = (wSizes[1] == ySizes[0]);
            break;
        case e_mme_atb:
            commonDimOK = (xSizes[1] == wSizes[1]);
            outputWidthOK = (wSizes[0] == ySizes[0]);
            break;
        case e_mme_atbt:
            commonDimOK = (xSizes[1] == wSizes[0]);
            outputWidthOK = (wSizes[1] == ySizes[0]);
            break;
        case e_mme_reductionAdd:
            checkReductionAddSizes(xSizes,
                                   wSizes,
                                   ySizes,
                                   packingFactor,
                                   reductionLevel,
                                   packingFactorOk,
                                   reductionLevelOk);
            commonDimOK = outputWidthOK = true; // In reductionAdd op, reduction parameters are checked
            break;
        default:
            MME_ASSERT(0, "should not get here");
    }
    if (!commonDimOK)
    {
        atomicColoredPrint(COLOR_RED, "Tensor dimension doesnt match on common dim\n")
    }
    if (!outputWidthOK)
    {
        atomicColoredPrint(COLOR_RED, "Output width doesnt match operand B width\n")
    }
    if (!packingFactorOk)
    {
        atomicColoredPrint(COLOR_RED, "Reduction packing factor is inconsistent\n")
    }
    if (!reductionLevelOk)
    {
        atomicColoredPrint(COLOR_RED, "Reduction level is inconsistent\n")
    }
    if (!batchSizesAreOK)
    {
        atomicColoredPrint(COLOR_RED, "Batch sizes are inconsistent\n")
    }
    return commonDimOK && outputWidthOK && packingFactorOk && reductionLevelOk;
}

bool DataGenerator::createParams(const nlohmann::json& testJson,
                                 bool enableSecondOutput,
                                 bool runOnSim,
                                 unsigned driverDevicesNr)
{
    // get tensor attribues
    MmeTensorAttributes xAttr, wAttr, yAttr, oAttr;
    xAttr.sizes.push_back(getSizeArrayFromVector(testJson["xSizes"].get<std::vector<unsigned>>()));
    wAttr.sizes.push_back(getSizeArrayFromVector(testJson["wSizes"].get<std::vector<unsigned>>()));
    yAttr.sizes.push_back(getSizeArrayFromVector(testJson["ySizes"].get<std::vector<unsigned>>()));
    xAttr.sizes.push_back(getSizeArrayFromVector(testJson["xSizes2"].get<std::vector<unsigned>>()));
    wAttr.sizes.push_back(getSizeArrayFromVector(testJson["wSizes2"].get<std::vector<unsigned>>()));
    yAttr.sizes.push_back(getSizeArrayFromVector(testJson["ySizes2"].get<std::vector<unsigned>>()));

    xAttr.strides.push_back(getSizeArrayFromVector(testJson["xStrides"].get<std::vector<unsigned>>()));
    wAttr.strides.push_back(getSizeArrayFromVector(testJson["wStrides"].get<std::vector<unsigned>>()));
    yAttr.strides.push_back(getSizeArrayFromVector(testJson["yStrides"].get<std::vector<unsigned>>()));
    xAttr.strides.push_back(getSizeArrayFromVector(testJson["xStrides2"].get<std::vector<unsigned>>()));
    wAttr.strides.push_back(getSizeArrayFromVector(testJson["wStrides2"].get<std::vector<unsigned>>()));
    yAttr.strides.push_back(getSizeArrayFromVector(testJson["yStrides2"].get<std::vector<unsigned>>()));

    xAttr.dims = testJson["xSizes"].get<std::vector<unsigned>>().size();
    wAttr.dims = testJson["wSizes"].get<std::vector<unsigned>>().size();
    yAttr.dims = testJson["ySizes"].get<std::vector<unsigned>>().size();
    getTensorAttrByOp(testJson, xAttr, wAttr, yAttr, oAttr);

    //Prepare strides.
    const MmeCommon::SizeArray* xStridesPtr[2];
    const MmeCommon::SizeArray* yStridesPtr[2];
    const MmeCommon::SizeArray* wStridesPtr[2];
    const MmeCommon::SizeArray* oStridesPtr[2];
    for (unsigned gemm = 0; gemm < 2; ++gemm)
    {
        xStridesPtr[gemm] = (xAttr.strides[gemm])[0] ? &(xAttr.strides[gemm]) : nullptr;
        yStridesPtr[gemm] = (yAttr.strides[gemm])[0] ? &(yAttr.strides[gemm]) : nullptr;
        wStridesPtr[gemm] = (wAttr.strides[gemm])[0] ? &(wAttr.strides[gemm]) : nullptr;
        oStridesPtr[gemm] = (oAttr.strides[gemm])[0] ? &(oAttr.strides[gemm]) : nullptr;
    }

    m_dataParams.operandInSram[e_mme_op_x] = testJson["xInSram"].get<bool>();
    m_dataParams.operandInSram[e_mme_op_w] = testJson["wInSram"].get<bool>();
    m_dataParams.operandInSram[e_mme_op_y] = testJson["yInSram"].get<bool>();
    m_dataParams.operandInSram[e_mme_op_o] = testJson["oInSram"].get<bool>();

    const EMmeOpType op = testJson["operation"].get<EMmeOpType>();
    unsigned gemmNr = testJson["dualGemm"].get<bool>() ? 2 : 1;
    unsigned packingFactor = testJson["packingFactor"].get<unsigned>();
    unsigned reductionLevel = testJson["reductionLevel"].get<unsigned>();
    m_dataParams.tensorParams.resize(gemmNr);
    for (unsigned gemm = 0; gemm < gemmNr; gemm++)
    {
        if (!verifyTensorSizes(xAttr.sizes[gemm], wAttr.sizes[gemm], yAttr.sizes[gemm], op, packingFactor, reductionLevel))
        {
            return false;
        }

        // make sim tensors
        m_dataParams.tensorParams[gemm].xHost = std::make_shared<MmeSimTensor>(xAttr.sizes[gemm],
                                                                               xAttr.dims,
                                                                               xAttr.dataType,
                                                                               nullptr,
                                                                               xAttr.fpBias,
                                                                               xAttr.infNanMode,
                                                                               xStridesPtr[gemm]);
        m_dataParams.tensorParams[gemm].xHost->setName("xTensor");

        m_dataParams.tensorParams[gemm].wHost = std::make_shared<MmeSimTensor>(wAttr.sizes[gemm],
                                                                               wAttr.dims,
                                                                               wAttr.dataType,
                                                                               nullptr,
                                                                               wAttr.fpBias,
                                                                               wAttr.infNanMode,
                                                                               wStridesPtr[gemm]);
        m_dataParams.tensorParams[gemm].wHost->setName("wTensor");

        m_dataParams.tensorParams[gemm].yHost = std::make_shared<MmeSimTensor>(yAttr.sizes[gemm],
                                                                               yAttr.dims,
                                                                               yAttr.dataType,
                                                                               nullptr,
                                                                               yAttr.fpBias,
                                                                               yAttr.infNanMode,
                                                                               yStridesPtr[gemm]);
        m_dataParams.tensorParams[gemm].yHost->setName("yTensor");

        m_dataParams.tensorParams[gemm].outRef = std::make_shared<MmeSimTensor>(oAttr.sizes[gemm],
                                                                                oAttr.dims,
                                                                                oAttr.dataType,
                                                                                nullptr,
                                                                                oAttr.fpBias,
                                                                                oAttr.infNanMode,
                                                                                oStridesPtr[gemm]);
        m_dataParams.tensorParams[gemm].outRef->setName("outRef");

        if (runOnSim)
        {
            m_dataParams.tensorParams[gemm].outDevSim0 = std::make_shared<MmeSimTensor>(oAttr.sizes[gemm],
                                                                                        oAttr.dims,
                                                                                        oAttr.dataType,
                                                                                        nullptr,
                                                                                        oAttr.fpBias,
                                                                                        oAttr.infNanMode,
                                                                                        oStridesPtr[gemm]);
            m_dataParams.tensorParams[gemm].outDevSim0->setName("outDevSim0" + gemm);
        }
        for (unsigned idx = 0; idx < driverDevicesNr; idx++)
        {
            auto tensor = std::make_shared<MmeSimTensor>(oAttr.sizes[gemm],
                                                         oAttr.dims,
                                                         oAttr.dataType,
                                                         nullptr,
                                                         oAttr.fpBias,
                                                         oAttr.infNanMode,
                                                         oStridesPtr[gemm]);
            tensor->setName("outDevChip0[" + std::to_string(idx) + "]");
            m_dataParams.tensorParams[gemm].outDevChip0.push_back(tensor);
            MME_ASSERT(idx == m_dataParams.tensorParams[gemm].outDevChip0.size() - 1, "device index is not registered");
        }

        if (enableSecondOutput)
        {
            m_dataParams.tensorParams[gemm].oHost = std::make_shared<MmeSimTensor>(oAttr.sizes[gemm],
                                                                                   oAttr.dims,
                                                                                   oAttr.dataType,
                                                                                   nullptr,
                                                                                   oAttr.fpBias,
                                                                                   oAttr.infNanMode,
                                                                                   oStridesPtr[gemm]);
            m_dataParams.tensorParams[gemm].oHost->setName("oTensor");

            if (runOnSim)
            {
                m_dataParams.tensorParams[gemm].outDevSim1 = std::make_shared<MmeSimTensor>(oAttr.sizes[gemm],
                                                                                            oAttr.dims,
                                                                                            oAttr.dataType,
                                                                                            nullptr,
                                                                                            oAttr.fpBias,
                                                                                            oAttr.infNanMode,
                                                                                            oStridesPtr[gemm]);
                m_dataParams.tensorParams[gemm].outDevSim1->setName("outDevSim1");
            }
            for (unsigned idx = 0; idx < driverDevicesNr; idx++)
            {
                auto tensor = std::make_shared<MmeSimTensor>(oAttr.sizes[gemm],
                                                             oAttr.dims,
                                                             oAttr.dataType,
                                                             nullptr,
                                                             oAttr.fpBias,
                                                             oAttr.infNanMode,
                                                             oStridesPtr[gemm]);
                tensor->setName("outDevChip1[" + std::to_string(idx) + "]");
                m_dataParams.tensorParams[gemm].outDevChip1.push_back(tensor);
                MME_ASSERT(idx == m_dataParams.tensorParams[gemm].outDevChip1.size() - 1,
                           "device index is not registered");
            }
        }
    }

    // Create the aux tensors
    if (testJson["isDeterministic"].get<bool>() &&
        testJson["operation"].get<EMmeOpType>() == MmeCommon::e_mme_dedw &&
        testJson["dedwCDConcurrency"].get<BoolWithUndef>() == MmeCommon::TurnedOn)
    {
        // Sanity checks
        MME_ASSERT(gemmNr == 1, "When deterministic CDC is set, only single gemm is allowed");

        auto& auxTensors = m_dataParams.tensorParams.front().auxTensors;
        unsigned cdConcurrencyLevel = testJson["reductionLevel"].get<int>();
        unsigned cdConcurrencyPackingFactor = testJson["packingFactor"].get<int>();
        auxTensors[CD_SCRATCHPAD].isSram = testJson["yInSram"].get<bool>();
        auxTensors[CD_REDUCTION].isSram = false;
        auxTensors[CD_SCRATCHPAD].isInput = false;  // The tensor needs to be defined as output for the checking
        auxTensors[CD_REDUCTION].isInput = true;    // The tensor is initialized and is input for the reductionAdd

        // Set the CD Scratchpad (primary) aux tensor to the same as dw output, except that its upper dim is Nx (N is cd con level)
        auto cdScratchpadSizes = wAttr.sizes[0];
        cdScratchpadSizes[MME_MAX_TENSOR_DIMS-1] *= cdConcurrencyLevel;
        auxTensors[CD_SCRATCHPAD].pTensor = std::make_shared<MmeSimTensor>(cdScratchpadSizes,
                                                                           wAttr.dims,
                                                                           wAttr.dataType,
                                                                           nullptr,
                                                                           wAttr.fpBias,
                                                                           wAttr.infNanMode,
                                                                           nullptr);
        auxTensors[CD_SCRATCHPAD].pTensor->setName("CdScratchpadAux");

        // Set the CD Reduction (secondary) aux tensor sizes to {N * Pack, Pack, 1, 1, 1}, where N is the cd concurrency level and
        // Pack is the cd concurrency packing factor
        SizeArray secndarySizes = {cdConcurrencyLevel * cdConcurrencyPackingFactor, cdConcurrencyPackingFactor, 1, 1, 1};
        auxTensors[CD_REDUCTION].pTensor = std::make_shared<MmeSimTensor>(secndarySizes,
                                                                          wAttr.dims,
                                                                          wAttr.dataType,
                                                                          nullptr,
                                                                          0,
                                                                          MmeCommon::e_mme_full_inf_nan,
                                                                          nullptr);
        auxTensors[CD_REDUCTION].pTensor->setName("CdReductionAux");
    }
    if (testJson["maskedBgemm"].get<bool>())
    {
        auto& auxTensors = m_dataParams.tensorParams.front().auxTensors;
        // Mask Bgemm A
        auxTensors[MASKED_BGEMM_A].pTensor = std::make_shared<MmeSimTensor>(xAttr.sizes[1],
                                                                    xAttr.dims,
                                                                    xAttr.dataType,
                                                                    nullptr,
                                                                    xAttr.fpBias,
                                                                    xAttr.infNanMode,
                                                                    xStridesPtr[1]);
        auxTensors[MASKED_BGEMM_A].pTensor->setName("xAuxTensor");
        auxTensors[MASKED_BGEMM_A].isSram = false;
        auxTensors[MASKED_BGEMM_A].isInput = true;
        // Mask Bgemm B
        auxTensors[MASKED_BGEMM_B].pTensor = std::make_shared<MmeSimTensor>(wAttr.sizes[1],
                                                                    wAttr.dims,
                                                                    wAttr.dataType,
                                                                    nullptr,
                                                                    wAttr.fpBias,
                                                                    wAttr.infNanMode,
                                                                    wStridesPtr[1]);
        auxTensors[MASKED_BGEMM_B].pTensor->setName("wAuxTensor");
        auxTensors[MASKED_BGEMM_B].isSram = false;
        auxTensors[MASKED_BGEMM_B].isInput = true;
    }

    return true;
}

void DataGenerator::getTensorAttrByOp(const nlohmann::json& testJson,
                                      MmeTensorAttributes& xAttr,
                                      MmeTensorAttributes& wAttr,
                                      MmeTensorAttributes& yAttr,
                                      MmeTensorAttributes& oAttr) const
{
    const EMmeDataType inDType = testJson["inTypeFloat"].get<EMmeDataType>();
    const EMmeDataType in2DType = testJson["in2TypeFloat"].get<EMmeDataType>();
    const EMmeDataType outDType = testJson["outTypeFloat"].get<EMmeDataType>();
    const InfNanMode infNanModeA = testJson["infNanModeA"].get<InfNanMode>();
    const InfNanMode infNanModeB = testJson["infNanModeB"].get<InfNanMode>();
    const InfNanMode outInfNanMode = testJson["infNanModeOut"].get<InfNanMode>();
    const unsigned biasIn = testJson["fp8BiasIn"].get<unsigned>();
    const unsigned biasIn2 = testJson["fp8BiasIn2"].get<unsigned>();
    const unsigned biasOut = testJson["fp8BiasOut"].get<unsigned>();
    switch (testJson["operation"].get<EMmeOpType>())
    {
        case e_mme_memcpy:
        case e_mme_trans:
        case e_mme_gemm_transpose:
        case e_mme_fwd:
        case e_mme_ab:
        case e_mme_atb:
        case e_mme_abt:
        case e_mme_atbt:
        case e_mme_reductionAdd:
            xAttr.infNanMode = infNanModeA;
            wAttr.infNanMode = infNanModeB;
            yAttr.infNanMode = outInfNanMode;
            xAttr.dataType = inDType;
            wAttr.dataType = in2DType;
            yAttr.dataType = outDType;
            xAttr.fpBias = getFpBias(inDType, biasIn);
            wAttr.fpBias = getFpBias(in2DType, biasIn2);
            yAttr.fpBias = getFpBias(outDType, biasOut);
            oAttr = yAttr;
            break;
        case e_mme_dedx:
        case e_mme_transposed_dedx:
            yAttr.infNanMode = infNanModeA;
            wAttr.infNanMode = infNanModeB;
            xAttr.infNanMode = outInfNanMode;
            yAttr.dataType = inDType;
            wAttr.dataType = in2DType;
            xAttr.dataType = outDType;
            yAttr.fpBias = getFpBias(inDType, biasIn);
            wAttr.fpBias = getFpBias(in2DType, biasIn2);
            xAttr.fpBias = getFpBias(outDType, biasOut);
            oAttr = xAttr;
            break;
        case e_mme_dedw:
            xAttr.infNanMode = infNanModeA;
            yAttr.infNanMode = infNanModeB;
            wAttr.infNanMode = outInfNanMode;
            xAttr.dataType = inDType;
            yAttr.dataType = in2DType;
            wAttr.dataType = outDType;
            xAttr.fpBias = getFpBias(inDType, biasIn);
            yAttr.fpBias = getFpBias(in2DType, biasIn2);
            wAttr.fpBias = getFpBias(outDType, biasOut);
            oAttr = wAttr;
            break;
        default:
            MME_ASSERT(0, "should not get here");
    }
}

void readFile(const char* fileName, pMMESimTensor& tensor)
{
    FILE* fin = fopen(fileName, "rb");
    if (!fin)
    {
        MME_ASSERT(0, "fileName not correct");
    }
    fseek(fin, 0, SEEK_END);
    long fsize = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    char* buffer = new char[fsize];
    int bytes_read = fread(buffer, sizeof(char), fsize, fin);
    tensor->setData(bytes_read, buffer);
    fclose(fin);
    delete[] buffer;
}

void DataGenerator::getRefTensorMapper(EMmeOpType op, unsigned gemm, pMMESimTensor mapperTensors[]) const
{
    switch (op)
    {
        case e_mme_memcpy:
        case e_mme_trans:
        case e_mme_gemm_transpose:
        case e_mme_fwd:
        case e_mme_ab:
        case e_mme_atb:
        case e_mme_abt:
        case e_mme_atbt:
        case e_mme_reductionAdd:
            mapperTensors[0] = m_dataParams.tensorParams[gemm].xHost;
            mapperTensors[1] = m_dataParams.tensorParams[gemm].wHost;
            mapperTensors[2] = m_dataParams.tensorParams[gemm].yHost;
            break;
        case e_mme_dedx:
        case e_mme_transposed_dedx:
            mapperTensors[0] = m_dataParams.tensorParams[gemm].yHost;
            mapperTensors[1] = m_dataParams.tensorParams[gemm].wHost;
            mapperTensors[2] = m_dataParams.tensorParams[gemm].xHost;
            break;
        case e_mme_dedw:
            mapperTensors[0] = m_dataParams.tensorParams[gemm].yHost;
            mapperTensors[1] = m_dataParams.tensorParams[gemm].xHost;
            mapperTensors[2] = m_dataParams.tensorParams[gemm].wHost;
            break;
        default:
            MME_ASSERT(0, "invalid operation");
            break;
    }
}

void DataGenerator::generateData(const nlohmann::json& testJson, bool enableSecondOutput, bool hasReduction)
{
    const bool extRefTest = testJson["extRefTest"].get<bool>();
    const auto op = testJson["operation"].get<EMmeOpType>();
    float outputInitialVal = shouldMemsetOutput(testJson) ? 0 : std::numeric_limits<float>::max();
    RandomSimTensorGenerator randomGen(m_dataParams.seed);
    unsigned gemmNr = testJson["dualGemm"].get<bool>() ? 2 : 1;

    if (extRefTest)
    {
        MME_ASSERT(gemmNr == 1, "external ref testing not supported in dual gemm mode yet");
        for (unsigned gemm = 0; gemm < gemmNr; gemm++)
        {
            pMMESimTensor mapperTensors[3];
            const char* inputA = "inputA.bin";
            const char* inputB = "inputB.bin";
            getRefTensorMapper(op, gemm, mapperTensors);
            readFile(inputA, mapperTensors[0]);
            readFile(inputB, mapperTensors[1]);
            fillTensorData(randomGen, mapperTensors[2], op, false, false, outputInitialVal, outputInitialVal);
        }
    }
    else
    {
        auto xMinValue = testJson["xMinVal"].get<float>();
        auto wMinValue = testJson["wMinVal"].get<float>();
        auto yMinValue = testJson["yMinVal"].get<float>();

        auto xMaxValue = testJson["xMaxVal"].get<float>();
        auto wMaxValue = testJson["wMaxVal"].get<float>();
        auto yMaxValue = testJson["yMaxVal"].get<float>();

        bool doNormalDistributionX = testJson.count("xStd") && (testJson["xStd"].get<float>() != 0);
        bool doNormalDistributionW = testJson.count("wStd") && (testJson["wStd"].get<float>() != 0);
        bool doNormalDistributionY = testJson.count("yStd") && (testJson["yStd"].get<float>() != 0);

        auto xStdDev = doNormalDistributionX ? testJson["xStd"].get<float>() : 0;
        auto xMean = doNormalDistributionX ? testJson["xMean"].get<float>() : 0;
        auto wStdDev = doNormalDistributionX ? testJson["wStd"].get<float>() : 0;
        auto wMean = doNormalDistributionX ? testJson["wMean"].get<float>() : 0;
        auto yStdDev = doNormalDistributionX ? testJson["yStd"].get<float>() : 0;
        auto yMean = doNormalDistributionX ? testJson["yMean"].get<float>() : 0;

        unsigned packingFactor = testJson["packingFactor"].get<unsigned>();
        unsigned reductionLevel = testJson["reductionLevel"].get<unsigned>();

        for (unsigned gemm = 0; gemm < gemmNr; gemm++)
        {
            // Identify the actual output tensor
            std::shared_ptr<MmeSimTensor> actualOutputTensor = getActualOutputTensor(op, gemm);
            // generate X
            fillTensorData(randomGen,
                           m_dataParams.tensorParams[gemm].xHost,
                           op,
                           doNormalDistributionX,
                           false,
                           xMinValue,
                           xMaxValue,
                           op == MmeCommon::e_mme_reductionAdd,
                           packingFactor,
                           reductionLevel,
                           xMean,
                           xStdDev);

            // generate W
            if (op != MmeCommon::e_mme_memcpy && op != MmeCommon::e_mme_trans)
            {
                fillTensorData(randomGen,
                               m_dataParams.tensorParams[gemm].wHost,
                               op,
                               doNormalDistributionW,
                               op == MmeCommon::e_mme_gemm_transpose,
                               wMinValue,
                               wMaxValue,
                               false,
                               1,
                               1,
                               wMean,
                               wStdDev);
            }
            // generate Y
            fillTensorData(randomGen,
                           m_dataParams.tensorParams[gemm].yHost,
                           op,
                           doNormalDistributionY,
                           false,
                           yMinValue,
                           yMaxValue,
                           false,
                           1,
                           1,
                           yMean,
                           yStdDev);

            // When reduction is valid, output must be initialized in both reference and
            // both devices
            if (hasReduction)
            {
                // Duplicate the actual output tensor to outRef
                randomGen.duplicate(actualOutputTensor, m_dataParams.tensorParams[gemm].outRef);
                if (m_dataParams.tensorParams[gemm].outDevSim0)
                {
                    randomGen.duplicate(actualOutputTensor, m_dataParams.tensorParams[gemm].outDevSim0);
                }
                for (auto& chipTensor : m_dataParams.tensorParams[gemm].outDevChip0)
                {
                    randomGen.duplicate(actualOutputTensor, chipTensor);
                }

                if (enableSecondOutput)
                {
                    randomGen.duplicate(actualOutputTensor, m_dataParams.tensorParams[gemm].oHost);
                    if (m_dataParams.tensorParams[gemm].outDevSim1)
                    {
                        randomGen.duplicate(actualOutputTensor, m_dataParams.tensorParams[gemm].outDevSim1);
                    }
                    for (auto& chipTensor : m_dataParams.tensorParams[gemm].outDevChip1)
                    {
                        randomGen.duplicate(actualOutputTensor, chipTensor);
                    }
                }
            }
            else
            {
                // fill outputs
                fillTensorData(randomGen, actualOutputTensor, op, false, false, outputInitialVal, outputInitialVal);
                fillTensorData(randomGen,
                               m_dataParams.tensorParams[gemm].outRef,
                               op,
                               false,
                               false,
                               outputInitialVal,
                               outputInitialVal);
                if (m_dataParams.tensorParams[gemm].outDevSim0)
                {
                    fillTensorData(randomGen,
                                   m_dataParams.tensorParams[gemm].outDevSim0,
                                   op,
                                   false,
                                   false,
                                   outputInitialVal,
                                   outputInitialVal);
                }
                for (auto& chipTensor : m_dataParams.tensorParams[gemm].outDevChip0)
                {
                    fillTensorData(randomGen, chipTensor, op, false, false, outputInitialVal, outputInitialVal);
                }
                if (enableSecondOutput)
                {
                    fillTensorData(randomGen,
                                   m_dataParams.tensorParams[gemm].oHost,
                                   op,
                                   false,
                                   false,
                                   outputInitialVal,
                                   outputInitialVal);
                    if (m_dataParams.tensorParams[gemm].outDevSim1)
                    {
                        fillTensorData(randomGen,
                                       m_dataParams.tensorParams[gemm].outDevSim1,
                                       op,
                                       false,
                                       false,
                                       outputInitialVal,
                                       outputInitialVal);
                    }
                    for (auto& chipTensor : m_dataParams.tensorParams[gemm].outDevChip1)
                    {
                        fillTensorData(randomGen, chipTensor, op, false, false, outputInitialVal, outputInitialVal);
                    }
                }

                bool isDeterministicCdConcurrency = testJson["isDeterministic"].get<bool>() &&
                                                    testJson["operation"].get<EMmeOpType>() == MmeCommon::e_mme_dedw &&
                                                    testJson["dedwCDConcurrency"].get<BoolWithUndef>() == MmeCommon::TurnedOn &&
                                                    testJson["reductionLevel"].get<unsigned>() > 1;
                if (isDeterministicCdConcurrency)
                {
                    // Sanity checks
                    MME_ASSERT(m_dataParams.tensorParams[0].auxTensors[CD_SCRATCHPAD].pTensor &&
                                m_dataParams.tensorParams[0].auxTensors[CD_REDUCTION].pTensor, "In deterministic CDC the CD aux tensors are required");
                    MME_ASSERT(gemmNr == 1, "In deterministic CDC only single gemm is expected");

                    // cd scratchpad aux tensor is an output of the first sub-operation that fills in its partial results.
                    fillTensorData(randomGen,
                                   m_dataParams.tensorParams[0].auxTensors[CD_SCRATCHPAD].pTensor,
                                   op,
                                   false,
                                   false,
                                   outputInitialVal,
                                   outputInitialVal);
                    // Init the secondary aux tensor for the reductionAdd packing
                    fillTensorData(randomGen,
                                   m_dataParams.tensorParams[0].auxTensors[CD_REDUCTION].pTensor,
                                   op,
                                   false,
                                   false,
                                   0,  // not used
                                   0,  // not used
                                   true,  // fill for packing tensor for reductionAdd
                                   packingFactor,
                                   reductionLevel,
                                   xMean,  // not used
                                   xStdDev);  // not used
                }
                if (testJson["maskedBgemm"].get<bool>())
                {
                    auto auxTensors = m_dataParams.tensorParams[0].auxTensors;
                    MME_ASSERT(auxTensors[MASKED_BGEMM_A].pTensor && auxTensors[MASKED_BGEMM_B].pTensor,
                               "In maskedBgemm the relevant aux tensors are required");

                    fillTensorData(randomGen,
                                   auxTensors[MASKED_BGEMM_A].pTensor,
                                   op,
                                   doNormalDistributionX,
                                   false,
                                   xMinValue,
                                   xMaxValue);
                    fillTensorData(randomGen,
                                   auxTensors[MASKED_BGEMM_B].pTensor,
                                   op,
                                   doNormalDistributionW,
                                   false,
                                   wMinValue,
                                   wMaxValue);
                }
            }
        }
    }
}

void DataGenerator::fillTensorData(RandomSimTensorGenerator& randomGen,
                                   pMMESimTensor& tensor,
                                   EMmeOpType op,
                                   bool normalDist,
                                   bool unitMatrix,
                                   float minValue,
                                   float maxValue,
                                   bool fillForReductionPacking,
                                   unsigned packingFactor,
                                   unsigned reductionLevel,
                                   float mean,
                                   float stdDev)
{
    if (fillForReductionPacking)
    {
        randomGen.fillForReductionPacking(tensor, packingFactor, reductionLevel);
    }
    else if (unitMatrix)
    {
        randomGen.generateUnitMatrix(tensor, 1, 0);
    }
    else if (normalDist)
    {
        randomGen.generateNormal(tensor, minValue, maxValue, mean, stdDev);
    }
    else if (minValue == maxValue)
    {
        randomGen.fill(tensor, maxValue);
    }
    else
    {
        randomGen.generateUniform(tensor, minValue, maxValue);
    }
}

bool DataGenerator::shouldMemsetOutput(const nlohmann::json& testJson)
{
    //  zeroMem the output if required or in case of power test
    bool isPowerTest = testJson["powerTest"].get<bool>();
    bool memsetOutput = testJson["memsetOutput"].get<bool>() || isPowerTest;

    // in case memsetVoidPixels is turned off - need to make sure the output data is initialized to 0.
    const EMmeOpType op = testJson["operation"].get<EMmeOpType>();
    bool memsetVoidPixels = testJson["memsetVoidPixels"].get<bool>();
    if ((op == e_mme_dedx || op == e_mme_transposed_dedx) && !memsetVoidPixels)
    {
        atomicColoredPrint(COLOR_YELLOW, "WARNING: memsetVoidPixels is not set in dedx - forcing memsetOutput.\n");
        memsetOutput = true;
    }

    //  in case of reduction we cant memZero the output otherwise operations like add/max might not actually be tested
    const bool isReduction = (testJson["reductionOp"].get<EMmeReductionOp>() != EMmeReductionOp::e_mme_reduction_none);
    //  specifically in dedwConcurrency we need to memZero the output because all MMEs will accumulate in parallel.
    const bool isDedwConcurrency = (testJson["dedwCDConcurrency"].get<BoolWithUndef>() == MmeCommon::TurnedOn);
    //  in power test several reductions are made that should eventually accumulate to 0 so we do want to zeroMem
    if (isReduction && !isDedwConcurrency && !isPowerTest)
    {
        MME_ASSERT((op != e_mme_dedx && op != e_mme_transposed_dedx) || memsetVoidPixels,
                   "dedx/transposed_dedx test with reduction must set memsetVoidPixels");
        MME_ASSERT(!memsetOutput, "reduction tests require non-zeroMem output - but the test requires it");
    }

    return memsetOutput;
}

pMMESimTensor DataGenerator::getActualOutputTensor(EMmeOpType op, unsigned gemm) const
{
    switch (op)
    {
        case e_mme_memcpy:
        case e_mme_trans:
        case e_mme_gemm_transpose:
        case e_mme_fwd:
        case e_mme_ab:
        case e_mme_atb:
        case e_mme_abt:
        case e_mme_atbt:
        case e_mme_reductionAdd:
            return m_dataParams.tensorParams[gemm].yHost;
        case e_mme_dedx:
        case e_mme_transposed_dedx:
            return m_dataParams.tensorParams[gemm].xHost;
        case e_mme_dedw:
            return m_dataParams.tensorParams[gemm].wHost;
        default:
            MME_ASSERT(0, "invalid operation");
            break;
    }
    return nullptr;
}
