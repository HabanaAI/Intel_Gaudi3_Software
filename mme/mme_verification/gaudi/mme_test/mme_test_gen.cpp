#include "mme_test_gen.h"
#include "mme_test.h"
#include "sim_tensor.h"
#include "tensor_utils.h"
#include <stdlib.h>

#include "mme_reference.h"
#include "sim_tensor_base.h"
#include "tensor_comparator.h"
#include "mme_assert.h"
#define FMT_HEADER_ONLY
#include <spdlog/fmt/bundled/format.h>

#define max(a, b)      (((a) > (b)) ? (a) : (b))
#define min(a, b)      (((a) < (b)) ? (a) : (b))
#define div_ceil(n, d) (((n) + (d) -1) / (d))

using namespace MmeCommon;

static void runBgemm(const MmeTestParams_t* params,
                     const convTestData_t* data,
                     const MmeStrategy& strategy,
                     const MmeTestParams& MmeUserTestParams,
                     int* soValue,
                     std::list<MmeRegWriteCmd>* cmds,
                     const Mme::Desc* prevDesc,
                     Mme::Desc* lastDesc,
                     uint32_t scale,
                     float memsetTensorVal,
                     uint32_t ref_lfsr_seed,
                     uint32_t ref_lfsr_poly,
                     EMmeOpType refOp,
                     bool verifMode,
                     unsigned testCounter)
{
    genRandomValues(data->hostIn, params->xMinVal, params->xMaxVal, params->fp, scale);
    genRandomValues(data->hostWeights, params->wMinVal, params->wMaxVal, params->fp, scale);
    RoundingMode euRm = params->rm;
    RoundingMode apRm = params->stochsaticConversion ? RoundingMode::StochasticRounding : params->rm;

    if (!params->skipReference)
    {
        atomicColoredPrint(COLOR_CYAN, "INFO: Running reference. (test #%u)\n", testCounter);
        data->hostOut->memsetTensor((char*) &memsetTensorVal);
        int outputShape[Mme::c_mme_max_tensor_dims];
        data->hostOut->copySizes(outputShape);
        MmeSimTensor grf(outputShape, params->ioDim, e_type_fp32);
        MmeSimTensor mmeOut(outputShape, params->ioDim, params->outputType);
        grf.memsetTensor((char*) &memsetTensorVal);
        data->hostRef->memsetTensor((char*) &memsetTensorVal);

        if (!params->incDec)
        {
            CPUCalculator commonCalc(e_mme_Gaudi, Mme::c_mme_max_tensor_dims, Mme::c_mme_max_conv_dims);
            commonCalc.doBatchGemm(grf, *data->hostIn, *data->hostWeights, refOp, euRm);
            pMMESimTensor pHostRef = std::make_shared<MmeSimTensor>(*data->hostRef);
            pMMESimTensor pMmeOut = std::make_shared<MmeSimTensor>(mmeOut);
            pMMESimTensor pGrf = std::make_shared<MmeSimTensor>(grf);
            commonCalc.doActivation(pMmeOut, pGrf, nullptr, nullptr, params->conv.relu, apRm);
            commonCalc.doMemoryWrite(pHostRef, pMmeOut, params->reductionOp, params->reductionRm);
        }
        atomicColoredPrint(COLOR_CYAN, "INFO: Running reference completed. (test #%u)\n", testCounter);
    }
    EMmeOpType opType = EMmeOpType::e_mme_fwd;
    switch (params->op)
    {
        case EConvTestOp::E_CONV_TEST_AB:
            opType = EMmeOpType::e_mme_ab;
            break;
        case EConvTestOp::E_CONV_TEST_ABT:
            opType = EMmeOpType::e_mme_abt;
            break;
        case EConvTestOp::E_CONV_TEST_ATB:
            opType = EMmeOpType::e_mme_atb;
            break;
        case EConvTestOp::E_CONV_TEST_ATBT:
            opType = EMmeOpType::e_mme_atbt;
            break;
        default:
            MME_ASSERT(0, "invalid operation");
    }

    atomicColoredPrint(COLOR_CYAN, "INFO: Generating code. (test #%u)\n", testCounter);
    executeBGemm(opType,
                 verifMode,
                 params->tid,
                 &params->conv,
                 params->soAddrLow,
                 params->soAddrHigh,
                 &data->in,
                 &data->weights,
                 &data->out,
                 euRm,
                 apRm,
                 &strategy,
                 &MmeUserTestParams,
                 soValue,
                 cmds,
                 prevDesc,
                 lastDesc);
}

void genConvTest(const MmeTestParams_t* params,
                 const convTestData_t* data,
                 int* soValue,
                 std::list<MmeRegWriteCmd>* cmds,
                 const Mme::Desc* prevDesc,
                 Mme::Desc* lastDesc,
                 bool verifMode,
                 unsigned testCounter)
{
    MmeTestParams MmeUserTestParams;
    MmeUserTestParams.randomMD = params->randomMD;
    MmeUserTestParams.wkldIdMD = params->dumpEn;
    MmeUserTestParams.repeats = params->repeats;
    MmeUserTestParams.sbResuseInStripes = params->sbReuseInStripes;
    MmeUserTestParams.incDec = params->incDec;
    MmeUserTestParams.maskSignals = params->loop;

    MmeStrategy strategy;
    strategy.geometry = params->geometry;
    strategy.loweringEn = params->lower;
    strategy.partial = false;
    strategy.signalPartial = params->signalPartial;
    strategy.sbReuse = params->sbReuse;
    strategy.unrollEn = params->unrollEn;
    strategy.dedwAsBgemmEn = params->dedwAsBgemmEn;
    strategy.recurringMisalignmentOptEn = params->recurringMisalignmentOptEn;
    strategy.pattern = params->pattern;
    strategy.memsetDedxVoidPixels = params->memsetVoidPixels;

    bool outputInSram = false;
    switch (params->op)
    {
        case E_CONV_TEST_FWD:
        case E_CONV_TEST_AB:
        case E_CONV_TEST_ABT:
        case E_CONV_TEST_ATB:
        case E_CONV_TEST_ATBT:
            outputInSram = params->yInSram;
            break;
        case E_CONV_TEST_DEDX:
            outputInSram = params->xInSram;
            break;
        case E_CONV_TEST_DEDW:
            outputInSram = params->wInSram;
            break;
        default:
            MME_ASSERT(0, "Unsupported operation");
    }
    strategy.dedwAsBgemmEn = params->dedwAsBgemmEn;
    if (params->dedwAsBgemmEn)
    {
        MME_ASSERT(
            params->memsetOutput && outputInSram,
            "When dedwAsBgemm optimization is enabled, memsetOutput must be set and the output tensor must be in sram");
    }
    srand(params->seed);
    uint32_t ref_lfsr_seed = (rand() << 16) | (rand() & 0xffff);
    uint32_t ref_lfsr_poly = (rand() << 16) | (rand() & 0xffff);

    MmeSimTensor::f32_t memsetTensorVal = params->incDec ? 0x00000000 : 0x77777777;
    unsigned scale = params->scaledRandomValues ? 10 : -1;
    RoundingMode euRm = params->rm;
    RoundingMode apRm = params->stochsaticConversion ? RoundingMode::StochasticRounding : params->rm;

    switch (params->op)
    {
        case E_CONV_TEST_FWD:
        {
            genRandomValues(data->hostIn, params->xMinVal, params->xMaxVal, params->fp, scale);
            genRandomValues(data->hostWeights, params->wMinVal, params->wMaxVal, params->fp, scale);
            if (!params->skipReference)
            {
                atomicColoredPrint(COLOR_CYAN, "INFO: Running reference. (test #%u)\n", testCounter);
                data->hostOut->memsetTensor((char*) &memsetTensorVal);
                int outputShape[Mme::c_mme_max_tensor_dims];
                data->hostOut->copySizes(outputShape);
                MmeSimTensor grf(outputShape, params->ioDim, e_type_fp32);
                grf.memsetTensor((char*) &memsetTensorVal);
                MmeSimTensor mmeOut(outputShape, params->ioDim, params->outputType);
                data->hostRef->memsetTensor((char*) &memsetTensorVal);
                if (!params->incDec)
                {
                    CPUCalculator commonCalc(e_mme_Gaudi, Mme::c_mme_max_tensor_dims, Mme::c_mme_max_conv_dims);

                    commonCalc.doConvolution(grf,
                                             *data->hostIn,
                                             *data->hostWeights,
                                             *data->hostOut,
                                             params->conv,
                                             EMmeOpType::e_mme_fwd,
                                             euRm);
                    pMMESimTensor pHostRef = std::make_shared<MmeSimTensor>(*data->hostRef);
                    pMMESimTensor pMmeOut = std::make_shared<MmeSimTensor>(mmeOut);
                    pMMESimTensor pGrf = std::make_shared<MmeSimTensor>(grf);
                    commonCalc.doActivation(pMmeOut, pGrf, nullptr, nullptr, params->conv.relu, apRm);
                    commonCalc.doMemoryWrite(pHostRef, pMmeOut, params->reductionOp, params->reductionRm);
                }
                atomicColoredPrint(COLOR_CYAN, "INFO: Reference run completed. (test #%u)\n", testCounter);
            }

            atomicColoredPrint(COLOR_CYAN, "INFO: Generating code. (test #%u)\n", testCounter);
            executeConvFWD(verifMode,
                           params->tid,
                           &params->conv,
                           params->soAddrLow,
                           params->soAddrHigh,
                           &data->in,
                           &data->weights,
                           &data->out,
                           euRm,
                           apRm,
                           &strategy,
                           &MmeUserTestParams,
                           soValue,
                           cmds,
                           prevDesc,
                           lastDesc);
            break;
        }
        case E_CONV_TEST_DEDX:
        {
            genRandomValues(data->hostWeights, params->wMinVal, params->wMaxVal, params->fp, scale);
            genRandomValues(data->hostOut, params->yMinVal, params->yMaxVal, params->fp, scale);
            if (!params->skipReference)
            {
                atomicColoredPrint(COLOR_CYAN, "INFO: Running reference. (test #%u)\n", testCounter);
                data->hostIn->memsetTensor((char*) &memsetTensorVal);
                int inputShape[Mme::c_mme_max_tensor_dims];
                data->hostIn->copySizes(inputShape);
                MmeSimTensor grf(inputShape, params->ioDim, e_type_fp32);
                grf.memsetTensor((char*) &memsetTensorVal);
                MmeSimTensor mmeOut(inputShape, params->ioDim, params->outputType);
                data->hostRef->memsetTensor((char*) &memsetTensorVal);
                if (!params->incDec)
                {
                    CPUCalculator commonCalc(e_mme_Gaudi, Mme::c_mme_max_tensor_dims, Mme::c_mme_max_conv_dims);

                    commonCalc.doConvolution(grf,
                                             *data->hostIn,
                                             *data->hostWeights,
                                             *data->hostOut,
                                             params->conv,
                                             EMmeOpType::e_mme_dedx,
                                             euRm);

                    pMMESimTensor pHostRef = std::make_shared<MmeSimTensor>(*data->hostRef);
                    pMMESimTensor pMmeOut = std::make_shared<MmeSimTensor>(mmeOut);
                    pMMESimTensor pGrf = std::make_shared<MmeSimTensor>(grf);
                    commonCalc.doActivation(pMmeOut, pGrf, nullptr, nullptr, params->conv.relu, apRm);
                    commonCalc.doMemoryWrite(pHostRef, pMmeOut, params->reductionOp, params->reductionRm);
                }
                atomicColoredPrint(COLOR_CYAN, "INFO: Reference run completed. (test #%u)\n", testCounter);
            }
            atomicColoredPrint(COLOR_CYAN, "INFO: Generating code. (test #%u)\n", testCounter);
            executeConvDEDX(verifMode,
                            params->tid,
                            &params->conv,
                            params->soAddrLow,
                            params->soAddrHigh,
                            &data->in,
                            &data->weights,
                            &data->out,
                            euRm,
                            apRm,
                            &strategy,
                            &MmeUserTestParams,
                            soValue,
                            cmds,
                            prevDesc,
                            lastDesc);

            break;
        }
        case E_CONV_TEST_DEDW:
        {
            genRandomValues(data->hostIn, params->xMinVal, params->xMaxVal, params->fp, scale);
            genRandomValues(data->hostOut, params->yMinVal, params->yMaxVal, params->fp, scale);

            if (!params->skipReference)
            {
                atomicColoredPrint(COLOR_CYAN, "INFO: Running reference. (test #%u)\n", testCounter);
                data->hostWeights->memsetTensor((char*) &memsetTensorVal);
                int weightsShape[Mme::c_mme_max_tensor_dims];
                data->hostWeights->copySizes(weightsShape);
                MmeSimTensor grf(weightsShape, params->conv.dim + 2, e_type_fp32);
                grf.memsetTensor((char*) &memsetTensorVal);
                MmeSimTensor mmeOut(weightsShape, params->conv.dim + 2, params->outputType);
                data->hostRef->memsetTensor((char*) &memsetTensorVal);
                if (!params->incDec)
                {
                    CPUCalculator commonCalc(e_mme_Gaudi, Mme::c_mme_max_tensor_dims, Mme::c_mme_max_conv_dims);

                    commonCalc.doConvolution(grf,
                                             *data->hostIn,
                                             *data->hostWeights,
                                             *data->hostOut,
                                             params->conv,
                                             EMmeOpType::e_mme_dedw,
                                             euRm);
                    pMMESimTensor pHostRef = std::make_shared<MmeSimTensor>(*data->hostRef);
                    pMMESimTensor pMmeOut = std::make_shared<MmeSimTensor>(mmeOut);
                    pMMESimTensor pGrf = std::make_shared<MmeSimTensor>(grf);
                    commonCalc.doActivation(pMmeOut, pGrf, nullptr, nullptr, params->conv.relu, apRm);
                    commonCalc.doMemoryWrite(pHostRef, pMmeOut, params->reductionOp, params->reductionRm);
                }
                atomicColoredPrint(COLOR_CYAN, "INFO: Reference run completed. (test #%u)\n", testCounter);
            }
            atomicColoredPrint(COLOR_CYAN, "INFO: Generating code. (test #%u)\n", testCounter);
            executeConvDEDW(verifMode,
                            params->tid,
                            &params->conv,
                            params->soAddrLow,
                            params->soAddrHigh,
                            &data->in,
                            &data->weights,
                            &data->out,
                            euRm,
                            apRm,
                            &strategy,
                            &MmeUserTestParams,
                            soValue,
                            cmds,
                            prevDesc,
                            lastDesc);

            break;

            case E_CONV_TEST_AB:
                runBgemm(params,
                         data,
                         strategy,
                         MmeUserTestParams,
                         soValue,
                         cmds,
                         prevDesc,
                         lastDesc,
                         scale,
                         memsetTensorVal,
                         ref_lfsr_seed,
                         ref_lfsr_poly,
                         EMmeOpType::e_mme_ab,
                         verifMode,
                         testCounter);
                break;
            case E_CONV_TEST_ABT:
                runBgemm(params,
                         data,
                         strategy,
                         MmeUserTestParams,
                         soValue,
                         cmds,
                         prevDesc,
                         lastDesc,
                         scale,
                         memsetTensorVal,
                         ref_lfsr_seed,
                         ref_lfsr_poly,
                         EMmeOpType::e_mme_abt,
                         verifMode,
                         testCounter);
                break;
            case E_CONV_TEST_ATB:
                runBgemm(params,
                         data,
                         strategy,
                         MmeUserTestParams,
                         soValue,
                         cmds,
                         prevDesc,
                         lastDesc,
                         scale,
                         memsetTensorVal,
                         ref_lfsr_seed,
                         ref_lfsr_poly,
                         EMmeOpType::e_mme_atb,
                         verifMode,
                         testCounter);
                break;
            case E_CONV_TEST_ATBT:
                runBgemm(params,
                         data,
                         strategy,
                         MmeUserTestParams,
                         soValue,
                         cmds,
                         prevDesc,
                         lastDesc,
                         scale,
                         memsetTensorVal,
                         ref_lfsr_seed,
                         ref_lfsr_poly,
                         EMmeOpType::e_mme_atbt,
                         verifMode,
                         testCounter);
                break;

            default:
                MME_ASSERT(0, "invalid operation");
        }
    }
}

static unsigned
getCDSize(EMmeOpType op, const MmeSimTensor* xTensor, const MmeSimTensor* wTensor, const MmeSimTensor* yTensor)
{
    const SizeArray& wSizes = wTensor->getSizes();
    unsigned filterSizes = multiplyElements(std::next(std::next(wSizes.begin())), wSizes.end());
    switch (op)
    {
        case EMmeOpType::e_mme_ab:
        case EMmeOpType::e_mme_abt:
            return xTensor->getSize(0);
        case EMmeOpType::e_mme_atb:
        case EMmeOpType::e_mme_atbt:
            return xTensor->getSize(1);
        case EMmeOpType::e_mme_fwd:
            return xTensor->getSize(0) * filterSizes;
        case EMmeOpType::e_mme_dedx:
            return yTensor->getSize(0) * filterSizes;
        case EMmeOpType::e_mme_dedw:
        {
            const SizeArray& ySizes = yTensor->getSizes();
            return multiplyElements(std::next(ySizes.begin()), ySizes.end());
        }
        default:
            MME_ASSERT(0, "invalid op");
    }
    return 0;
}

static EMmeOpType gaudiOpToCommonOp(EConvTestOp op)
{
    switch (op)
    {
        case E_CONV_TEST_FWD:
            return EMmeOpType::e_mme_fwd;
        case E_CONV_TEST_DEDX:
            return EMmeOpType::e_mme_dedx;
        case E_CONV_TEST_DEDW:
            return EMmeOpType::e_mme_dedw;
        case E_CONV_TEST_AB:
            return EMmeOpType::e_mme_ab;
        case E_CONV_TEST_ABT:
            return EMmeOpType::e_mme_abt;
        case E_CONV_TEST_ATB:
            return EMmeOpType::e_mme_atb;
        case E_CONV_TEST_ATBT:
            return EMmeOpType::e_mme_atbt;
            break;
        default:
            MME_ASSERT(0, "invalid operation");
            return EMmeOpType::e_mme_fwd;
    }
}

static MmeSimTensor*
getOutputTensor(EConvTestOp op, MmeSimTensor* xTensor, MmeSimTensor* wTensor, MmeSimTensor* yTensor)
{
    switch (op)
    {
        case E_CONV_TEST_FWD:
        case E_CONV_TEST_AB:
        case E_CONV_TEST_ABT:
        case E_CONV_TEST_ATB:
        case E_CONV_TEST_ATBT:
            return yTensor;
        case E_CONV_TEST_DEDX:
            return xTensor;
        case E_CONV_TEST_DEDW:
            return wTensor;
        default:
            MME_ASSERT(0, "invalid operation");
            return yTensor;
    }
}

bool compareResults(testInfo_t& ti, bool firstDeviceIsChip, char* devBAddr, unsigned testCounter)
{
    bool equal = true;
    EMmeOpType refOp = gaudiOpToCommonOp(ti.params.op);
    std::string testInfoStr = test2text(&ti.params);
    MmeSimTensor* mmeOut = getOutputTensor(ti.params.op, ti.data.hostIn, ti.data.hostWeights, ti.data.hostOut);
    pMMESimTensor pMMEOut = std::make_shared<MmeSimTensor>(*mmeOut);

    std::string firstDevName = firstDeviceIsChip ? "Device[0]" : "Simulator";
    std::string secondDevName = firstDeviceIsChip ? "" : "Device[0]";

    TensorComparator comparator(getCDSize(refOp, ti.data.hostIn, ti.data.hostWeights, ti.data.hostOut),
                                mmeOut->getElementType());

    if (ti.params.skipReference)
    {
        atomicColoredPrint(COLOR_YELLOW, "INFO: Skipping reference. (test #%u)\n", testCounter);
    }
    else
    {
        atomicColoredPrint(COLOR_CYAN,
                           "INFO: Comparing the results of %s to the "
                           "reference. (test #%u)\n",
                           firstDevName.c_str(),
                           testCounter);
        pMMESimTensor pHostRef = std::make_shared<MmeSimTensor>(*ti.data.hostRef);
        equal &= comparator.doCompare(pMMEOut,
                                      fmt::format("first output of {}", firstDevName),
                                      pHostRef,
                                      "Reference",
                                      testCounter,
                                      testInfoStr);
    }

    if (devBAddr != nullptr)  // comparing two devices
    {
        atomicColoredPrint(COLOR_CYAN,
                           "INFO: Comparing the output of %s and %s. (test #%u)\n",
                           firstDevName.c_str(),
                           secondDevName.c_str(),
                           testCounter);

        MmeSimTensor devBOutput(mmeOut, devBAddr);
        pMMESimTensor pDevBOutput = std::make_shared<MmeSimTensor>(devBOutput);

        equal &= comparator.doCompareBitExact(pMMEOut,
                                              fmt::format("first output of {}", firstDevName),
                                              pDevBOutput,
                                              fmt::format("first output of {}", secondDevName),
                                              testCounter,
                                              testInfoStr);
    }

    return equal;
}
