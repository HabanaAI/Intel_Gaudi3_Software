#include "src/mme_common/common_geo_attr.h"
#include "general_utils.h"
#include "include/mme_common/mme_common_enum.h"
#include "src/mme_common/mme_hal_factory.h"
#include "mme_params_factory.h"
#include "src/mme_common/mme_geo_factory.h"
#include "include/mme_common/mme_brain.h"
#include "mme_verification/common/mme_reg_write_cmd.h"
#include "sim_tensor.h"
#include "mme_user.h"
#include "common/mme_test_global_conf.h"
#include "print_utils.h"
#include "utils.h"
#include "include/mme_common/workarounds.h"
#include "src/utils/logger.h"
// coral includes
#include "coral_user_program_base.h"
#include "coral_user_utils.h"

namespace MmeCommon
{
MmeUser::MmeUser(const ChipType chipType, unsigned mmeNr)
: m_chipType(chipType), m_mmeHal(getMmeHal(chipType)), m_mmeNr(mmeNr)
{
}

void MmeUser::initConv(const ConvolutionParams& in, MmeConv* out)
{
    for (unsigned i = 0; i < ConvolutionParams::maxConvDim; i++)
    {
        out->stride[i] = in.convStride[i];
        out->dilation[i] = in.dilation[i];
        out->padding[i] = in.padding[i];
    }
}

void setUndefToTurnedOff(BoolWithUndef& f)
{
    if (f == Undefined)
    {
        f = TurnedOff;
    }
}

void flattenTensorsForReductionAdd(MmeLayerParams& params)
{
    unsigned packingFactor = params.strategy.packingFactor;
    unsigned reductionLevel = params.strategy.reductionLevel;

    unsigned numOutputElements = multiplyElements(params.y.sizes.begin(), params.y.sizes.end());
    unsigned numWElements = numOutputElements * reductionLevel;
    params.w.sizes = {numOutputElements / packingFactor, packingFactor * reductionLevel, 1, 1, 1};
    params.y.sizes = {numOutputElements / packingFactor, packingFactor, 1, 1, 1};
    params.w.strides = {1, numOutputElements / packingFactor, numWElements, numWElements, numWElements};
    params.y.strides = {1, numOutputElements / packingFactor, numOutputElements, numOutputElements, numOutputElements};
}

void MmeUser::createSubNodeParams(const MmeLayerParams& params,
                                  const std::vector<MmeTensorParams>& tensorParams,
                                  SubParamsVec& subParamsVec,
                                  std::vector<OperandRoles>& operandRolesVec)
{
    // Deterministic cd concurrency is implemented by creating two params: one for the original node that writes
    // the partial results to the primary aux tensor, and then reductionAdd node that sums up the slices
    if (params.isDeterministicCdConcurrency())
    {
        auto& auxTensors = tensorParams[0].auxTensors;
        // Get the actual concurrency level enabled by the geometry & the given reductionLevel
        const upCommonGeoAttr& geoAttr = MmeCommon::getGeoAttr(m_chipType, params);
        unsigned actualReductionLevel = geoAttr->getGeometryCdConcurrency();
        // Sanity checks
        MME_ASSERT(auxTensors[CD_SCRATCHPAD].pTensor && auxTensors[CD_REDUCTION].pTensor,
            "Deterministic cd concurrency requires that CD aux tensors are set");

        // The first subParam is identical. Its output is auxTensor[0] with sizes that match the original output
        // Params for 1st operation
        auto subParams = params;
        subParams.opType = MmeCommon::e_mme_deterministic_dedw;
        subParamsVec.push_back(subParams);

        // The second operation performs reduction-add between the partial results in the primary aux
        subParams = params;
        subParams.opType = MmeCommon::e_mme_reductionAdd;
        initMmeTensorView(*auxTensors[CD_REDUCTION].pTensor, &subParams.x);
        initMmeTensorView(*auxTensors[CD_SCRATCHPAD].pTensor, &subParams.w);
        initMmeTensorView(*tensorParams[0].wHost, &subParams.y);
        // Flatten w and y tensors
        unsigned packingFactor = params.strategy.packingFactor;
        unsigned numOutputElements =
            std::accumulate(subParams.y.sizes.begin(), subParams.y.sizes.end(), 1, std::multiplies<>());
        unsigned numPrimaryAuxElements = numOutputElements * actualReductionLevel;
        subParams.w.sizes = {numOutputElements / packingFactor, packingFactor * actualReductionLevel, 1, 1, 1};
        subParams.y.sizes = {numOutputElements / packingFactor, packingFactor, 1, 1, 1};
        subParams.w.strides = {1,
                               numOutputElements / packingFactor,
                               numPrimaryAuxElements,
                               numPrimaryAuxElements,
                               numPrimaryAuxElements};
        subParams.y.strides = {1,
                               numOutputElements / packingFactor,
                               numOutputElements,
                               numOutputElements,
                               numOutputElements};
        subParams.memoryCfg.reductionOp = e_mme_reduction_none;
        subParams.strategy.cdConcurrencyEn = TurnedOff;
        subParamsVec.push_back(subParams);

        // Map the operands of the 2 sub-params to the proper aux tensors:
        // - Op 1 (dedw): output tensor of the dedw node to auxTensor 0
        // - Op 2 (reductionAdd): x is mapped to the secondary aux, w is mapped to the primary aux
        operandRolesVec.push_back({INPUT_TENSOR_A, INPUT_TENSOR_B, AUX_ROLE_CD_SCRATCHPAD});
        operandRolesVec.push_back({AUX_ROLE_CD_REDUCTION, AUX_ROLE_CD_SCRATCHPAD, OUTPUT_TENSOR_C});
    }
    else if (params.strategy.maskedBgemm)
    {
        subParamsVec = {params};
        // Map all operands to aux tensors (relevnt only to even activations)
        operandRolesVec.push_back({AUX_ROLE_MASKED_BGEMM_A, AUX_ROLE_MASKED_BGEMM_B, OUTPUT_TENSOR_C});
    }
    else
    {
        subParamsVec = {params};
        operandRolesVec.push_back({INPUT_TENSOR_A, INPUT_TENSOR_B, OUTPUT_TENSOR_C});
    }
}

void MmeUser::createLayerParams(const EMmeOpType opType,
                                const unsigned wkldId,
                                const ConvolutionParams& conv,
                                const std::vector<MmeTensorParams>& tensorParams,
                                const MmeStrategy& strategy,
                                const MmeControls& controls,
                                const MmeMemoryConfig& memoryConfig,
                                const MmeTestParams& testParams,
                                SubParamsVec& subParamsVec,
                                OperandRolesVec& operandRolesVec)
{
    MmeLayerParams params = getMmeLayerParams(m_chipType);
    params.opType = opType;

    initConv(conv, &params.conv);
    initMmeTensorView(*tensorParams.front().xHost, &params.x);
    initMmeTensorView(*tensorParams.front().yHost, &params.y);
    initMmeTensorView(*tensorParams.front().wHost, &params.w);
    if (strategy.maskedBgemm)
    {
        //  add mask tensors
        initMmeTensorView(*tensorParams.front().auxTensors[MASKED_BGEMM_A].pTensor, &params.xAux);
        initMmeTensorView(*tensorParams.front().auxTensors[MASKED_BGEMM_B].pTensor, &params.wAux);
        initMmeTensorView(*tensorParams.front().yHost, &params.yAux);
    }
    params.spBase = 0;
    params.spSize = 1;
    for (unsigned dim = 1; dim < MME_MAX_TENSOR_DIMS; dim++)
    {
        params.spSize *= (params.isDedxOperation()) ? params.x.sizes[dim] : params.y.sizes[dim];
    }

    params.strategy = strategy;
    params.controls = controls;
    params.tracing.ctxId = wkldId;
    params.tracing.traceMode = testParams.traceMode;
    params.memoryCfg = memoryConfig;
    if (params.controls.signalingMode == e_mme_signaling_amount)
    {
        // not the most accurate solution but the idea here is to select a value that will be
        // lower than signaling for every output but not low enough so it would default to signaling once per desc
        unsigned signalsPerPipelineLevel = div_round_up(params.spSize, 128 * 4);
        params.controls.signalAmount = params.strategy.pipelineLevel * signalsPerPipelineLevel;
    }
    if (opType == e_mme_reductionAdd)
    {
        flattenTensorsForReductionAdd(params);
    }

    // When useBrain is set: respect the concurrency fields regardless of values.
    // When useBrain is not used, then if set respect it, otherwise set to turnedOff.
    if (testParams.useBrain)
    {
        // When useBrain is set, geometry, pattern and concurrency fields are set by mme brain instead of the json
        MmeCommon::MmeBrain brain(m_chipType);

        // Choose concurrency, geo and pattern (if not set)
        brain.getRecommendedStrategy(params);
    }
    else
    {
        setUndefToTurnedOff(params.strategy.batchConcurrencyEn);
        setUndefToTurnedOff(params.strategy.cdConcurrencyEn);
    }

    createSubNodeParams(params, tensorParams, subParamsVec, operandRolesVec);
}

void MmeUser::initMmeTensorView(const MmeSimTensor& tensor, MmeTensorView* view)
{
    view->elementType = tensor.getElementType();
    tensor.copySizes((int*) &view->sizes[0]);
    tensor.copyStrides((int*) &view->strides[0]);
    memset(&view->bases[0], 0, view->strides.size() * sizeof(view->strides[0]));
    for (unsigned dim = tensor.getDim(); dim < MME_MAX_TENSOR_DIMS; dim++)
    {
        view->strides[dim] = view->strides[dim - 1] * view->sizes[dim - 1];
        view->sizes[dim] = 1;
    }
}

void MmeUser::setInputs(EMmeOpType op, bool& isInputX, bool& isInputW, bool& isInputY)
{
   EMmeInternalOperand internalOpX = mmeOpToInternalOp(e_mme_op_x, op);
   EMmeInternalOperand internalOpY = mmeOpToInternalOp(e_mme_op_y, op);
   EMmeInternalOperand internalOpW = mmeOpToInternalOp(e_mme_op_w, op);
   isInputX = (internalOpX == e_mme_op_a || internalOpX == e_mme_op_b);
   isInputY = (internalOpY == e_mme_op_a || internalOpY == e_mme_op_b);
   isInputW = (internalOpW == e_mme_op_a || internalOpW == e_mme_op_b);
}

void MmeUser::makeOperandsInfo(const EMmeOpType op,
                               const MmeDataParams& dataParams,
                               const MmeTensorParams& tensorParams,
                               OperandsInfo* info)
{
    MME_ASSERT_PTR(info);
    switch (op)
    {
        case e_mme_dedx:
        case e_mme_transposed_dedx:
            info->a = tensorParams.yHost.get();
            info->aInSram = dataParams.operandInSram[e_mme_op_y];

            info->b = tensorParams.wHost.get();
            info->bInSram = dataParams.operandInSram[e_mme_op_w];

            info->c = tensorParams.xHost.get();
            info->cInSram = dataParams.operandInSram[e_mme_op_x];
            break;
        case e_mme_dedw:
        case e_mme_deterministic_dedw:
            info->a = tensorParams.xHost.get();
            info->aInSram = dataParams.operandInSram[e_mme_op_x];

            info->b = tensorParams.yHost.get();
            info->bInSram = dataParams.operandInSram[e_mme_op_y];

            info->c = tensorParams.wHost.get();
            info->cInSram = dataParams.operandInSram[e_mme_op_w];
            break;
        case e_mme_memcpy:
        case e_mme_trans:
        case e_mme_gemm_transpose:
        case e_mme_fwd:
        case e_mme_ab:
        case e_mme_atb:
        case e_mme_abt:
        case e_mme_atbt:
        case e_mme_reductionAdd:
            info->a = tensorParams.xHost.get();
            info->aInSram = dataParams.operandInSram[e_mme_op_x];

            info->b = tensorParams.wHost.get();
            info->bInSram = dataParams.operandInSram[e_mme_op_w];

            info->c = tensorParams.yHost.get();
            info->cInSram = dataParams.operandInSram[e_mme_op_y];
            break;
        default:
            MME_ASSERT(0, "invalid operation");
    }
}

void MmeUser::getMemAttribPtrs(const bool isSram,
                               const MmeTensorMemoryAttrib& attrs,
                               MmeMemoryUsage& usage,
                               uint64_t** offsetPtr,
                               const uint64_t** sizePtr,
                               const uint64_t** basePtr)
{
    if (isSram)
    {
        *offsetPtr = &usage.sramUsage;
        *sizePtr = &attrs.sramSize;
        *basePtr = &attrs.sramBase;
    }
    else
    {
        *offsetPtr = &usage.hbmUsage;
        *sizePtr = &attrs.hbmSize;
        *basePtr = &attrs.hbmBase;
    }
}

void MmeUser::allocTensorDeviceMem(const bool isSram,
                                   const bool isInput,
                                   const MmeSimTensor& t,
                                   const MmeTensorMemoryAttrib& memAttrib,
                                   MmeMemoryUsage& memUsage,
                                   uint64_t& addr,
                                   bool isAligned)
{
    const unsigned alignment = m_mmeHal.getMemoryClSize();
    bool allocateInSram = isSram;
    uint64_t* offsetPtr;
    const uint64_t* memoryTotalSizePtr;
    const uint64_t* basePtr;

    getMemAttribPtrs(allocateInSram, memAttrib, memUsage, &offsetPtr, &memoryTotalSizePtr, &basePtr);
    MME_ASSERT((*basePtr) % m_mmeHal.getMemoryClSize() == 0, "base addr should be CL aligned");

    const unsigned unalignedOffset = isAligned ? 0 : (1 + rand() % (alignment - 1));
    addr = *basePtr + *offsetPtr + unalignedOffset;

    uint64_t tensorSize = t.getMemorySize();
    if (*offsetPtr + tensorSize > *memoryTotalSizePtr)
    {
        atomicColoredPrint(COLOR_YELLOW,
                           "[MME USER] Warning - Tensor - %s doesnt fit in %s - trying to allocate in %s\n",
                           t.getName().c_str(),
                           allocateInSram ? "SRAM" : "HBM",
                           allocateInSram ? "HBM" : "SRAM");
        // flip the allocation target memory
        allocateInSram = !allocateInSram;
        getMemAttribPtrs(allocateInSram, memAttrib, memUsage, &offsetPtr, &memoryTotalSizePtr, &basePtr);
        MME_ASSERT((*basePtr) % m_mmeHal.getMemoryClSize() == 0, "base addr should be CL aligned");
        MME_ASSERT(*offsetPtr + tensorSize <= *memoryTotalSizePtr, "Tensor doesnt fit in HBM & in SRAM");
        addr = *basePtr + *offsetPtr + unalignedOffset;
    }

    unsigned paddedTensorSize = div_round_up(tensorSize + unalignedOffset, alignment) * alignment;
    (*offsetPtr) += paddedTensorSize;

    MmePciDma dma;
    dma.size = tensorSize;
    dma.deviceAddr = addr;
    dma.hostAddr = t.data();
    dma.host2device = isInput;
    dma.isSram = isSram;
    memUsage.dmaList.push_back(dma);
}

void MmeUser::allocPrimaryTensors(const EMmeOpType op,
                              MmeDataParams& dataParams,
                              MmeTensorParams& tensorParams,
                              const MmeTensorMemoryAttrib& memAttrib,
                              MmeMemoryUsage& memUsage,
                              bool alignedAddresses)
{
    // Alloc tensors by their order in the tensor structs: x, w, y, o
    bool isInputX, isInputW, isInputY;
    setInputs(op, isInputX, isInputW, isInputY);

    if (tensorParams.xHost)
    {
        allocTensorDeviceMem(dataParams.operandInSram[e_mme_op_x], isInputX, *tensorParams.xHost, memAttrib, memUsage, tensorParams.xAddr, alignedAddresses);
    }
    if (tensorParams.wHost)
    {
        allocTensorDeviceMem(dataParams.operandInSram[e_mme_op_w], isInputW, *tensorParams.wHost, memAttrib, memUsage, tensorParams.wAddr, alignedAddresses);
    }
    if (tensorParams.yHost)
    {
        allocTensorDeviceMem(dataParams.operandInSram[e_mme_op_y], isInputY, *tensorParams.yHost, memAttrib, memUsage, tensorParams.yAddr, alignedAddresses);
    }
    if (tensorParams.oHost)
    {
        allocTensorDeviceMem(dataParams.operandInSram[e_mme_op_o], false,    *tensorParams.oHost, memAttrib, memUsage, tensorParams.oAddr, alignedAddresses);
    }
}

void MmeUser::allocAuxTensors(const EMmeOpType op,
                              MmeDataParams& dataParams,
                              MmeTensorParams& tensorParams,
                              const MmeTensorMemoryAttrib& memAttrib,
                              MmeMemoryUsage& memUsage,
                              bool alignedAddresses)
{

    // Alloc the aux tensors
    for (int i=0; i<MmeAuxTensorIdx::AUX_TENSOR_MAX_NUM; i++)
    {
        // todo: clean the code when masked bgemm aux tensors move to the front set of tensors
        AuxTensorData auxData = dataParams.tensorParams.front().auxTensors[i];
        if (auxData.pTensor != nullptr)
        {
            allocTensorDeviceMem(auxData.isSram, auxData.isInput, *auxData.pTensor, memAttrib, memUsage, auxData.addr, alignedAddresses);
            dataParams.tensorParams.front().auxTensors[i] = auxData;
        }
    }
}

void MmeUser::allocTensorsDeviceMem(const EMmeOpType op,
                                    const MmeDataParams& dataParams,
                                    const MmeTensorParams& tensorParams,
                                    const MmeTensorMemoryAttrib& memAttrib,
                                    MmeMemoryUsage& memUsage,
                                    uint64_t& addrA,
                                    uint64_t& addrB,
                                    uint64_t& addrC,
                                    uint64_t& addrO,
                                    bool alignedAddresses)
{
    OperandsInfo opInfo;
    makeOperandsInfo(op, dataParams, tensorParams, &opInfo);

    // operand A
    allocTensorDeviceMem(opInfo.aInSram, true, *opInfo.a, memAttrib, memUsage, addrA, alignedAddresses);

    // operand B
    if (!m_dmaDesc)
    {
        allocTensorDeviceMem(opInfo.bInSram, true, *opInfo.b, memAttrib, memUsage, addrB, alignedAddresses);
    }

    // operand C
    allocTensorDeviceMem(opInfo.cInSram, false, *opInfo.c, memAttrib, memUsage, addrC, alignedAddresses);

    // operand O
    if (tensorParams.oHost)
    {
        allocTensorDeviceMem(dataParams.operandInSram[e_mme_op_o],
                             false,
                             *tensorParams.oHost,
                             memAttrib,
                             memUsage,
                             addrO,
                             alignedAddresses);
    }
    else
    {
        addrO = 0;
    }
};

const MmeRecipe& MmeUser::getRecipe() const
{
    return m_descGenerator->getRecipe();
}
const CommonGeoAttr& MmeUser::getGeoAttr() const
{
    return m_descGenerator->getGeoAttr();
}

// This is the entry point of mme_user. It simulates the graph compiler in the mme_test. receives params for the mme,
// and outputs the activation list
bool MmeUser::createActivations(const EMmeOpType op,
                                const ConvolutionParams& conv,
                                const MmeTensorMemoryAttrib& memAttrib,
                                const MmeStrategy& strategy,
                                const MmeControls& controls,
                                const MmeMemoryConfig& memoryConfig,
                                MmeDataParams& dataParams,
                                MmeMemoryUsage& memUsage,
                                unsigned& actNum,
                                const MmeTestParams& testParams,
                                unsigned testId)
{
    // create activation list
    OperandRolesVec operandRolesVec;
    createLayerParams(op,
                      dataParams.wkldId,
                      conv,
                      dataParams.tensorParams,
                      strategy,
                      controls,
                      memoryConfig,
                      testParams,
                      m_subParams,
                      operandRolesVec);
    bool checkRoi = dataParams.memAccessChecker != nullptr;

    m_descGenerator =
        MmeCommon::MmeDescriptorGeneratorBase::createMmeDescGenerator(m_chipType,
                                                                      m_subParams[0].isNativeDmaOperation(),
                                                                      m_mmeNr);
    if (!m_descGenerator)
    {
        atomicColoredPrint(COLOR_RED, "Failed to create descriptor generator\n");
        return false;
    }

    // Generate activations
    m_descGenerator->setParams(m_subParams[0]);
    m_descGenerator->mmeGenerateActivations();
    actNum = m_descGenerator->getMmeActivationNr();
    const auto recipeDebugInfo = m_descGenerator->getRecipeDebugInfo();
    if (!recipeDebugInfo.empty())
    {
        for (const auto& dbgStr : recipeDebugInfo)
        {
            atomicColoredPrint(COLOR_BLUE, "[DEBUG] %s\n", dbgStr.c_str());
        }
    }

    const auto DescCacheDebugInfo = m_descGenerator->getDescCacheDebugInfo();
    if (!DescCacheDebugInfo.empty())
    {
        for (const auto& dbgStr : DescCacheDebugInfo)
        {
            atomicColoredPrint(COLOR_BLUE, "[DEBUG] %s\n", dbgStr.c_str());
        }
    }

    // Patch activations
    patchActivations(op,
                     memAttrib,
                     strategy,
                     dataParams,
                     memUsage,
                     testId);

    return true;
}

void MmeUser::doTensorAllocation(const EMmeOpType op,
                           const MmeTensorMemoryAttrib& memAttrib,
                           const MmeStrategy& strategy,
                           MmeDataParams& dataParams,
                           MmeMemoryUsage& memUsage)
{
    unsigned gemmNr = (strategy.dualGemm ? 2 : 1);

    for (unsigned gemm = 0; gemm < gemmNr; gemm++)
    {
        uint64_t addrA, addrB, addrC, addrO;
        allocPrimaryTensors(op,
                        dataParams,
                        dataParams.tensorParams[gemm],
                        memAttrib,
                        memUsage,
                        strategy.alignedAddresses);
    }
    allocAuxTensors(op,
                    dataParams,
                    dataParams.tensorParams[0],
                    memAttrib,
                    memUsage,
                    strategy.alignedAddresses);
}

void MmeUser::getTensorAddresses(const EMmeOpType op,
                                 const MmeTensorParams& tParams,
                                 uint64_t& addrA,
                                 uint64_t& addrB,
                                 uint64_t& addrC,
                                 uint64_t& addrO)
{
    EMmeInternalOperand internalOpX = mmeOpToInternalOp(e_mme_op_x, op);
    EMmeInternalOperand internalOpY = mmeOpToInternalOp(e_mme_op_y, op);
    EMmeInternalOperand internalOpW = mmeOpToInternalOp(e_mme_op_w, op);
    addrA = (internalOpX == e_mme_op_a ? tParams.xAddr : (internalOpY == e_mme_op_a ? tParams.yAddr : tParams.wAddr));
    addrB = (internalOpX == e_mme_op_b ? tParams.xAddr : (internalOpY == e_mme_op_b ? tParams.yAddr : tParams.wAddr));
    addrC = (internalOpX == e_mme_op_c ? tParams.xAddr : (internalOpY == e_mme_op_c ? tParams.yAddr : tParams.wAddr));
    addrO = tParams.oAddr;
}

bool MmeUser::patchActivations(const EMmeOpType op,
                               const MmeTensorMemoryAttrib& memAttrib,
                               const MmeStrategy& strategy,
                               MmeDataParams& dataParams,
                               MmeMemoryUsage& memUsage,
                               unsigned testId)
{
    // This struct holds for each port its actual tensor address and location (isSram).
    // It always holds a, b and c. If needed, it also holds aux tensors and secondary output
    MmePatchMetaData patchMetaData;

    // allocate tensors and patch activations
    if (strategy.dualGemm)
    {
        //  this logic will actually be done in runtime
        //  patch the actual gemm sizes, overriding the configuration done on desc generation
        for (unsigned gemm = 0; gemm < 2; gemm++)
        {
            uint64_t addrA, addrB, addrC, addrO;
            const auto& tParams = dataParams.tensorParams[gemm];
            getTensorAddresses(op, tParams, addrA, addrB, addrC, addrO);

            MmeTensorView x, w, y;
            initMmeTensorView(*tParams.xHost, &x);
            initMmeTensorView(*tParams.wHost, &w);
            initMmeTensorView(*tParams.yHost, &y);
            m_descGenerator->patchDualGemm(x,
                                           w,
                                           y,
                                           addrA,
                                           addrB,
                                           addrC,
                                           addrO,
                                           dataParams.operandInSram[e_mme_op_y],
                                           dataParams.operandInSram[e_mme_op_o],
                                           gemm);
        }
    }
    else
    {
        uint64_t addrA, addrB, addrC, addrO;
        const auto& tParams = dataParams.tensorParams.front();
        getTensorAddresses(op, tParams, addrA, addrB, addrC, addrO);

        EMmeOperand operandA = getInputFromOperation(op, true);
        EMmeOperand operandB = getInputFromOperation(op, false);
        EMmeOperand operandC = getOutputFromOperation(op);

        patchMetaData.bOperandUsed = (op != e_mme_memcpy && op != e_mme_trans);
        patchMetaData.oOperandUsed = (addrO != 0);

        patchMetaData.tensorMetaData[INPUT_TENSOR_A] = {addrA, dataParams.operandInSram[operandA]};
        patchMetaData.tensorMetaData[OUTPUT_TENSOR_C] = {addrC, dataParams.operandInSram[operandC]};
        if (patchMetaData.bOperandUsed)
        {
            patchMetaData.tensorMetaData[INPUT_TENSOR_B] = {addrB, dataParams.operandInSram[operandB]};
        }
        if (patchMetaData.oOperandUsed)
        {
            patchMetaData.tensorMetaData[OUTPUT_TENSOR_O] = {addrO, dataParams.operandInSram[e_mme_op_o]};
        }

        // Set the aux tensors
        const auto& auxData = dataParams.tensorParams[0].auxTensors;
        for (unsigned i=0; i < MmeAuxTensorIdx::AUX_TENSOR_MAX_NUM; i++)
        {
            MmeAuxTensorIdx idx = (MmeAuxTensorIdx)i;
            if (auxData[idx].pTensor)
            {
                TensorRoles role = m_descGenerator->getAuxTensorRole(idx);
                patchMetaData.tensorMetaData[role] = {auxData[idx].addr, auxData[idx].isSram};
            }
        }

        // Patch the mme tensors
        bool checkRoi = dataParams.memAccessChecker != nullptr;
        m_descGenerator->patchMmeDescriptors(patchMetaData, checkRoi);
    }

    for (int mmeIdx = 0; mmeIdx < m_mmeNr; mmeIdx++)
    {
        m_descGenerator->mmePatchSyncObjects(mmeIdx,
                                             dataParams.syncObjects[mmeIdx].Primary.second,
                                             dataParams.syncObjects[mmeIdx].Secondary.second,
                                             dataParams.syncObjects[mmeIdx].PrimarySlave.second,
                                             dataParams.syncObjects[mmeIdx].SecondarySlave.second);
    }

    m_descGenerator->patchContextId(testId);

    if (GCFG_PRINT_MME_DESCRIPTORS.value())
    {
        m_descGenerator->dumpDescriptors(true);
        auto dump = m_descGenerator->dumpDescriptors(false);
        for (const auto& actDump : dump)
        {
            LOG_INFO(MME_DESC_DUMP, "{}", fmt::join(actDump, "\n"));
        }
    }
    return true;
}

void MmeUser::patchTensors(MmeDataParams& dataParams, const uint64_t sramUsage, const uint64_t hbmUsage)
{
    for (auto operand : {e_mme_op_x, e_mme_op_w, e_mme_op_y, e_mme_op_o})
    {
        m_descGenerator->mmeIncrementDataTensorViews(operand, dataParams.operandInSram[operand] ? sramUsage : hbmUsage);
    }
}

void MmeUser::initProgram(std::list<CPProgram>& progs, std::list<CPProgram>& powerProgs, unsigned stream)
{
    for (unsigned mmeIdx = 0; mmeIdx < m_mmeNr; mmeIdx++)
    {
        progs.push_back(CPProgram(getMmeQueueId(mmeIdx, stream)));
        powerProgs.push_back(CPProgram(getMmeQueueId(mmeIdx, stream)));
    }
}

void MmeUser::addCmdsToProg(std::vector<MmeQmanCmd>& cmds, CPProgram& prog, CPProgram& powerProg)
{
    bool setup = true;
    for (auto& cmd : cmds)
    {
        auto& p = setup ? prog : powerProg;
        // define setup for next iteration.
        setup &= !cmd.power_last_setup_cmd;
        if (cmd.cmd_type == MME_REG_WRITE)
        {
            p.addTclSeq((cmd.reg_offset + getMmeCtrlBase()) & 0xFFFF, cmd.reg_values.data(), cmd.reg_values.size());
        }
        else if (cmd.cmd_type == MME_FENCE)
        {
            static const unsigned c_max_fence_val = 15;
            for (unsigned f = 0; f < cmd.fence_value; f += c_max_fence_val)
            {
                unsigned incWaitVal = std::min(c_max_fence_val, cmd.fence_value - f);
                pushFenceCmd(p, cmd.fence_idx, incWaitVal);
            }
        }
        else if (cmd.cmd_type == MME_WAIT)
        {
            pushWaitCmd(p, cmd.wait_cycles, cmd.wait_value, cmd.wait_idx);
        }
        else
        {
            MME_ASSERT(0, "invalid cmd type");
        }
    }
}

void MmeUser::generateSingleTestProgram(std::list<CPProgram>& progs,
                                        std::list<CPProgram>& powerProgs,
                                        std::vector<MmeQmanCmd> cmds[],
                                        SyncObjectManager& soMgr,
                                        MmeDataParams& dataParams,
                                        const bool firstValidTest,
                                        const bool clipInfIn,
                                        const bool configLfsr,
                                        const LfsrData& lfsrData,
                                        const unsigned seed,
                                        const unsigned stream,
                                        const unsigned testGroupSize,
                                        const unsigned testIdInGroup,
                                        PmuConfig pmuCfgMode)
{
    for (unsigned mmeIdx = 0; mmeIdx < m_mmeNr; mmeIdx++)
    {
        auto it = progs.begin();
        auto itPower = powerProgs.begin();
        //  need to advance the pointer to the relevant MME.
        //  if MMEs are limited use the same mme for multiple programs
        //  we actually dont need a list here IMO, in the future change to vector and clean this code
        std::advance(it, mmeIdx);
        std::advance(itPower, mmeIdx);

        CPProgram& prog = *it;
        CPProgram& powerProg = *itPower;

        // configurations that are relevant only for first test in group
        if (!m_powerTest && testIdInGroup == 0)
        {
            if (canDoStaticConfig() && firstValidTest && !m_powerTest /* TODO: loop*/)
            {
                addMmeLfsrInitSequence(prog, seed, lfsrData, configLfsr);
                addClipInfInputConfig(prog, clipInfIn);
                // need to add a barrier to make sure all static config are finished before the MME begins testing
                addMessageBarrier(prog);
            }
        }

        if (!m_powerTest)
        {
            soMgr.addPoleMonitor(prog,
                                 stream,
                                 mmeIdx,
                                 testIdInGroup,
                                 dataParams.syncObjects[mmeIdx].Primary.first,
                                 dataParams.syncObjects[mmeIdx].Secondary.first,
                                 dataParams.syncObjects[mmeIdx].PrimarySlave.first,
                                 dataParams.syncObjects[mmeIdx].SecondarySlave.first,
                                 testGroupSize,
                                 dataParams.soValues[mmeIdx]);
        }
        // begin building the unified program
        addCmdsToProg(cmds[mmeIdx], prog, powerProg);
        if (firstValidTest)
        {
            setPMUConfiguration(prog, pmuCfgMode);
        }
    }
    // Block the Qman for power tests
    if (m_powerTest && !m_scalFw)
    {
        for (auto& prog : progs)
        {
            setSoForPowerTest(prog, false);
        }
        for (auto& powerProg : powerProgs)
        {
            setSoForPowerTest(powerProg, true);
        }
    }
}

void MmeUser::setPMUConfiguration(CPProgram& prog, PmuConfig pmuCfgMode)
{
#if POWER_TEST_TODO
    // TODO: Check setPMUConfiguration(...) of Goya2
#endif  // POWER_TEST_TODO
}

std::array<std::pair<uint32_t, uint32_t>, 4> MmeUser::getRandomRedundancyFmaWithBitMask(unsigned seed)
{
    // randomly choose which fma is redundant and needs to be disabled.
    char* noRandom = getenv("MME_DISABLE_REDUN_RNDM");
    // default values - fma33 is redundant.
    std::array<std::pair<uint32_t, uint32_t>, 4> redun_val = {
        {{0x20, 0xFFFFFFFF}, {0x20, 0xFFFFFFFF}, {0x20, 0xFFFFFFFF}, {0x20, 0xFFFFFFFF}}};
    if (!noRandom)
    {
        std::mt19937 mt(seed);
        std::uniform_int_distribution<unsigned> dist(0, 32);
        for (auto& val : redun_val)
        {
            unsigned rval = dist(mt);  // choose random value between 0-32
            uint32_t bitMask = ~(1ull << rval);
            val = {rval, bitMask};
        }
    }
    return redun_val;
}
}  // namespace MmeCommon
