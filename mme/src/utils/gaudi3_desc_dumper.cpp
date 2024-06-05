#include "include/utils/gaudi3_desc_dumper.h"
#include "spdlog/common.h"

void Gaudi3DescriptorDumper::dumpDescriptor(const gaudi3::Mme::Desc& desc, bool fullDump)
{
    dumpAddress(desc.baseAddrCOut1.addr, "BaseAddrCout1");
    dumpAddress(desc.baseAddrCOut0.addr, "BaseAddrCout0");
    dumpAddress(desc.baseAddrA.addr, "BaseAddrA");
    dumpAddress(desc.baseAddrB.addr, "BaseAddrB", true);
    dumpBrain(desc.brains);
    dumpHeader(desc.header);
    dumpTensor(desc.tensorA, "tensorA", desc.header.dualGemm);
    dumpTensor(desc.tensorB, "tensorB", desc.header.dualGemm);
    dumpTensor(desc.tensorCOut, "tensorCout", desc.header.dualGemm);
    dumpAgu(desc);
    dumpUnsignedHex(desc.spatialSizeMinus1A, "spatialSizeMinus1A", false);
    dumpUnsignedHex(desc.spatialSizeMinus1B, "spatialSizeMinus1B", false);
    dumpUnsignedHex(desc.spatialSizeMinus1Cout, "spatialSizeMinus1Cout", true);
    dumpConv(desc.conv);
    dumpOuterLoop(desc.outerLoop);
    dumpUnsignedHex(desc.numIterationsMinus1, "numIterationsMinus1", true);
    dumpSBRepeat(desc.sbRepeat);
    dumpSyncObject(desc.syncObject);
    dumpNumerics(desc.numerics);
    dumpAxiAwUserData(desc.axiAwUserData);
    dumpAxiUserData(desc.axiUserDataA, "axiUserDataA");
    dumpAxiUserData(desc.axiUserDataB, "axiUserDataB");
    dumpAxiUserData(desc.axiUserDataCout, "axiUserDataCout");
    dumpCacheData(desc.axiCacheData);
    dumpPerfEvent(desc.perfEvtIn, "perfEvtIn");
    dumpPerfEvent(desc.perfEvtOut, "perfEvtOut");
    dumpPerfEvent(desc.perfEvtEU, "perfEvtEU");
    if (fullDump)
    {
        dumpRateLimits(desc.rateLimiter);
        dumpPower(desc.powerLoop);
        dumpUnsignedHex(desc.wkldID, "wkldID", true);
    }
}

void Gaudi3DescriptorDumper::dumpBrain(const gaudi3::Mme::MmeBrainsCtrl& brains)
{
    m_stream << "Brains:" << std::endl;
    m_stream << "    "
             << fmt::format("AguA: loopMask: {:#x} masterEn: {} slaveEn {}",
                            brains.aguA.loopMask,
                            brains.aguA.masterEn,
                            brains.aguA.slaveEn)
             << std::endl;
    m_stream << "    "
             << fmt::format("AguB: loopMask: {:#x} masterEn: {} slaveEn {}",
                            brains.aguB.loopMask,
                            brains.aguB.masterEn,
                            brains.aguB.slaveEn)
             << std::endl;
    m_stream << "    "
             << fmt::format("AguOut: loopMask: {:#x} masterEn: {} slaveEn {}",
                            brains.aguOut.loopMask,
                            brains.aguOut.masterEn,
                            brains.aguOut.slaveEn)
             << std::endl;
    m_stream << "    "
             << fmt::format("eu: loopMask: {:#x} masterEn: {} slaveEn {}",
                            brains.eu.loopMask,
                            brains.eu.masterEn,
                            brains.eu.slaveEn)
             << std::endl;
    m_stream << "    "
             << fmt::format("ap: loopMask: {:#x} masterEn: {} slaveEn {}",
                            brains.ap.loopMask,
                            brains.ap.masterEn,
                            brains.ap.slaveEn)
             << std::endl;
    m_stream << "    "
             << fmt::format("AguOutDma: loopMask: {:#x} masterEn: {} slaveEn {}",
                            brains.aguOutDma.loopMask,
                            brains.aguOutDma.masterEn,
                            brains.aguOutDma.slaveEn)
             << std::endl;
    m_stream << "    " << fmt::format("dw[0] : {:#x} dw[1] : {:#x}", brains.dw[0], brains.dw[1]) << std::endl;
}

void Gaudi3DescriptorDumper::dumpHeader(const gaudi3::Mme::MmeHeader& header)
{
    m_stream << "Header:" << std::endl;
    m_stream << "    "
             << fmt::format("transA: {} transB: {} sbTransA: {} sbTransB: {} accumEn: {}",
                            header.transA,
                            header.transB,
                            header.sbTransA,
                            header.sbTransB,
                            header.accumEn)
             << std::endl;
    m_stream << "    "
             << fmt::format("lowerA: {} lowerB: {} rollAccum: {:#x}", header.lowerA, header.lowerB, header.rollAccums)
             << std::endl;
    m_stream << "    "
             << fmt::format("storeEn0: {} storeEn1: {} reluEn: {} ", header.storeEn0, header.storeEn1, header.reluEn)
             << std::endl;
    m_stream << "    "
             << fmt::format("doubleAccums: {} bgemm: {} shuffleA: {} roundingMode: {}",
                            header.doubleAccums,
                            header.bgemm,
                            header.shuffleA,
                            header.roundingMode)
             << std::endl;
    m_stream << "    " << fmt::format("noRollUp: {} nullDesc: {}", header.noRollup, header.nullDesc) << std::endl;
    m_stream << "    " << fmt::format("dataTypeIn: {} dataTypeOut: {}", header.dataTypeIn, header.dataTypeOut)
             << std::endl;
    m_stream << "    "
             << fmt::format("swapBaseAndOffsetA: {} swapBaseAndOffsetB: {} swapBaseAndOffsetOut: {}",
                            header.swapBaseAndOffsetA,
                            header.swapBaseAndOffsetB,
                            header.swapBaseAndOffsetOut)
             << std::endl;
    m_stream << "    "
             << fmt::format("opANonShared: {} clipFpEu: {} clipFpAp: {} sbACacheEn: {} sbBCacheEn: {}",
                            header.opANonShared,
                            header.clipFpEu,
                            header.clipFpAp,
                            header.sbACacheEn,
                            header.sbBCacheEn)
             << std::endl;
    m_stream << "    "
             << fmt::format("partialHeightLoopA: {:#x} partialHeightLoopB: {:#x}",
                            header.partialHeightLoopA,
                            header.partialHeightLoopB)
             << std::endl;
    m_stream << "    "
             << fmt::format("teBypassA {} teBypassB: {} storeColorSet0: {} storeColorSet1: {}",
                            header.teBypassA,
                            header.teBypassB,
                            header.storeColorSet0,
                            header.storeColorSet1)
             << std::endl;
    m_stream << "    "
             << fmt::format("teAccelA: {} sftzFp32ToFp8: {} wbCacheEn: {}",
                            header.teAccelA,
                            header.sftzFp32ToFp8,
                            header.wbCacheEn)
             << std::endl;
    m_stream << "    " << fmt::format("dualGemm: {} dmaMode: {} ftz: {}", header.dualGemm, header.dmaMode, header.ftz)
             << std::endl;
    m_stream << "    " << fmt::format("dw[0]: {:#x}  dw[1]: {:#x}", header.dw[0], header.dw[1]) << std::endl;
}

void Gaudi3DescriptorDumper::dumpTensor(const gaudi3::Mme::MmeTensorDesc& tensorDesc,
                                        const std::string& context,
                                        bool dualGemm)
{
    m_stream << "Tensor: " << context << std::endl;
    if (dualGemm)
    {
        dumpArray(tensorDesc.dualGemm.validElements[0], "    dualGemm.validElements[0]");
        dumpArray(tensorDesc.dualGemm.validElements[1], " dualGemm.validElements[1]", true);
        dumpUnsignedHex(tensorDesc.dualGemm.spatialSizeMinus1Gemm1, "dualGemm.spatialSizeMinus1Gemm1", true);
        dumpArray(tensorDesc.dualGemm.roiSize[0], "    dualGemm.roiSize[0]");
        dumpArray(tensorDesc.dualGemm.roiSize[1], "    dualGemm.roiSize[1]");
        dumpArray(tensorDesc.dualGemm.spatialStrides, "    dualGemm.spatialSize", true);
        dumpAddress(tensorDesc.dualGemm.baseAddrGemm1.addr, "    dualGemm.baseAddrGemm1", true);
        dumpArray(tensorDesc.dualGemm.startOffset[0], "    dualGemm.startOffset[0]", true);
        dumpArray(tensorDesc.dualGemm.startOffset[1], "    dualGemm.startOffset[1]", true);
        dumpAddress(tensorDesc.dualGemm.baseAddrGemm1Dup.addr, "    dualGemm.baseAddrGemm1Dup", true);
    }
    else
    {
        dumpArray(tensorDesc.validElements, "    validElements", true);
        dumpArray(tensorDesc.loopStride, "    loopStride", true);
        dumpArray(tensorDesc.roiSize, "    roiSize", true);
        dumpArray(tensorDesc.spatialStrides, "    spatialStrides", true);
        dumpArray(tensorDesc.startOffset, "    startOffset", true);
    }
    m_stream << "    "
             << fmt::format("dw[0]: {:#x} dw[1]: {:#x} dw[2]: {:#x} dw[3]: {:#x} dw[4]: {:#x}",
                            tensorDesc.dw[0],
                            tensorDesc.dw[1],
                            tensorDesc.dw[2],
                            tensorDesc.dw[3],
                            tensorDesc.dw[4])
             << std::endl;
    m_stream << "    "
             << fmt::format("dw[5]: {:#x} dw[6]: {:#x} dw[7]: {:#x} dw[8]: {:#x}",
                            tensorDesc.dw[5],
                            tensorDesc.dw[6],
                            tensorDesc.dw[7],
                            tensorDesc.dw[8])
             << std::endl;
    m_stream << "    "
             << fmt::format("dw[9]: {:#x} dw[10]: {:#x} dw[11]: {:#x} dw[12]: {:#x}",
                            tensorDesc.dw[9],
                            tensorDesc.dw[10],
                            tensorDesc.dw[11],
                            tensorDesc.dw[12])
             << std::endl;
    m_stream << "    "
             << fmt::format("dw[13]: {:#x} dw[14]: {:#x} dw[15]: {:#x} dw[16]: {:#x}",
                            tensorDesc.dw[13],
                            tensorDesc.dw[14],
                            tensorDesc.dw[15],
                            tensorDesc.dw[16])
             << std::endl;
    m_stream << "    "
             << fmt::format("dw[17]: {:#x} dw[18]: {:#x} dw[19]: {:#x} dw[20]: {:#x}  dw[21]: {:#x}",
                            tensorDesc.dw[17],
                            tensorDesc.dw[18],
                            tensorDesc.dw[19],
                            tensorDesc.dw[20],
                            tensorDesc.dw[21])
             << std::endl;
}

void Gaudi3DescriptorDumper::dumpAgu(const gaudi3::Mme::Desc& desc)
{
    m_stream << "AguIn: " << std::endl;
    for (unsigned core = gaudi3::Mme::MME_MASTER; core < gaudi3::Mme::MME_PAIR_SIZE; core++)
    {
        for (unsigned sb = 0; sb < gaudi3::Mme::c_mme_sb_nr; sb++)
        {
            std::string masterOrSlave =
                (core == gaudi3::Mme::MME_MASTER) ? std::string("MASTER") : std::string("SLAVE");
            std::string context = "aguIn[" + masterOrSlave + "][" + std::to_string(sb) + "]";
            m_stream << "    roiBaseOffset: ";
            dumpArray(desc.aguIn[core][sb].roiBaseOffset, context, true);
        }
    }
    m_stream << "AguOut: " << std::endl;
    for (unsigned core = gaudi3::Mme::MME_MASTER; core < gaudi3::Mme::MME_PAIR_SIZE; core++)
    {
        std::string masterOrSlave = (core == gaudi3::Mme::MME_MASTER) ? std::string("MASTER") : std::string("SLAVE");
        std::string context = "aguOut[" + masterOrSlave + "]";
        m_stream << "    roiBaseOffset: ";
        dumpArray(desc.aguOut[core].roiBaseOffset, context, true);
    }
}

void Gaudi3DescriptorDumper::dumpConv(const gaudi3::Mme::MmeConvDesc& conv)
{
    m_stream << "Conv: " << std::endl;
    m_stream << "    "
             << fmt::format("kernelSizeMinus1.dim[0]: {:#x} kernelSizeMinus1.dim[1]: {:#x} kernelSizeMinus1.dim[2]: "
                            "{:#x} kernelSizeMinus1.dim[3]: {:#x}",
                            conv.kernelSizeMinus1.dim[0],
                            conv.kernelSizeMinus1.dim[1],
                            conv.kernelSizeMinus1.dim[2],
                            conv.kernelSizeMinus1.dim[3])
             << std::endl;
    m_stream << "    "
             << fmt::format("associatedDims[0]: dimA: {} dimB: {} dimOut: {}",
                            conv.associatedDims[0].dimA,
                            conv.associatedDims[0].dimB,
                            conv.associatedDims[0].dimOut)
             << std::endl;
    m_stream << "    "
             << fmt::format("associatedDims[1]: dimA: {} dimB: {} dimOut: {}",
                            conv.associatedDims[1].dimA,
                            conv.associatedDims[1].dimB,
                            conv.associatedDims[1].dimOut)
             << std::endl;
    m_stream << "    "
             << fmt::format("associatedDims[2]: dimA: {} dimB: {} dimOut: {}",
                            conv.associatedDims[2].dimA,
                            conv.associatedDims[2].dimB,
                            conv.associatedDims[2].dimOut)
             << std::endl;
    m_stream << "    "
             << fmt::format("associatedDims[3]: dimA: {} dimB: {} dimOut: {}",
                            conv.associatedDims[3].dimA,
                            conv.associatedDims[3].dimB,
                            conv.associatedDims[3].dimOut)
             << std::endl;
    m_stream << "    " << fmt::format("dw[0]: {:#x} ", conv.kernelSizeMinus1.dw) << std::endl;
    m_stream << "    "
             << fmt::format("w[1]: {:#x} w[2]: {:#x} w[3]: {:#x} w[4]: {:#x}",
                            conv.associatedDims[0].w,
                            conv.associatedDims[1].w,
                            conv.associatedDims[2].w,
                            conv.associatedDims[3].w)
             << std::endl;
}

void Gaudi3DescriptorDumper::dumpOuterLoop(const gaudi3::Mme::MmeOuterLoop& outerLoop)
{
    m_stream << "OuterLoop: " << std::endl;
    m_stream << "    "
             << fmt::format("associatedDims[0]: dimA: {} dimB: {} dimOut: {}",
                            outerLoop.associatedDims.dimA,
                            outerLoop.associatedDims.dimB,
                            outerLoop.associatedDims.dimOut)
             << std::endl;
    m_stream << "    " << fmt::format("sizeMinus1: {:#x}", outerLoop.sizeMinus1) << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", outerLoop.dw) << std::endl;
}

void Gaudi3DescriptorDumper::dumpSBRepeat(const gaudi3::Mme::MmeSBRepeat& sbRepeat)
{
    m_stream << "SBRepeat:" << std::endl;
    m_stream << "    "
             << fmt::format("repeatAMinus1: {:#x} repeatBMinus1: {:#x}", sbRepeat.repeatAMinus1, sbRepeat.repeatBMinus1)
             << std::endl;
    m_stream << "    "
             << fmt::format("repeatAMask: {:#x} repeatBMask: {:#x}", sbRepeat.repeatAMask, sbRepeat.repeatBMask)
             << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", sbRepeat.dw) << std::endl;
}

void Gaudi3DescriptorDumper::dumpSyncObject(const gaudi3::Mme::MmeSyncObject& syncObject)
{
    m_stream << "SyncObject:" << std::endl;
    m_stream << "    "
             << fmt::format("signalMask0 {:#x} signalEn0: {} signalMask1: {:#x} signalEn1: {}",
                            syncObject.signalMask0,
                            syncObject.signalEn0,
                            syncObject.signalMask1,
                            syncObject.signalEn1)
             << std::endl;
    m_stream << "    "
             << fmt::format("masterWaitForSlaveFence: {} slaveSendFence2Master: {} slaveSignalEn: {}",
                            syncObject.masterWaitForSlaveFence,
                            syncObject.slaveSendFence2Master,
                            syncObject.slaveSignalEn)
             << std::endl;
    m_stream << "    "
             << fmt::format("slave0UseSlaveSOAddr: {} slave1UseSlaveSOAddr: {} slave0UseMasterSoAddrPlus4: {} "
                            "slave1UseMasterSOAddrPlus4: {}",
                            syncObject.slave0UseSlaveSOAddr,
                            syncObject.slave1UseSlaveSOAddr,
                            syncObject.slave0UseMasterSOAddrPlus4,
                            syncObject.slave1UseMasterSOAddrPlus4)
             << std::endl;
    m_stream << "    "
             << fmt::format("so0Addr: {:#x} soVal.so0Value: {:#x} so0Val.soPerfEn: {} so0Val.soOp: {}",
                            syncObject.so0Addr,
                            syncObject.so0Val.soValue,
                            syncObject.so0Val.soPerfEn,
                            syncObject.so0Val.soOp)
             << std::endl;
    m_stream << "    "
             << fmt::format("so1Addr: {:#x} soVal.so1Value: {:#x} so1Val.soPerfEn: {} so1Val.soOp: {}",
                            syncObject.so1Addr,
                            syncObject.so1Val.soValue,
                            syncObject.so1Val.soPerfEn,
                            syncObject.so1Val.soOp)
             << std::endl;
    m_stream << "    "
             << fmt::format("slaveSo0Addr: {:#x} slaveSo1Addr: {}", syncObject.slaveSo0Addr, syncObject.slaveSo1Addr)
             << std::endl;
    m_stream << "    "
             << fmt::format("dw[0]: {:#x} dw[1]: {:#x} dw[2]: {:#x} dw[3]: {:#x}",
                            syncObject.dw[0],
                            syncObject.dw[1],
                            syncObject.dw[2],
                            syncObject.dw[3])
             << std::endl;
    m_stream << "    "
             << fmt::format("dw[4]: {:#x} dw[5]: {:#x} dw[6]: {:#x} ",
                            syncObject.dw[4],
                            syncObject.dw[5],
                            syncObject.dw[6])
             << std::endl;
}

void Gaudi3DescriptorDumper::dumpNumerics(const gaudi3::Mme::MmeNumericFlavors& numerics)
{
    m_stream << "Numerics:" << std::endl;
    m_stream << "    "
             << fmt::format("biasA: {} biasB: {} biasOut: {} accRoundinMode: {}",
                            numerics.biasA,
                            numerics.biasB,
                            numerics.biasOut,
                            numerics.accRoundingMode)
             << std::endl;
    m_stream << "    "
             << fmt::format("fp8FlavorA: {} fp8FlavorB: {} fp8FlavorOut: {}",
                            numerics.fp8FlavorA,
                            numerics.fp8FlavorB,
                            numerics.fp8FlavorOut)
             << std::endl;
    m_stream << "    "
             << fmt::format("fp16FlavorA: {} fp16FlavorB: {} fp16FlavorOut: {}",
                            numerics.fp16FlavorA,
                            numerics.fp16FlavorB,
                            numerics.fp16FlavorOut)
             << std::endl;
    m_stream << "    "
             << fmt::format("infNanModeA: {} infNanModeB: {} infNanModeOut: {}",
                            numerics.infNanModeA,
                            numerics.infNanModeB,
                            numerics.infNanModeOut)
             << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", numerics.dw) << std::endl;
}

void Gaudi3DescriptorDumper::dumpAxiAwUserData(const gaudi3::Mme::MmeAwUserData& awUserData)
{
    m_stream << "AxiAwUserData: " << std::endl;
    m_stream << "    "
             << fmt::format("first: {:#x} steady: {:#x} mask: {:#x}",
                            awUserData.first,
                            awUserData.steady,
                            awUserData.mask)
             << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", awUserData.dw) << std::endl;
}

void Gaudi3DescriptorDumper::dumpAxiUserData(const gaudi3::Mme::MmeUserData& userData, const std::string& context)
{
    m_stream << "AxiUserData: " << context << std::endl;
    m_stream << "    "
             << fmt::format("qosFirst: {:#x} qosSteady: {:#x} qosMask: {:#x}",
                            userData.qosFirst,
                            userData.qosSteady,
                            userData.qosMask);
    m_stream << "    " << fmt::format("mcid: {:#x} clss: {:#x}", userData.mcid, userData.clss) << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", userData.dw) << std::endl;
}

void Gaudi3DescriptorDumper::dumpCacheData(const gaudi3::Mme::MmeCacheData& cacheData)
{
    m_stream << "CacheData: " << std::endl;
    m_stream << "    "
             << fmt::format("aguA: {:#x} aguB: {:#x} aguOut: {:#x}", cacheData.aguA, cacheData.aguB, cacheData.aguOut)
             << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", cacheData.dw) << std::endl;
}

void Gaudi3DescriptorDumper::dumpPerfEvent(const gaudi3::Mme::MmePerfEvt& perfEvt, const std::string& context)
{
    m_stream << "PerfEvent: " << context << std::endl;
    m_stream << "    "
             << fmt::format("value: {:#x} rst: {} incEn: {:#b} startEndEn: {:#b}",
                            perfEvt.value,
                            perfEvt.rst,
                            perfEvt.incEn,
                            perfEvt.startEndEn)
             << std::endl;
    m_stream << "    "
             << fmt::format("loopMask: {:#x} operand: {:#b} slaveSendsPerfEvents: {}",
                            perfEvt.loopMask,
                            perfEvt.operand,
                            perfEvt.slaveSendsPerfEvent)
             << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", perfEvt.dw) << std::endl;
}

void Gaudi3DescriptorDumper::dumpRateLimits(const gaudi3::Mme::MmeRateLimiter& rateLimiter)
{
    m_stream << "RateLimiter: " << std::endl;
    m_stream << "    "
             << fmt::format("aguA: {:#x} aguB: {:#x} aguOut: {:#x} eu: {:#x}",
                            rateLimiter.aguA,
                            rateLimiter.aguB,
                            rateLimiter.aguOut,
                            rateLimiter.eu)
             << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", rateLimiter.dw) << std::endl;
}

void Gaudi3DescriptorDumper::dumpPower(const gaudi3::Mme::MmePower& power)
{
    m_stream << "Power: " << std::endl;
    m_stream << "    " << fmt::format("loopCtrl: {:#b} loopMd: {:#x} ", power.loopCtrl, power.loopMd) << std::endl;
    m_stream << "    "
             << fmt::format("pmuRlSaturation: {:#x} sbOppDisA: {} sbOppDisB: {}",
                            power.pmuRlSaturation,
                            power.sbOppDisA,
                            power.sbOppDisB)
             << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", power.dw) << std::endl;
}
