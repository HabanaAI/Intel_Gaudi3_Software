#include "include/utils/gaudi2_desc_dumper.h"
#include "spdlog/common.h"

void Gaudi2DescriptorDumper::dumpDescriptor(const Gaudi2::Mme::Desc& desc, bool fullDump)
{
    dumpAddress(desc.baseAddrCOut1.addr, "BaseAddrCout1");
    dumpAddress(desc.baseAddrCOut0.addr, "BaseAddrCout0");
    dumpAddress(desc.baseAddrA.addr, "BaseAddrA");
    dumpAddress(desc.baseAddrB.addr, "BaseAddrB", true);
    dumpBrain(desc.brains);
    dumpHeader(desc.header);
    dumpCtrl(desc.ctrl);
    dumpTensor(desc.tensorA, "tensorA");
    dumpTensor(desc.tensorB, "tensorB");
    dumpTensor(desc.tensorCOut, "tensorCout");
    dumpSyncObject(desc.syncObject);
    dumpAgu(desc);
    dumpUnsignedHex(desc.spatialSizeMinus1A, "spatialSizeMinus1A", false);
    dumpUnsignedHex(desc.spatialSizeMinus1B, "spatialSizeMinus1B", false);
    dumpUnsignedHex(desc.spatialSizeMinus1Cout, "spatialSizeMinus1Cout", true);
    dumpConv(desc.conv);
    dumpOuterLoop(desc.outerLoop);
    dumpUnsignedHex(desc.numIterationsMinus1, "numIterationsMinus1", true);
    dumpSBRepeat(desc.sbRepeat);
    dumpFp8Bias(desc.fp8Bias);
    dumpAxiUserData(desc.axiUserData);
    dumpAddress(desc.slaveSyncObject0Addr, "slaveSyncObject0Addr", false);
    dumpAddress(desc.slaveSyncObject1Addr, "slaveSyncObject0Addr", true);
    dumpPerfEvent(desc.perfEvtIn, "perfEvtIn");
    dumpPerfEvent(desc.perfEvtOut, "perfEvtOut");
    if (fullDump)
    {
        dumpRateLimits(desc.rateLimiter);
        dumpPcu(desc.pcu);
        dumpPowerLoop(desc.powerLoop);
        dumpUnsignedHex(desc.wkldID, "wkldID", true);
    }
}
void Gaudi2DescriptorDumper::dumpBrain(const Gaudi2::Mme::MmeBrainsCtrl& brains)
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
             << fmt::format("AguOut0: loopMask: {:#x} masterEn: {} slaveEn {}",
                            brains.aguOut0.loopMask,
                            brains.aguOut0.masterEn,
                            brains.aguOut0.slaveEn)
             << std::endl;
    m_stream << "    "
             << fmt::format("AguOut1: loopMask: {:#x} masterEn: {} slaveEn {}",
                            brains.aguOut1.loopMask,
                            brains.aguOut1.masterEn,
                            brains.aguOut1.slaveEn)
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
             << fmt::format("decEn: {} shuffleA: {} bgemm {} clipFpEu {} clipFpAp {}",
                            brains.decEn,
                            brains.shuffleA,
                            brains.bgemm,
                            brains.clipFpEu,
                            brains.clipFpAp)
             << std::endl;
    m_stream << "    "
             << fmt::format("sbACacheEn {} sbBCacheEn {} roundingMode {} reluEn {} noRollUp {} nullDesc {}",
                            brains.sbACacheEn,
                            brains.sbBCacheEn,
                            brains.roundingMode,
                            brains.reluEn,
                            brains.noRollup,
                            brains.nullDesc)
             << std::endl;
    m_stream << "    " << fmt::format("dw[0] : {:#x} dw[1] : {:#x}", brains.dw[0], brains.dw[1]) << std::endl;
}
void Gaudi2DescriptorDumper::dumpCtrl(const Gaudi2::Mme::MmeCtrl& ctrl)
{
    m_stream << "Ctrl: " << std::endl;
    m_stream << "    "
             << "eus[0]: " << std::endl;
    m_stream << "    "
             << fmt::format("sb0En: {} sb1En: {} sb2En: {} sb3En: {} sb4En: {}",
                            ctrl.eus[0].sb0En,
                            ctrl.eus[0].sb1En,
                            ctrl.eus[0].sb2En,
                            ctrl.eus[0].sb3En,
                            ctrl.eus[0].sb4En)
             << std::endl;
    m_stream << "    "
             << fmt::format("in0En: {} in1En: {} sb0Out: {} sb1Out: {} sb2Out: {}",
                            ctrl.eus[0].in0En,
                            ctrl.eus[0].in1En,
                            ctrl.eus[0].sb0OutEn,
                            ctrl.eus[0].sb2OutEn,
                            ctrl.eus[0].sb3OutEn)
             << std::endl;
    m_stream << "    "
             << fmt::format("sb0Sel: {:#x} sb1Sel: {:#x} sb2Sel: {:#x} sb3Sel: {:#x} sb4Sel: {:#x}",
                            ctrl.eus[0].sb0Sel,
                            ctrl.eus[0].sb1Sel,
                            ctrl.eus[0].sb2Sel,
                            ctrl.eus[0].sb3Sel,
                            ctrl.eus[0].sb4Sel)
             << std::endl;
    m_stream << "    "
             << "eus[1]: " << std::endl;
    m_stream << "    "
             << fmt::format("sb0En: {} sb1En: {} sb2En: {} sb3En: {} sb4En: {}",
                            ctrl.eus[1].sb0En,
                            ctrl.eus[1].sb1En,
                            ctrl.eus[1].sb2En,
                            ctrl.eus[1].sb3En,
                            ctrl.eus[1].sb4En)
             << std::endl;
    m_stream << "    "
             << fmt::format("in0En: {} in1En: {} sb0Out: {} sb1Out: {} sb2Out: {}",
                            ctrl.eus[1].in0En,
                            ctrl.eus[1].in1En,
                            ctrl.eus[1].sb0OutEn,
                            ctrl.eus[1].sb2OutEn,
                            ctrl.eus[1].sb3OutEn)
             << std::endl;
    m_stream << "    "
             << fmt::format("sb0Sel: {:#x} sb1Sel: {:#x} sb2Sel: {:#x} sb3Sel: {:#x} sb4Sel: {:#x}",
                            ctrl.eus[1].sb0Sel,
                            ctrl.eus[1].sb1Sel,
                            ctrl.eus[1].sb2Sel,
                            ctrl.eus[1].sb3Sel,
                            ctrl.eus[1].sb4Sel)
             << std::endl;
    m_stream << "    " << fmt::format("dw[0]: {:#x}   dw[1]: {:#x}", ctrl.dw[0], ctrl.dw[1]) << std::endl;
}
void Gaudi2DescriptorDumper::dumpHeader(const Gaudi2::Mme::MmeHeader& header)
{
    m_stream << "Header:" << std::endl;
    m_stream << "    "
             << fmt::format("transA: {} transB: {} advanceA: {} advanceB: {} advanceC: {}",
                            header.transA,
                            header.transB,
                            header.advanceA,
                            header.advanceB,
                            header.advanceC)
             << std::endl;
    m_stream << "    "
             << fmt::format("lowerA: {} lowerB: {} accumEn: {} rollAccum: {:#x}",
                            header.lowerA,
                            header.lowerB,
                            header.accumEn,
                            header.rollAccums)
             << std::endl;
    m_stream << "    "
             << fmt::format("aguReadsA: {:#x} aguReadsB: {:#x} doubleAccums: {} storeEn0: {} storeEn1: {}",
                            header.aguReadsA,
                            header.aguReadsB,
                            header.doubleAccums,
                            header.storeEn0,
                            header.storeEn1)
             << std::endl;
    m_stream << "    " << fmt::format("dataTypeIn: {} dataTypeOut: {}", header.dataTypeIn, header.dataTypeOut)
             << std::endl;
    m_stream << "    "
             << fmt::format("swapBaseAndOffsetA: {} swapBaseAndOffsetB: {} swapBaseAndOffsetOut: {}",
                            header.swapBaseAndOffsetA,
                            header.swapBaseAndOffsetB,
                            header.swapBaseAndOffsetOut)
             << std::endl;
    m_stream << "    "
             << fmt::format("storeColorSet0: {} storeColorSet1: {} hx2: {}",
                            header.storeColorSet0,
                            header.storeColorSet1,
                            header.hx2)
             << std::endl;
    m_stream << "    "
             << fmt::format("partialHeightLoopA: {:#x} partialHeightLoopB: {:#x} teBypassA {} teBypassB: {}",
                            header.partialHeightLoopA,
                            header.partialHeightLoopB,
                            header.teBypassA,
                            header.teBypassB)
             << std::endl;
    m_stream << "    " << fmt::format("dw[0]: {:#x}  dw[1]: {:#x}", header.dw[0], header.dw[1]) << std::endl;
}
void Gaudi2DescriptorDumper::dumpTensor(const Gaudi2::Mme::MmeTensorDesc& tensorDesc, const std::string& context)
{
    m_stream << "Tensor: " << context << std::endl;
    dumpArray(tensorDesc.validElements, "    validElements", true);
    dumpArray(tensorDesc.loopStride, "    loopStride", true);
    dumpArray(tensorDesc.roiSize, "    roiSize", true);
    dumpArray(tensorDesc.spatialStrides, "    spatialStrides", true);
    dumpArray(tensorDesc.startOffset, "    startOffset", true);

    m_stream << "    "
             << fmt::format("dw[0]: {:#x} dw[1]: {:#x} dw[2]: {:#x} dw[3]: {:#x} dw[4]: {:#x}",
                            tensorDesc.validElements[0],
                            tensorDesc.validElements[1],
                            tensorDesc.validElements[2],
                            tensorDesc.validElements[3],
                            tensorDesc.validElements[4])
             << std::endl;
    m_stream << "    "
             << fmt::format("dw[5]: {:#x} dw[6]: {:#x} dw[7]: {:#x} dw[8]: {:#x}  dw[9]: {:#x}",
                            tensorDesc.loopStride[0],
                            tensorDesc.loopStride[1],
                            tensorDesc.loopStride[2],
                            tensorDesc.loopStride[3],
                            tensorDesc.loopStride[4])
             << std::endl;
    m_stream << "    "
             << fmt::format("dw[10]: {:#x} dw[11]: {:#x} dw[12]: {:#x} dw[13]: {:#x}",
                            tensorDesc.roiSize[0],
                            tensorDesc.roiSize[1],
                            tensorDesc.roiSize[2],
                            tensorDesc.roiSize[3])
             << std::endl;
    m_stream << "    "
             << fmt::format("dw[14]: {:#x} dw[15]: {:#x} dw[16]: {:#x} dw[17]: {:#x}",
                            tensorDesc.spatialStrides[0],
                            tensorDesc.spatialStrides[1],
                            tensorDesc.spatialStrides[2],
                            tensorDesc.spatialStrides[3])
             << std::endl;
    m_stream << "    "
             << fmt::format("dw[18]: {:#x} dw[19]: {:#x} dw[20]: {:#x} dw[21]: {:#x}",
                            tensorDesc.startOffset[0],
                            tensorDesc.startOffset[1],
                            tensorDesc.startOffset[2],
                            tensorDesc.startOffset[3])
             << std::endl;
}
void Gaudi2DescriptorDumper::dumpSyncObject(const Gaudi2::Mme::MmeSyncObject& syncObject)
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
             << fmt::format("dw[0]: {:#x} dw[1]: {:#x} dw[2]: {:#x} dw[3]: {:#x} dw[4]: {:#x}",
                            syncObject.dw[0],
                            syncObject.dw[1],
                            syncObject.dw[2],
                            syncObject.dw[3],
                            syncObject.dw[4])
             << std::endl;
}
void Gaudi2DescriptorDumper::dumpAgu(const Gaudi2::Mme::Desc& desc)
{
    m_stream << "AguIn: " << std::endl;
    for (unsigned sb = 0; sb < Gaudi2::Mme::c_mme_sb_nr; sb++)
    {
        for (unsigned core = Gaudi2::Mme::MME_CORE_MASTER; core < Gaudi2::Mme::MME_CORE_PAIR_SIZE; core++)
        {
            std::string masterOrSlave = (core == 0) ? std::string("MASTER") : std::string("SLAVE");
            std::string context = "aguIn[" + std::to_string(sb) + "][" + masterOrSlave + "]";
            m_stream << "    roiBaseOffset: ";
            dumpArray(desc.aguIn[sb][core].roiBaseOffset, context, true);
        }
    }
    m_stream << "AguOut: " << std::endl;
    for (unsigned sb = 0; sb < Gaudi2::Mme::c_mme_wb_nr; sb++)
    {
        for (unsigned core = Gaudi2::Mme::MME_CORE_MASTER; core < Gaudi2::Mme::MME_CORE_PAIR_SIZE; core++)
        {
            std::string masterOrSlave = (core == 0) ? std::string("MASTER") : std::string("SLAVE");
            std::string context = "aguOut[" + std::to_string(sb) + "][" + masterOrSlave + "]";
            m_stream << "    roiBaseOffset: ";
            dumpArray(desc.aguOut[sb][core].roiBaseOffset, context, true);
        }
    }
}
void Gaudi2DescriptorDumper::dumpConv(const Gaudi2::Mme::MmeConvDesc& conv)
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
void Gaudi2DescriptorDumper::dumpOuterLoop(const Gaudi2::Mme::MmeOuterLoop& outerLoop)
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
void Gaudi2DescriptorDumper::dumpSBRepeat(const Gaudi2::Mme::MmeSBRepeat& sbRepeat)
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
void Gaudi2DescriptorDumper::dumpFp8Bias(const Gaudi2::Mme::MmeFP8Bias fp8Bias)
{
    m_stream << "Fp8Bias: " << std::endl;
    m_stream << "    " << fmt::format("a: {:#x} b: {:#x} out: {:#x}", fp8Bias.a, fp8Bias.b, fp8Bias.out) << std::endl;
}
void Gaudi2DescriptorDumper::dumpAxiUserData(const Gaudi2::Mme::MmeUserData& userData)
{
    m_stream << "AxiUserData: " << std::endl;
    m_stream << "    "
             << fmt::format("first: {:#x} steady: {:#x} mask: {:#x}", userData.first, userData.steady, userData.mask)
             << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", userData.dw) << std::endl;
}
void Gaudi2DescriptorDumper::dumpRateLimits(const Gaudi2::Mme::MmeRateLimiter& rateLimiter)
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
void Gaudi2DescriptorDumper::dumpPerfEvent(const Gaudi2::Mme::MmePerfEvt& perfEvt, const std::string& context)
{
    m_stream << "PerfEvent: " << context << std::endl;
    m_stream << "    "
             << fmt::format("value: {:#x} rst: {} incMask: {:#b} startEndMask: {:#b}",
                            perfEvt.value,
                            perfEvt.rst,
                            perfEvt.incMask,
                            perfEvt.startEndMask)
             << std::endl;
    m_stream << "    "
             << fmt::format("loopMask: {:#x} operand: {:#b} slaveSendsPerfEvents: {}",
                            perfEvt.loopMask,
                            perfEvt.operand,
                            perfEvt.slaveSendsPerfEvent)
             << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", perfEvt.dw) << std::endl;
}
void Gaudi2DescriptorDumper::dumpPcu(const Gaudi2::Mme::MmePCU& pcu)
{
    m_stream << "PCU: " << std::endl;
    m_stream << "    " << fmt::format("rlSaturation: {:#x}", pcu.rlSaturation) << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", pcu.dw) << std::endl;
}
void Gaudi2DescriptorDumper::dumpPowerLoop(const Gaudi2::Mme::MmePowerLoop& powerLoops)
{
    m_stream << "PowerLoops: " << std::endl;
    m_stream << "    "
             << fmt::format("ctrlMstr: {:#b} ctrlSlv: {:#b} powerLoopMd: {:#x}",
                            powerLoops.ctrlMstr,
                            powerLoops.ctrlSlv,
                            powerLoops.powerLoopMd)
             << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", powerLoops.dw) << std::endl;
}
