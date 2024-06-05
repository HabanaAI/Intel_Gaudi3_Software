#include "include/utils/gaudi_desc_dumper.h"
#include "spdlog/common.h"

void GaudiDescriptorDumper::dumpDescriptor(const Mme::Desc& desc, bool fullDump)
{
    dumpAddress(desc.baseAddrHighS, "baseAddrHighS");
    dumpAddress(desc.baseAddrLowS, "baseAddrLowS");
    dumpAddress(desc.baseAddrHighL, "baseAddrHighL");
    dumpAddress(desc.baseAddrLowL, "baseAddrLowL", true);
    dumpAddress(desc.baseAddrHighO, "baseAddrHighO");
    dumpAddress(desc.baseAddrLowO, "baseAddrLowO", true);
    dumpHeader(desc.header);
    dumpConv(desc.conv);
    dumpUnsignedHex(desc.numIterationsMinus1, "numIterationsMinus1", true);
    dumpOuterLoop(desc.outerLoop);
    dumpTensor(desc.tensorS, "tensorS");
    dumpTensor(desc.tensorL, "tensorL");
    dumpTensor(desc.tensorO, "tensorO");
    dumpAgu(desc);
    dumpSBRepeat(desc.sbRepeat);
    dumpSyncObject(desc.syncObject);
    dumpAxiUserData(desc.axiUserData);
    dumpPerfEvent(desc.perfEvtS, "perfEvtS");
    dumpPerfEvent(desc.perfEvtL[Mme::e_mme_local], "perfEvtL[local]");
    dumpPerfEvent(desc.perfEvtL[Mme::e_mme_remote], "perfEvtL[remote]");
    dumpPerfEvent(desc.perfEvtO[Mme::e_mme_local], "perfEvtO[local]");
    dumpPerfEvent(desc.perfEvtO[Mme::e_mme_remote], "perfEvtO[remote]");
    if (fullDump)
    {
        dumpUnsignedHex(desc.paddingValueS, "paddingValueS");
        dumpUnsignedHex(desc.paddingValueL, "paddingValueL", true);
        dumpRateLimits(desc.rateLimiter);


        dumpPcu(desc.pcu);
    }
}
void GaudiDescriptorDumper::dumpHeader(const Mme::MmeHeader& header)
{
    m_stream << "header:" << std::endl;
    m_stream << "    " << fmt::format("transS: {} transL: {} transO: {} ", header.transS, header.transL, header.transO)
             << std::endl;
    m_stream << "    "
             << fmt::format("advanceS: {} advanceL: {} advanceO: {}", header.advanceS, header.advanceL, header.advanceO)
             << std::endl;
    m_stream << "    "
             << fmt::format("lowerL: {} lowerS: {} accumMask: {:#x} accStoreIncDisable: {}",
                            header.lowerL,
                            header.lowerS,
                            header.accumMask,
                            header.accStoreIncDisable)
             << std::endl;
    m_stream << "    "
             << fmt::format("roundingMode: {:#x} accum: {} storeEn: {} rollAccums {:#x}",
                            header.roundingMode,
                            header.accum,
                            header.storeEn,
                            header.rollAccums)
             << std::endl;
    m_stream << "    " << fmt::format("dataTypeIn: {} dataTypeOut: {}", header.dataTypeIn, header.dataTypeOut)
             << std::endl;
    m_stream << "    "
             << fmt::format("signalMask: {:#x} signalEn: {} reluEn: {}",
                            header.signalMask,
                            header.signalEn,
                            header.reluEn)
             << std::endl;
    m_stream << "    "
             << fmt::format("partialHeightLoop - Shared: {:#x} LLocal: {:#x} LRemote: {:#x}",
                            header.partialHeightLoopS,
                            header.partialHeightLoopLLocal,
                            header.partialHeightLoopLRemote)
             << std::endl;
    m_stream << "    "
             << fmt::format("                    OLocal: {:#x} ORemote: {:#x}",
                            header.partialHeightLoopOLocal,
                            header.partialHeightLoopORemote)
             << std::endl;
    m_stream << "    " << fmt::format("fpEn: {} euBEn: {}", header.fpEn, header.euBEn) << std::endl;
    m_stream << "    " << fmt::format("dw[0]: {:#x}  dw[1]: {:#x}", header.dw[0], header.dw[1]) << std::endl;
}
void GaudiDescriptorDumper::dumpTensor(const Mme::MmeTensorDesc& tensorDesc, const std::string& context)
{
    m_stream << "Tensor: " << context << std::endl;
    m_stream << "    ";
    dumpArray(tensorDesc.validElements, "validElements", true);
    m_stream << "    ";
    dumpArray(tensorDesc.loopStride, "loopStride", true);
    m_stream << "    ";
    dumpArray(tensorDesc.roiSize, "roiSize", true);
    m_stream << "    ";
    dumpArray(tensorDesc.spatialStrides, "spatialStrides", true);
    m_stream << "    ";
    dumpUnsignedHex(tensorDesc.spatialSizeMinus1, "spatialSizeMinus1", true);

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
             << fmt::format("dw[14]: {:#x} dw[15]: {:#x} dw[16]: {:#x} dw[17]: {:#x} dw[18]: {:#x}",
                            tensorDesc.spatialStrides[0],
                            tensorDesc.spatialStrides[1],
                            tensorDesc.spatialStrides[2],
                            tensorDesc.spatialStrides[3],
                            tensorDesc.spatialSizeMinus1)
             << std::endl;
}
void GaudiDescriptorDumper::dumpSyncObject(const Mme::MmeSyncObject& syncObject)
{
    m_stream << "syncObject:" << std::endl;
    m_stream << "    "
             << fmt::format("addLow[0]: {:#x}, addrLow[1]: {:#x} addrHigh: {:#x} ",
                            syncObject.addrLow[0],
                            syncObject.addrLow[1],
                            syncObject.addrHigh)
             << std::endl;
    m_stream << "    "
             << fmt::format("SoVal: value: {:#x} perfEn: {} operation: {}",
                            syncObject.value,
                            syncObject.perfEn,
                            syncObject.operation)
             << std::endl;
    m_stream << "    " << fmt::format("dw[0]: {:#x} dw[1]: {:#x}", syncObject.dw[0], syncObject.dw[1]) << std::endl;
}
void GaudiDescriptorDumper::dumpAgu(const Mme::Desc& desc)
{
    m_stream << "AguS: " << std::endl;
    dumpArray(desc.aguS.roiBaseOffset, "    roiBaseOffset", true);
    dumpArray(desc.aguS.startOffset, "    startOffset", true);
    m_stream << "AguL-Local: " << std::endl;
    dumpArray(desc.aguL[Mme::e_mme_local].roiBaseOffset, "    roiBaseOffset", true);
    dumpArray(desc.aguL[Mme::e_mme_local].startOffset, "    startOffset", true);
    m_stream << "AguL-Remote: " << std::endl;
    dumpArray(desc.aguL[Mme::e_mme_remote].roiBaseOffset, "    roiBaseOffset", true);
    dumpArray(desc.aguL[Mme::e_mme_remote].startOffset, "     startOffset", true);
    m_stream << "AguO-Local: " << std::endl;
    dumpArray(desc.aguO[Mme::e_mme_local].roiBaseOffset, "    roiBaseOffset", true);
    dumpArray(desc.aguO[Mme::e_mme_local].startOffset, "    startOffset", true);
    m_stream << "AguO-Remote: " << std::endl;
    dumpArray(desc.aguO[Mme::e_mme_remote].roiBaseOffset, "    roiBaseOffset", true);
    dumpArray(desc.aguO[Mme::e_mme_remote].startOffset, "    startOffset", true);
}
void GaudiDescriptorDumper::dumpConv(const Mme::MmeConvDesc& conv)
{
    m_stream << "conv: " << std::endl;
    dumpArray(conv.kernelSizeMinus1.dim, "    kernelSizeMinus1.dim", true);
    m_stream << "    "
             << fmt::format("associatedDims[0]: dimS: {} dimL: {} dimO: {}",
                            conv.associatedDims[0].dimS,
                            conv.associatedDims[0].dimL,
                            conv.associatedDims[0].dimO)
             << std::endl;
    m_stream << "    "
             << fmt::format("associatedDims[1]: dimS: {} dimL: {} dimO: {}",
                            conv.associatedDims[1].dimS,
                            conv.associatedDims[1].dimL,
                            conv.associatedDims[1].dimO)
             << std::endl;
    m_stream << "    "
             << fmt::format("associatedDims[2]: dimS: {} dimL: {} dimO: {}",
                            conv.associatedDims[2].dimS,
                            conv.associatedDims[2].dimL,
                            conv.associatedDims[2].dimO)
             << std::endl;
    m_stream << "    "
             << fmt::format("associatedDims[3]: dimS: {} dimL: {} dimO: {}",
                            conv.associatedDims[3].dimS,
                            conv.associatedDims[3].dimL,
                            conv.associatedDims[3].dimO)
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
void GaudiDescriptorDumper::dumpOuterLoop(const Mme::MmeOuterLoop& outerLoop)
{
    m_stream << "outerLoop: " << std::endl;
    m_stream << "    "
             << fmt::format("associatedDims: dimS: {} dimL: {} dimO: {}",
                            outerLoop.associatedDims.dimS,
                            outerLoop.associatedDims.dimL,
                            outerLoop.associatedDims.dimO)
             << std::endl;
    m_stream << "    " << fmt::format("sizeMinus1: {:#x}", outerLoop.sizeMinus1) << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", outerLoop.dw) << std::endl;
}
void GaudiDescriptorDumper::dumpSBRepeat(const Mme::MmeSBRepeat& sbRepeat)
{
    m_stream << "sbRepeat:" << std::endl;
    m_stream << "    "
             << fmt::format("repeatSMinus1: {:#x} aguSLoopMask: {:#x} loadS: {} teEnS: {}",
                            sbRepeat.repeatSMinus1,
                            sbRepeat.aguSLoopMask,
                            sbRepeat.loadS,
                            sbRepeat.teEnS)
             << std::endl;
    m_stream << "    "
             << fmt::format("repeatLMinus1: {:#x} aguLLoopMask: {:#x} loadL: {} teEnL: {}",
                            sbRepeat.repeatLMinus1,
                            sbRepeat.aguLLoopMask,
                            sbRepeat.loadL,
                            sbRepeat.teEnL)
             << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", sbRepeat.dw) << std::endl;
}
void GaudiDescriptorDumper::dumpAxiUserData(const Mme::MmeUserData& userData)
{
    m_stream << "axiUserData: " << std::endl;
    m_stream << "    "
             << fmt::format("first: {:#x} steady: {:#x} mask: {:#x}", userData.first, userData.steady, userData.mask)
             << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", userData.dw) << std::endl;
}
void GaudiDescriptorDumper::dumpRateLimits(const Mme::MmeRateLimeter& rateLimiter)
{
    m_stream << "rateLimiter: " << std::endl;
    m_stream << "    "
             << fmt::format("aguS: {:#x} aguL: {:#x} aguO: {:#x} ",
                            rateLimiter.aguS,
                            rateLimiter.aguL,
                            rateLimiter.aguO)
             << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", rateLimiter.dw) << std::endl;
}
void GaudiDescriptorDumper::dumpPerfEvent(const Mme::MmePerfEvt& perfEvt, const std::string& context)
{
    m_stream << "perfEvent: " << context << std::endl;
    m_stream << "    "
             << fmt::format("value: {:#x} rst: {} incMask: {:#b} startEndMask: {:#b}",
                            perfEvt.value,
                            perfEvt.rst,
                            perfEvt.incMask,
                            perfEvt.startEndMask)
             << std::endl;
    m_stream << "    " << fmt::format("loopMask: {:#x}", perfEvt.loopMask) << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", perfEvt.dw) << std::endl;
}
void GaudiDescriptorDumper::dumpPcu(const Mme::MmePCU& pcu)
{
    m_stream << "pcu: " << std::endl;
    m_stream << "    " << fmt::format("rlSaturation: {:#x}", pcu.rlSaturation) << std::endl;
    m_stream << "    " << fmt::format("dw: {:#x}", pcu.dw) << std::endl;
}

