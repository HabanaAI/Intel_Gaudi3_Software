#include "mme_code_generation.h"
#include "gaudi2/mme.h"
#include "gaudi3/mme.h"
#include "include/mme_common/mme_common_enum.h"
#include "utils.h"

#define MME_REG_OFFSET(RegBlock, field) ((size_t) & (((RegBlock*) 0)->field))

namespace MmeCommon
{
template<typename Desc, typename MmeCmd, typename RegBlock>
void MmeCodeGenerator<Desc, MmeCmd, RegBlock>::pushCmd(std::vector<MmeQmanCmd>& cmds,
                                                       const size_t offset,
                                                       const unsigned size,
                                                       const void* const value)
{
    MmeQmanCmd cmd;
    cmd.cmd_type = MME_REG_WRITE;
    cmd.reg_values.insert(cmd.reg_values.begin(), (uint32_t*) value, ((uint32_t*) value) + size);
    cmd.reg_offset = (unsigned) offset;
    cmd.fence_idx = 0;
    cmd.fence_value = 0;
    cmds.push_back(cmd);
}

template<typename Desc, typename MmeCmd, typename RegBlock>
void MmeCodeGenerator<Desc, MmeCmd, RegBlock>::range2cmds(const Desc* desc,
                                                          int start,
                                                          int end,
                                                          std::vector<MmeQmanCmd>& cmds)
{
    if (m_dmaDesc)
    {
        range2cmdsDma(desc, start, end, cmds);
    }
    else
    {
        range2cmdsMme(desc, start, end, cmds);
    }
}

template<typename Desc, typename MmeCmd, typename RegBlock>
void MmeCodeGenerator<Desc, MmeCmd, RegBlock>::range2cmdsMme(const Desc* desc,
                                                             int start,
                                                             int end,
                                                             std::vector<MmeQmanCmd>& cmds)
{
    const uint32_t* dwDesc = (const uint32_t*) desc;

    if (start % 2)
    {
        pushCmd(cmds, MME_REG_OFFSET(RegBlock, descWrap.desc) + (start * sizeof(uint32_t)), 1, &dwDesc[start]);
        start++;
    }

    pushCmd(cmds,
            MME_REG_OFFSET(RegBlock, descWrap.desc) + (start * sizeof(uint32_t)),
            (end & ~1) - start,
            &dwDesc[start]);

    if (end % 2)
    {
        pushCmd(cmds, MME_REG_OFFSET(RegBlock, descWrap.desc) + ((end - 1) * sizeof(uint32_t)), 1, &dwDesc[end - 1]);
    }
}

template<typename Desc, typename MmeCmd, typename RegBlock>
void MmeCodeGenerator<Desc, MmeCmd, RegBlock>::range2cmdsDma(const Desc* desc,
                                                             int start,
                                                             int end,
                                                             std::vector<MmeQmanCmd>& cmds)
{
    const uint32_t* dwDesc = (const uint32_t*) desc;

    if (start % 2)
    {
        pushCmd(cmds, MME_REG_OFFSET(RegBlock, descWrap.dmaDesc) + (start * sizeof(uint32_t)), 1, &dwDesc[start]);
        start++;
    }

    pushCmd(cmds,
            MME_REG_OFFSET(RegBlock, descWrap.dmaDesc) + (start * sizeof(uint32_t)),
            (end & ~1) - start,
            &dwDesc[start]);

    if (end % 2)
    {
        pushCmd(cmds, MME_REG_OFFSET(RegBlock, descWrap.dmaDesc) + ((end - 1) * sizeof(uint32_t)), 1, &dwDesc[end - 1]);
    }
}

template<>
void MmeCodeGenerator<Gaudi2::Mme::Desc, Gaudi2::Mme::MmeCmd, Gaudi2::Mme::RegBlock>::range2cmdsDma(
    const Gaudi2::Mme::Desc* desc,
    int start,
    int end,
    std::vector<MmeQmanCmd>& cmds)
{
    MME_ASSERT(0, "Gaudi2 doesnt support dma Desc");
}

template<typename Desc, typename MmeCmd, typename RegBlock>
void MmeCodeGenerator<Desc, MmeCmd, RegBlock>::desc2cmds(const bool fullDesc,
                                                         const bool mask[sizeof(Desc)],
                                                         const Desc* currDesc,
                                                         const Desc* prevDesc,
                                                         std::vector<MmeQmanCmd>& cmds,
                                                         Desc* deviceDesc)
{
    const unsigned descSizeInDW = sizeof(Desc) / sizeof(uint32_t);
    if (prevDesc && !fullDesc)
    {
        // build a copy of the target descriptor. it is comprised of all the parts of the
        // new descriptor we need to write (according to validMask)
        // and parts of the previous descriptor we dont care about leaving behind.
        for (unsigned i = 0; i < sizeof(Desc); i++)
        {
            ((char*) deviceDesc)[i] = mask[i] ? ((char*) currDesc)[i] : ((char*) prevDesc)[i];
        }

        // compare the target descriptor with the previous descriptor, add cmds to write over any difference.
        int start = -1;
        for (int i = 0; i < descSizeInDW; i++)
        {
            bool diff = ((uint32_t*) deviceDesc)[i] != ((uint32_t*) prevDesc)[i];
            if (diff != (start >= 0))
            {
                if (diff)
                {
                    start = i;
                }
                else
                {
                    range2cmds(currDesc, start, i, cmds);
                    start = -1;
                }
            }
        }

        if (start >= 0)
        {
            range2cmds(currDesc, start, descSizeInDW, cmds);
        }
    }
    else
    {
        if (deviceDesc)
        {
            memcpy(deviceDesc, currDesc, sizeof(Desc));
        }

        range2cmds(currDesc, 0, descSizeInDW, cmds);
    }
}

template<typename Desc, typename MmeCmd, typename RegBlock>
unsigned MmeCodeGenerator<Desc, MmeCmd, RegBlock>::getWkldID(const unsigned wkldID, const unsigned descIdx)
{
    const unsigned c_desc_idx_bits = 8;
    const unsigned c_desc_idx_mask = (1 << c_desc_idx_bits) - 1;
    unsigned ret = (wkldID << c_desc_idx_bits) + (descIdx & c_desc_idx_mask);
    return ret;
}

template<typename Desc, typename MmeCmd, typename RegBlock>
void MmeCodeGenerator<Desc, MmeCmd, RegBlock>::logMemAccess(const MmeCmd cmd,
                                                            unsigned mmeIdx,
                                                            const MmeActivation<Desc>& activation,
                                                            MmeMemAccessChecker* accessChecker)
{
    if (accessChecker)
    {
        OverlapDescriptor overlapDesc;
        overlapDesc.engineID = mmeIdx;
        overlapDesc.engineIDForDepCtx = overlapDesc.engineID;
        overlapDesc.numSignals = activation.numSignals;

        const OverlapRoi* a = nullptr;
        const OverlapRoi* b = nullptr;
        const OverlapRoi* c = nullptr;

        if (activation.isGemm ||
            (activation.getDesc(mmeIdx).header.transA && !activation.getDesc(mmeIdx).header.transB))  // Bgemm & FWD
        {
            a = &activation.roiX;
            b = &activation.roiW;
            c = &activation.roiY;
        }
        else if (activation.getDesc(mmeIdx).header.transA && activation.getDesc(mmeIdx).header.transB)  // DEDX
        {
            a = &activation.roiY;
            b = &activation.roiW;
            c = &activation.roiX;
        }
        else if (!activation.getDesc(mmeIdx).header.transA && !activation.getDesc(mmeIdx).header.transB)  // DEDW
        {
            a = &activation.roiX;
            b = &activation.roiY;
            c = &activation.roiW;
        }
        else
        {
            MME_ASSERT(0, "should not get here");
        }

        if (doesCmdExecutesOperand(&activation.getDesc(mmeIdx), cmd, true /* Operand A */))
        {
            overlapDesc.inputRois.push_back(*a);
        }

        if (doesCmdExecutesOperand(&activation.getDesc(mmeIdx), cmd, false /* Operand B */))
        {
            overlapDesc.inputRois.push_back(*b);
        }

        if (cmd.aguOut != 0)
        {
            overlapDesc.outputRois.push_back(*c);
            overlapDesc.outputRois.push_back(activation.roiO);
        }
        unsigned descColor =
            activation.getDesc(mmeIdx).header.storeColorSet0;  // if color0 set -> value 0, if color1 set -> value 1
        accessChecker->addDesc(overlapDesc, descColor, false);
    }
}

template<typename MmeCmd>
inline constexpr bool supportsDma()
{
    return std::is_same<gaudi3::Mme::MmeCmd, MmeCmd>::value;
}

template<typename Desc, typename MmeCmd, typename RegBlock>
void MmeCodeGenerator<Desc, MmeCmd, RegBlock>::buildCmds(const MmeTestParams& testParams,
                                                         std::vector<MmeQmanCmd> cmds[],
                                                         MmeDataParams& dataParams,
                                                         bool& firstValidTest,
                                                         MmeMemoryUsage testMemUsage)
{
    MmeDescriptorGenerator<Desc>* descGenerator = dynamic_cast<MmeDescriptorGenerator<Desc>*>(m_descGenerator.get());
    // not supported yet
    MME_ASSERT(!testParams.maskSignals, "Need to be implemented");
    MME_ASSERT(!descGenerator->getMmeActivations().empty(), "empty activation list !");

    unsigned fenceValue = 0;
    unsigned descCtr = 0;

    MmeQmanCmd fenceCmd;
    fenceCmd.cmd_type = MME_FENCE;
    fenceCmd.fence_idx = getFenceIdxForFenceWaitCmds();
    fenceCmd.fence_value = fenceValue;
    fenceValue = 0;

    Desc localDescs[m_mmeNr][2];  // double buffering for desc2cmds
    Desc* localPrevDesc[m_mmeNr];
    if (firstValidTest)
    {
        for (unsigned mmeIdx = 0; mmeIdx < m_mmeNr; ++mmeIdx)
        {
            localPrevDesc[mmeIdx] = nullptr;
        }
        firstValidTest = false;
    }
    else
    {
        for (unsigned mmeIdx = 0; mmeIdx < m_mmeNr; ++mmeIdx)
        {
            localPrevDesc[mmeIdx] = m_dmaDesc ? &m_deviceDescDma[mmeIdx] : &m_deviceDescMme[mmeIdx];
        }
    }

    ActivationVec<Desc> localActivations = descGenerator->getMmeActivations();
    if (testParams.powerTest)
    {
        // MME_ASSERT(!testParams.fullDesc, "fullDesc should be off in power test");
        MME_ASSERT(testParams.powerLoops > 0, "should have power loops");
        MME_ASSERT((testParams.powerLoops * testParams.repeats) % 2 == 0, "should even number of power loops");

        zeroSignalForPowerTest(localActivations);
    }
    bool powerIncDecReduction = testParams.powerTest;
    unsigned powerReductionCtr = 0;
    for (unsigned i = 0; i < testParams.repeats; i++)
    {
        for (unsigned loop = 0; loop < (testParams.powerTest ? testParams.powerLoops : 1); loop++)
        {
            if (powerIncDecReduction)
            {
                auto op = (powerReductionCtr % 2) ? EMmeReductionOp::e_mme_reduction_add
                                                  : EMmeReductionOp::e_mme_reduction_add;
                addReductionToPowerActivations(localActivations, op);
                powerReductionCtr++;
            }
            for (auto& activation : localActivations)
            {
                // TODO: non primary
                for (unsigned mmeIdx = 0; mmeIdx < m_mmeNr; ++mmeIdx)
                {
                    Desc& desc = activation.getDesc(mmeIdx);
                    if (loop % 2 == 1 && testParams.powerTest)
                        desc.baseAddrA.addr = desc.baseAddrA.addr + testMemUsage.hbmUsage + 0x20000 ;
                    unsigned wkldIdVal = getWkldID(dataParams.wkldId, descCtr);
                    descGenerator->patchDebugWkldID(wkldIdVal, desc);

                    bool mask[sizeof(Desc)];
                    bool aguOut1FromAguOut0_DW0, aguOut1FromAguOut0_DW1_4;
                    descGenerator->mmeGetDescValidMask(desc, mask, &aguOut1FromAguOut0_DW0, &aguOut1FromAguOut0_DW1_4);

                    unsigned& targetSoValue = dataParams.soValues[mmeIdx];
                    targetSoValue += activation.numSignals;

                    if (testParams.powerTest && !m_scalFw && descCtr == 0)
                    {
                        Desc firstDesc = desc;
                        removeReductionToPowerDesc(firstDesc);
                        desc2cmds(testParams.fullDesc,
                                  mask,
                                  &firstDesc,
                                  nullptr,
                                  cmds[mmeIdx],
                                  &localDescs[mmeIdx][descCtr % 2]);
                        localPrevDesc[mmeIdx] = &localDescs[mmeIdx][descCtr % 2];
                        cmds[mmeIdx].back().power_last_setup_cmd = true;
                    }

                    desc2cmds(testParams.fullDesc,
                              mask,
                              &desc,
                              (testParams.powerTest && !m_scalFw) ? nullptr : localPrevDesc[mmeIdx],
                              cmds[mmeIdx],
                              &localDescs[mmeIdx][descCtr % 2]);

                    localPrevDesc[mmeIdx] = &localDescs[mmeIdx][descCtr % 2];

                    if (testParams.prefetchB || testParams.prefetchA)
                    {
                        /*
                         * to prefetch an operand we will trigger the same descriptor twice.
                         * the first cmd will turn on only the AGUs that read the operand (using the aguReadsX field).
                         * and a second time, triggering all the engines except the AGUs that were triggered in the
                         * previous cmd.
                         *
                         * the first cmd will turn on copyAndInc to copy the descriptor,
                         * but because it also increments the shadow counter the next cmd needs to set descSel to -1
                         * to temporarily decrement the shadow counter, negating the previous increment.
                         */
                        unsigned aguInMask = 0;
                        if (testParams.prefetchA)
                        {
                            aguInMask = getAguInAMask(desc);
                        }
                        if (testParams.prefetchB)
                        {
                            aguInMask = getAguInBMask(desc);
                        }

                        MmeCmd cmd;
                        cmd.dw = 0;
                        cmd.aguIn = aguInMask;
                        cmd.aguOut = 0;
                        if constexpr (supportsDma<MmeCmd>())
                        {
                            cmd.aguOutDma = 0;
                            cmd.dmaDesc = m_dmaDesc;
                            cmd.nullDmaDesc = 0;
                        }
                        cmd.eu = 0;
                        cmd.ap = 0;
                        cmd.copyAndInc = 1;
                        cmd.descSel = 0;
                        cmd.maskIdleIndication = 0;
                        cmd.nullDesc = 0;
                        pushCmd(cmds[mmeIdx], MME_REG_OFFSET(RegBlock, descWrap.cmd), 1, &cmd.dw);

                        cmd.aguIn = ~aguInMask;
                        cmd.aguOut = ~0;  //  trick to set all bits to 1 regardless of field size
                        if constexpr (supportsDma<MmeCmd>())
                        {
                            cmd.aguOutDma = ~0;
                            cmd.dmaDesc = m_dmaDesc;
                        }
                        cmd.eu = 1;
                        cmd.ap = 1;
                        cmd.copyAndInc = 0;
                        cmd.descSel = 3;  // = -1
                        pushCmd(cmds[mmeIdx], MME_REG_OFFSET(RegBlock, descWrap.cmd), 1, &cmd.dw);
                    }
                    else
                    {
                        MmeCmd cmd;
                        cmd.dw = 0;
                        cmd.aguIn = ~0;  //  trick to set all bits to 1 regardless of field size
                        cmd.aguOut = ~0;  //  trick to set all bits to 1 regardless of field size
                        if constexpr (supportsDma<MmeCmd>())
                        {
                            cmd.aguOutDma = ~0;
                            cmd.dmaDesc = m_dmaDesc;
                            cmd.nullDmaDesc = 0;
                        }
                        cmd.eu = 1;
                        cmd.ap = 1;
                        cmd.copyAndInc = 1;
                        cmd.descSel = 0;
                        cmd.maskIdleIndication = 0;
                        cmd.nullDesc = 0;
                        // TODO: aguOut1FromAguOut0

                        pushCmd(cmds[mmeIdx], MME_REG_OFFSET(RegBlock, descWrap.cmd), 1, &cmd.dw);
                        logMemAccess(cmd, mmeIdx, activation, dataParams.memAccessChecker.get());
                    }
                }

                // advance double buffer to next index to avoid overiding localPrevDesc while building localDesc
                descCtr++;
            }
        }
        if (testParams.powerTest && !m_scalFw && testParams.powerIdleCycles && testParams.powerIdleLoops)
        {
            for (unsigned mmeIdx = 0; mmeIdx < m_mmeNr; ++mmeIdx)
            {
                MmeQmanCmd waitCmd;
                waitCmd.cmd_type = MME_WAIT;
                waitCmd.wait_idx = getFenceIdxForFenceWaitCmds();
                waitCmd.wait_value = 1;
                waitCmd.wait_cycles = testParams.powerIdleCycles;

                for (unsigned idleCtr = 0; idleCtr < testParams.powerIdleLoops; idleCtr++)
                {
                    cmds[mmeIdx].push_back(waitCmd);
                }

                fenceCmd.fence_value = testParams.powerIdleLoops;
                cmds[mmeIdx].push_back(fenceCmd);
                fenceCmd.fence_value = 0;
            }
        }
    }

    for (unsigned mmeIdx = 0; mmeIdx < m_mmeNr; ++mmeIdx)
    {
        if (m_dmaDesc)
        {
            memcpy(&m_deviceDescDma[mmeIdx], localPrevDesc[mmeIdx], sizeof(Desc));
        }
        else
        {
            memcpy(&m_deviceDescMme[mmeIdx], localPrevDesc[mmeIdx], sizeof(Desc));
        }
    }
}

template<typename Desc, typename MmeCmd, typename RegBlock>
void MmeCodeGenerator<Desc, MmeCmd, RegBlock>::createNullDescCmds(MmeDataParams& dataParams,
                                                                  std::vector<MmeQmanCmd> cmds[])
{
    auto& testJson = *dataParams.testJson;
    bool dualNullDesc = testJson["dualNullDesc"].get<bool>();
    // in dual nullDesc mode the MME will signal twice to the syncObject defined in the DMA descriptor.
    std::vector<Desc>& deviceDesc = m_dmaDesc ? m_deviceDescDma : m_deviceDescMme;
    MmeMemAccessChecker* accessChecker = dataParams.memAccessChecker.get();
    EMmeOperand operandC = getOutputFromOperation(testJson["operation"].get<EMmeOpType>());
    unsigned primaryOutputColor =
        (testJson["useSameColorSet"].get<bool>() || dataParams.operandInSram[operandC]) ? 0 : 1;
    unsigned secondaryOutputColor = 2 /* no signal */;
    if (testJson["secondaryOutput"].get<bool>())
    {
        secondaryOutputColor =
            (testJson["useSameColorSet"].get<bool>() || dataParams.operandInSram[e_mme_op_o]) ? 0 : 1;
    }
    bool color0Used = primaryOutputColor == 0 || secondaryOutputColor == 0;
    bool color1Used = primaryOutputColor == 1 || secondaryOutputColor == 1;

    for (unsigned mmeIdx = 0; mmeIdx < m_mmeNr; ++mmeIdx)
    {
        unsigned numSignals = dualNullDesc ? 2 : 1;
        // nullDesc will signal to same SO as the main test
        unsigned& targetSoValue = dataParams.soValues[mmeIdx];
        uint32_t targetSoAddr = dataParams.syncObjects[mmeIdx].Primary.second;
        uint32_t targetSecondarySoAddr = dataParams.syncObjects[mmeIdx].Secondary.second;
        uint32_t targetSlaveSoAddr = dataParams.syncObjects[mmeIdx].PrimarySlave.second;
        uint32_t targetSlaveSecondarySoAddr = dataParams.syncObjects[mmeIdx].SecondarySlave.second;

        // zero bit field
        deviceDesc[mmeIdx].syncObject.dw[0] = 0;
        // configure syncObject for nullDesc.
        deviceDesc[mmeIdx].syncObject.signalEn0 = color0Used;
        deviceDesc[mmeIdx].syncObject.signalEn1 = color1Used;

        unsigned slaveColor0Addr = 0, slaveColor1Addr = 0;
        if (color0Used)
        {
            deviceDesc[mmeIdx].syncObject.so0Addr = primaryOutputColor == 0 ? targetSoAddr : targetSecondarySoAddr;
            slaveColor0Addr = primaryOutputColor == 0 ? targetSlaveSoAddr : targetSlaveSecondarySoAddr;
        }
        if (color1Used)
        {
            deviceDesc[mmeIdx].syncObject.so1Addr = primaryOutputColor == 1 ? targetSoAddr : targetSecondarySoAddr;
            slaveColor1Addr = primaryOutputColor == 1 ? targetSlaveSoAddr : targetSlaveSecondarySoAddr;
        }

        deviceDesc[mmeIdx].syncObject.slaveSignalEn = targetSlaveSoAddr ? 1 : 0;
        deviceDesc[mmeIdx].syncObject.slave0UseSlaveSOAddr = slaveColor0Addr ? 1 : 0;
        deviceDesc[mmeIdx].syncObject.slave1UseSlaveSOAddr = slaveColor1Addr ? 1 : 0;
        if constexpr (std::is_same<gaudi3::Mme::Desc, Desc>::value)
        {
            deviceDesc[mmeIdx].syncObject.slaveSo0Addr = slaveColor0Addr;
            deviceDesc[mmeIdx].syncObject.slaveSo1Addr = slaveColor1Addr;
        }
        else if constexpr (std::is_same<Gaudi2::Mme::Desc, Desc>::value)
        {
            deviceDesc[mmeIdx].slaveSyncObject0Addr = slaveColor0Addr;
            deviceDesc[mmeIdx].slaveSyncObject1Addr = slaveColor1Addr;
        }
        else
        {
            MME_ASSERT(0, "nullDesc not supported for chip yet");
        }

        deviceDesc[mmeIdx].syncObject.so0Val.soValue = 1;
        deviceDesc[mmeIdx].syncObject.so0Val.soPerfEn = 0;
        deviceDesc[mmeIdx].syncObject.so0Val.soReserved = 0;
        deviceDesc[mmeIdx].syncObject.so0Val.soOp = 1;
        deviceDesc[mmeIdx].syncObject.so1Val.soValue = 1;
        deviceDesc[mmeIdx].syncObject.so1Val.soReserved = 0;
        deviceDesc[mmeIdx].syncObject.so1Val.soPerfEn = 0;
        deviceDesc[mmeIdx].syncObject.so1Val.soOp = 1;
        if (targetSlaveSoAddr)
        {
            numSignals *= 2;
            deviceDesc[mmeIdx].syncObject.masterWaitForSlaveFence = 0;
            deviceDesc[mmeIdx].syncObject.slaveSendFence2Master = 0;
        }
        else
        {
            deviceDesc[mmeIdx].syncObject.masterWaitForSlaveFence = 1;
            deviceDesc[mmeIdx].syncObject.slaveSendFence2Master = 1;
            deviceDesc[mmeIdx].syncObject.slave0UseSlaveSOAddr = 0;
            deviceDesc[mmeIdx].syncObject.slave1UseSlaveSOAddr = 0;
        }
        MME_ASSERT(deviceDesc[mmeIdx].syncObject.dw[2] == 0x80000001,
                   "Null desc sync object is not in the correct value");
        MME_ASSERT(deviceDesc[mmeIdx].syncObject.dw[4] == 0x80000001,
                   "Null desc sync object is not in the correct value");

        unsigned start =
            (MME_REG_OFFSET(RegBlock, descWrap.desc.syncObject.dw[0]) - MME_REG_OFFSET(RegBlock, descWrap.desc)) / 4;
        unsigned end = (sizeof(Desc::syncObject)) / 4 + 1;
        range2cmds(&deviceDesc[mmeIdx], start, start + end, cmds[mmeIdx]);
        //  in Gaudi2 we also have to specifically write the slave address since it is outside the syncObject struct
        if constexpr (std::is_same<Gaudi2::Mme::Desc, Desc>::value)
        {
            if (targetSlaveSoAddr)
            {
                start = (MME_REG_OFFSET(Gaudi2::Mme::RegBlock, descWrap.desc.slaveSyncObject0Addr) -
                         MME_REG_OFFSET(Gaudi2::Mme::RegBlock, descWrap.desc)) /
                        4;
                end = 2;  // two Dwords - slaveSyncObject0Addr/slaveSyncObject1Addr
                range2cmds(&deviceDesc[mmeIdx], start, start + end, cmds[mmeIdx]);
            }
        }

        MmeCmd cmd;
        cmd.dw = 0;
        cmd.aguIn = ~0;
        cmd.aguOut = ~0;
        if constexpr (supportsDma<MmeCmd>())
        {
            cmd.aguOutDma = ~0;
            cmd.dmaDesc = m_dmaDesc;
            cmd.nullDmaDesc = m_dmaDesc || dualNullDesc;
        }
        cmd.eu = 1;
        cmd.ap = 1;
        cmd.copyAndInc = 1;
        cmd.descSel = 0;
        cmd.maskIdleIndication = 0;
        cmd.nullDesc = !m_dmaDesc || dualNullDesc;
        pushCmd(cmds[mmeIdx], MME_REG_OFFSET(RegBlock, descWrap.cmd), 1, &cmd.dw);

        if (accessChecker)
        {
            OverlapDescriptor overlapDesc;
            overlapDesc.engineID = mmeIdx;
            overlapDesc.engineIDForDepCtx = overlapDesc.engineID;
            overlapDesc.numSignals = numSignals;
            accessChecker->addDesc(overlapDesc, primaryOutputColor, true);
        }
        targetSoValue += numSignals;
    }
}

template<typename Desc, typename MmeCmd, typename RegBlock>
void MmeCodeGenerator<Desc, MmeCmd, RegBlock>::addReductionToPowerActivations(ActivationVec<Desc>& powerActivations,
                                                                              MmeCommon::EMmeReductionOp op)
{
    for (auto& act : powerActivations)
    {
        for (auto& desc : act.descriptors)
        {
            addReductionToPowerDesc(desc, op);
        }
    }
}

template class MmeCodeGenerator<Gaudi2::Mme::Desc, Gaudi2::Mme::MmeCmd, Gaudi2::Mme::RegBlock>;
template class MmeCodeGenerator<gaudi3::Mme::Desc, gaudi3::Mme::MmeCmd, gaudi3::Mme::RegBlock>;
}  // namespace MmeCommon
