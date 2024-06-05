#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <string>
#include <variant>

#include "scal_internal/pkt_macros.hpp"
#include "scal_test_utils.h"
#include "logger.h"
#include "scal_internal/sched_pkts.hpp"
#include "scal_internal/eng_pkts.hpp"
#include "scal_test_pqm_pkt_utils.h"

#include "gaudi3/asic_reg/gaudi3_blocks.h" // arc_acp_eng base-address

std::variant<G2Packets , G3Packets> createBuildPkts(scalDeviceType device)
{
    if (device == scalDeviceType::dtGaudi3)
        return G3Packets ();
    else
        return G2Packets ();
}

inline bool file_exists(const std::string& name)
{
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

bool str_replace(std::string& str, const std::string& from, const std::string& to)
{
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

std::string getConfigFilePath(const char* configFileName)
{
    std::vector<std::string> jsonSearchPaths;
    const char* envConfVarValue = getenv("SCAL_TEST_CFG_PATH");
    if (envConfVarValue)
    {
        jsonSearchPaths.push_back(envConfVarValue);
        configFileName = envConfVarValue;
    }
    std::string filename = configFileName;
    if (filename.find(internalFileSignature) == 0 || filename.empty()) return filename;

    const char* envConfDirVarValue = getenv("SCAL_TEST_CFG_DIR_PATH");
    std::string configFileFullPath;
    if (envConfDirVarValue)
    {
        configFileFullPath = envConfDirVarValue + ("/" + filename);
    }
    const char* subDirs[] = {"", "configs", "tests/configs", "configs_internal"};
    for (const auto& subDir : subDirs)
    {
        std::string configFileFullPathTests = configFileFullPath;
        str_replace(configFileFullPathTests, "configs", subDir);
        jsonSearchPaths.push_back(configFileFullPathTests);
    }
    for (const auto& jsonSearchPath : jsonSearchPaths)
    {
        if(file_exists(jsonSearchPath))
        {
            return jsonSearchPath;
        }
    }
    return "";
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SchedCmd::Init(char *cmdBufferBase, unsigned cmdBufferSize, unsigned stream_dccm_buf_size, scalDeviceType device)
{
    m_cmdBufferBase = cmdBufferBase;
    m_cmdBufferSize = cmdBufferSize;
    m_offset = 0;
    m_pi3 = 0;
    m_streamDccmBufSize = stream_dccm_buf_size;
    m_canReset = false;
    deviceType = device;
    m_buildPkt = createBuildPkts(device);
}

void SchedCmd::setPi(uint64_t pi)
{
    m_pi3 = pi;
}
void SchedCmd::PadReset()
{
    // pad the command buffer till the end
    // and start using the buffer from offset 0 again
    unsigned  left = m_cmdBufferSize - m_offset;
    m_pi3 += left;

    fillPktNoSize<NopCmdPkt>(m_buildPkt, m_cmdBufferBase+m_offset, left);
    m_offset = 0;
}


unsigned SchedCmd::CheckCmdWrap(unsigned newCmdSize)
{
    //  scheduler commands cannot wrap on the scheduler internal dccm buffer
    //    the dccm size is defined in the config (json) and can be obtained by:
    //   rc = scal_stream_get_info(m_computeStreamHandle, &m_streamInfo);
    //   stream_cmd_buffer_dccm_size = m_streamInfo.command_alignment
    //
    //   so if our host cmd buffer is 2K and dccm buffer is 512 bytes
    //     our command cannot cross  512,1024,1536,2048
    unsigned currQuarter = m_offset / m_streamDccmBufSize;
    unsigned newQuarter = (m_offset+newCmdSize-1) / m_streamDccmBufSize;

    if(newQuarter != currQuarter)
        return (m_streamDccmBufSize-(m_offset%m_streamDccmBufSize)); // how much padding needed
    return 0;// OK
}

bool SchedCmd::Check(unsigned newCmdSize, bool isDirectMode)
{
    if (isDirectMode)
    {
        if (m_offset + newCmdSize > m_cmdBufferSize)
        {
            unsigned padding = m_cmdBufferSize - m_offset;
            assert (padding % PqmPktUtils::getNopCmdSize() == 0);
            int nopCmdAmount = padding / PqmPktUtils::getNopCmdSize();
            for (int i = 0; i < nopCmdAmount; i++)
            {
                PqmPktUtils::buildNopCmd(m_cmdBufferBase + m_offset);
                m_offset += PqmPktUtils::getNopCmdSize();
            }
            m_offset = 0;
            m_pi3 += padding;
        }
    }
    else
    {
        if (m_offset + newCmdSize >= m_cmdBufferSize)
        {
            if (m_canReset)
                PadReset();// will set m_offset to 0
            else
                return false;
        }

        unsigned padding = CheckCmdWrap(newCmdSize);
        if(padding != 0)
        {
            fillPktNoSize<NopCmdPkt>(m_buildPkt, m_cmdBufferBase + m_offset, padding);
            m_offset += padding;
            m_pi3 += padding;
        }
    }

    m_pi3 += newCmdSize;
    return true;
}


bool SchedCmd::NopCmd(uint32_t padding)
{
    unsigned newCmdSize = getPktSize<NopCmdPkt>(m_buildPkt) + padding * sizeof(uint32_t);// padding is in DWORDS;

    if (m_offset + newCmdSize > m_cmdBufferSize)
        return false;

    fillPktNoSize<NopCmdPkt>(m_buildPkt, m_cmdBufferBase+m_offset, newCmdSize);
    m_offset += newCmdSize;
    m_pi3 += newCmdSize;
    return true;
}

bool SchedCmd::FenceCmd(uint32_t fence_id, uint32_t target, bool forceFwFence, bool isDirectMode)
{
    unsigned newCmdSize = 0;

    if (isDirectMode)
    {
        newCmdSize = PqmPktUtils::getFenceCmdSize();
        if (!Check(newCmdSize, true))
            return false;
        PqmPktUtils::buildPqmFenceCmd((uint8_t*)m_cmdBufferBase + m_offset, fence_id, 1, target);
    }
    else if ((deviceType != dtGaudi2) && (!forceFwFence))
    {
        newCmdSize = getPktSize<AcpFenceWaitPkt>(m_buildPkt);
        if (!Check(newCmdSize))
            return false;
        fillPkt<AcpFenceWaitPkt>(m_buildPkt, m_cmdBufferBase+m_offset, fence_id, target);
    }
    else
    {
        newCmdSize = getPktSize<FenceWaitPkt>(m_buildPkt);
        if (!Check(newCmdSize))
            return false;
        fillPkt<FenceWaitPkt>(m_buildPkt, m_cmdBufferBase+m_offset, fence_id, target);
    }

    m_offset += newCmdSize;

    return true;
}

bool SchedCmd::PdmaTransferCmd(bool isDirectMode, uint64_t dst, uint64_t src, uint32_t size, uint32_t engine_group_type,
                               int32_t workload_type, uint32_t payload, uint64_t pay_addr, uint32_t signal_to_cg,
                               uint32_t compGrpIdx, bool wr_comp, scal_completion_group_infoV2_t* cgInfo)
{
    // calculate size of a batch-pdma command, with a batch size of 1
    unsigned newCmdSize = PqmPktUtils::getPdmaCmdSize(isDirectMode, m_buildPkt, wr_comp, 1);
    if (!Check(newCmdSize, isDirectMode))
        return false;

    uint64_t    longSoSmIndex  = 0;
    unsigned    longSoIndex    = 0;
    if (isDirectMode && cgInfo != nullptr)
    {
        longSoSmIndex  = cgInfo->long_so_sm;
        longSoIndex    = cgInfo->long_so_index;
    }

    PqmPktUtils::sendPdmaCommand(isDirectMode, m_buildPkt, m_cmdBufferBase + m_offset, src, dst, size,
        engine_group_type, workload_type, 0/*ctxId*/, payload, pay_addr/*payloadAddr*/, 0/*bMemset*/, signal_to_cg/*signal_to_cg*/,
        wr_comp, compGrpIdx/*completionGroupIndex*/, longSoSmIndex, longSoIndex);

    m_offset += newCmdSize;

    return true;
}

bool SchedCmd::AllocAndDispatchBarrier(uint32_t comp_group_index, uint32_t target_value, uint32_t rel_so_set,uint32_t num_engine_group_type, uint8_t *engine_group_type)
{
    bool ret = AllocBarrier(comp_group_index,target_value, rel_so_set);
    if (ret) ret = DispatchBarrier(num_engine_group_type, engine_group_type);
    return ret;
}

bool SchedCmd::AllocBarrier(uint32_t comp_group_index, uint32_t target_value, uint32_t rel_so_set)
{
    unsigned newCmdSize = getPktSize<AllocBarrierV2bPkt>(m_buildPkt);
    if (!Check(newCmdSize))
        return false;
    const EngineGroupArrayType engineGroupType {};
    fillPkt<AllocBarrierV2bPkt>(m_buildPkt, m_cmdBufferBase+m_offset, comp_group_index, target_value, false, (rel_so_set != 0), 0, engineGroupType, 0, 0);
    m_offset += newCmdSize;
    return true;
}


bool SchedCmd::DispatchBarrier(uint32_t num_engine_group_type, uint8_t *engine_group_type)
{
unsigned newCmdSize = getPktSize<DispatchBarrierPkt>(m_buildPkt);
    if (!Check(newCmdSize))
        return false;
    fillPkt<DispatchBarrierPkt>(m_buildPkt, m_cmdBufferBase+m_offset, num_engine_group_type, engine_group_type, 0);
    m_offset += newCmdSize;
    return true;
}
bool SchedCmd::DispatchComputeEcbList(uint32_t engine_group_type,
                               bool single_static_chunk,
                               bool single_dynamic_chunk,
                               uint32_t static_ecb_list_offset,
                               uint32_t dynamic_ecb_list_addr)
{
unsigned newCmdSize = getPktSize<DispatchComputeEcbListPkt>(m_buildPkt);
    if (!Check(newCmdSize))
        return false;

    fillScalPkt<DispatchComputeEcbListPkt>(m_buildPkt, m_cmdBufferBase+m_offset, engine_group_type,
                     single_static_chunk, single_dynamic_chunk,
                     static_ecb_list_offset, dynamic_ecb_list_addr);
    m_offset += newCmdSize;
    return true;
}

bool SchedCmd::UpdateRecipeBase(const uint16_t *recipe_base_indexes,
                               uint32_t num_engine_group_type,
                               uint8_t* engine_group_type,
                               const uint64_t* recipe_base_addrsses,
                               uint32_t num_recipe_base_elements)
{
    unsigned newCmdSize = getPktSize<UpdateRecipeBaseV2Pkt>(m_buildPkt, num_recipe_base_elements);

    fillPktNoSize<UpdateRecipeBaseV2Pkt>(m_buildPkt, m_cmdBufferBase+m_offset, num_recipe_base_elements, recipe_base_addrsses, recipe_base_indexes,
                                num_engine_group_type, engine_group_type);


    if (!Check(newCmdSize))
        return false;

    m_offset += newCmdSize;
    return true;
}

bool SchedCmd::FenceIncImmediate(uint32_t fence_count, uint8_t *arr_fence_id, bool forceFwFence)
{
    if (fence_count != 1)
    {
        return false;
    }

    unsigned newCmdSize = 0;
    if ((deviceType != dtGaudi2) && (!forceFwFence))
    {
        newCmdSize = getPktSize<AcpFenceIncImmediatePkt>(m_buildPkt);
        if (!Check(newCmdSize))
            return false;
        fillPkt<AcpFenceIncImmediatePkt>(m_buildPkt, m_cmdBufferBase+m_offset, arr_fence_id[0], 1 /* target */);
    }
    else
    {
        newCmdSize = getPktSize<FenceIncImmediatePkt>(m_buildPkt);
        if (!Check(newCmdSize))
            return false;
        fillPkt<FenceIncImmediatePkt>(m_buildPkt, m_cmdBufferBase+m_offset, arr_fence_id[0]);
    }

    m_offset += newCmdSize;

    return true;
}

bool SchedCmd::lbwWrite(uint64_t dst_addr, uint32_t data, bool block_stream, bool isDirectMode)
{
    unsigned newCmdSize = isDirectMode ? PqmPktUtils::getMsgLongCmdSize() : getPktSize<LbwWritePkt>(m_buildPkt);
    if (!Check(newCmdSize, isDirectMode)) return false;
    if (isDirectMode)
    {
        PqmPktUtils::buildPqmMsgLong(m_cmdBufferBase + m_offset, data, dst_addr);
    } else
    {
        fillPkt<LbwWritePkt>(m_buildPkt, m_cmdBufferBase + m_offset, (uint32_t)dst_addr, data, block_stream);
    }
    m_offset += newCmdSize;
    return true;
}

bool SchedCmd::lbwBurstWrite(uint32_t dst_addr, uint32_t* data, bool block_stream)
{
    unsigned newCmdSize = getPktSize<LbwBurstWritePkt>(m_buildPkt);
    if (!Check(newCmdSize)) return false;
    fillPkt<LbwBurstWritePkt>(m_buildPkt, m_cmdBufferBase + m_offset, dst_addr, data, block_stream);
    m_offset += newCmdSize;
    return true;
}

bool SchedCmd::lbwRead(uint32_t dst_addr, uint32_t src_addr, uint32_t size)
{
    unsigned newCmdSize = getPktSize<LbwReadPkt>(m_buildPkt);
    if (!Check(newCmdSize)) return false;
    fillPkt<LbwReadPkt>(m_buildPkt, m_cmdBufferBase + m_offset, dst_addr, src_addr, size);
    m_offset += newCmdSize;
    return true;
}

bool SchedCmd::memFence(bool arc, bool dup_eng, bool arc_dma)
{
    unsigned newCmdSize = getPktSize<MemFencePkt>(m_buildPkt);
    if (!Check(newCmdSize)) return false;
    fillPkt<MemFencePkt>(m_buildPkt, m_cmdBufferBase + m_offset, arc, dup_eng, arc_dma);
    m_offset += newCmdSize;
    return true;
}

uint64_t SchedCmd::getArcAcpEngBaseAddr(unsigned smIndex)
{
    uint64_t smBase = 0;

    switch (smIndex)
    {
        case 0:
            smBase = mmHD0_ARC_FARM_ARC0_ACP_ENG_BASE;
            break;
        case 1:
            smBase = mmHD0_ARC_FARM_ARC1_ACP_ENG_BASE;
            break;

        case 2:
            smBase = mmHD1_ARC_FARM_ARC0_ACP_ENG_BASE;
            break;
        case 3:
            smBase = mmHD1_ARC_FARM_ARC1_ACP_ENG_BASE;
            break;

        case 4:
            smBase = mmHD2_ARC_FARM_ARC0_ACP_ENG_BASE;
            break;
        case 5:
            smBase = mmHD2_ARC_FARM_ARC1_ACP_ENG_BASE;
            break;

        case 6:
            smBase = mmHD3_ARC_FARM_ARC0_ACP_ENG_BASE;
            break;
        case 7:
            smBase = mmHD3_ARC_FARM_ARC1_ACP_ENG_BASE;
            break;

        case 8:
            smBase = mmHD4_ARC_FARM_ARC0_ACP_ENG_BASE;
            break;
        case 9:
            smBase = mmHD4_ARC_FARM_ARC1_ACP_ENG_BASE;
            break;

        case 10:
            smBase = mmHD5_ARC_FARM_ARC0_ACP_ENG_BASE;
            break;
        case 11:
            smBase = mmHD5_ARC_FARM_ARC1_ACP_ENG_BASE;
            break;

        case 12:
            smBase = mmHD6_ARC_FARM_ARC0_ACP_ENG_BASE;
            break;
        case 13:
            smBase = mmHD6_ARC_FARM_ARC1_ACP_ENG_BASE;
            break;

        case 14:
            smBase = mmHD7_ARC_FARM_ARC0_ACP_ENG_BASE;
            break;
        case 15:
            smBase = mmHD7_ARC_FARM_ARC1_ACP_ENG_BASE;
            break;

        default:
            assert(0);
    }

    return smBase;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
EcbListCmd::EcbListCmd(char *bufferBase, unsigned bufferSize, scalDeviceType device)
{
    m_bufferBase = bufferBase;
    m_bufferSize = bufferSize;
    m_offset = 0;
    m_buildPkt = createBuildPkts(device);
    m_streamDccmBufSize = STATIC_COMPUTE_ECB_LIST_BUFF_SIZE;
}

bool EcbListCmd::Check(unsigned newCmdSize)
{
    if (m_offset + newCmdSize >= m_bufferSize)
    {
        Pad(m_bufferSize,false);
        return false;
    }
    unsigned padding = 0;
    unsigned offsetQ = m_offset % m_streamDccmBufSize;
    if(offsetQ + newCmdSize > m_streamDccmBufSize)
        padding = m_streamDccmBufSize - offsetQ;
    if(padding != 0)
    {
        unsigned new_padding = (padding - getPktSize<EngEcbNopPkt>(m_buildPkt))/sizeof(uint32_t);
        fillPkt<EngEcbNopPkt>(m_buildPkt, m_bufferBase+m_offset, false, 0, false, new_padding);
        m_offset += padding;
    }
    return true;
}

void EcbListCmd::Pad(uint32_t until, bool switchCQ)
{
    // pad the buffer
    if (until == 0)
        until = m_bufferSize;
    unsigned  left = until - m_offset;
    unsigned new_padding = (left - getPktSize<EngEcbNopPkt>(m_buildPkt))/sizeof(uint32_t);
    fillPkt<EngEcbNopPkt>(m_buildPkt, m_bufferBase+m_offset, false, 0, switchCQ, new_padding);
    m_offset = until;
}


bool EcbListCmd::NopCmd(uint32_t padding, bool switchCQ, bool yield, uint32_t dma_completion)
{
    unsigned newCmdSize = getPktSize<EngEcbNopPkt>(m_buildPkt) + padding * sizeof(uint32_t);// padding is in DWORDS
    if (m_offset + newCmdSize > m_bufferSize)
        return false;
    fillPkt<EngEcbNopPkt>(m_buildPkt, m_bufferBase+m_offset, yield, dma_completion, switchCQ, padding);
    m_offset += newCmdSize;
    return true;
}

bool EcbListCmd::ArcListSizeCmd(uint32_t list_size, bool yield, uint32_t dma_completion)
{
    unsigned newCmdSize = getPktSize<EngEcbSizePkt>(m_buildPkt);

    if (!Check(newCmdSize))
        return false;
    fillPkt<EngEcbSizePkt>(m_buildPkt, m_bufferBase+m_offset, yield, dma_completion, 0, list_size);
    m_offset += newCmdSize;
    return true;
}

bool EcbListCmd::staticDescCmd(uint32_t cpu_index, uint32_t size, uint32_t addr_offset, uint32_t addr_index, bool yield)
{
    unsigned newCmdSize = getPktSize<EngStaticDescPkt>(m_buildPkt);
    if (!Check(newCmdSize))
        return false;
    fillPkt<EngStaticDescPkt>(m_buildPkt, m_bufferBase+m_offset, yield, cpu_index, size, addr_offset, addr_index);
    m_offset += newCmdSize;
    return true;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
