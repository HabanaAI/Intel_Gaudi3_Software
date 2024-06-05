#include "qman_cp_info.hpp"

#include "defenders.h"
#include "define.hpp"

#include "synapse_runtime_logging.h"

#include "runtime/qman/common/parser/sync_manager_info.hpp"

#include "gaudi/gaudi_packets.h"

#include <sstream>

using eQmanType           = common::eQmanType;
using MonitorStateMachine = common::MonitorStateMachine;
using SyncManagerInfo     = common::SyncManagerInfo;

using namespace gaudi;

static const uint16_t MINIMAL_GAUDI_PACKET_SIZE = 2;

const std::string QmanCpInfo::m_packetsName[] = {
    "NA",               // 0x0
    "WREG-32",          // 0x1
    "WREG-Bulk",        // 0x2
    "MSG-Long",         // 0x3
    "MSG-Short",        // 0x4
    "CP-DMA",           // 0x5
    "Repeat",           // 0x6
    "MSG-Prot",         // 0x7
    "Fence",            // 0x8
    "LIN-DMA",          // 0x9
    "NOP",              // 0xA
    "Stop",             // 0xB
    "ARB-Point",        // 0xC
    "Wait",             // 0xD
    "NA-2"              // 0xE
    "Load-And-Execute"  // 0xF
};

const std::array<std::string, SYNC_MNGR_LAST> QmanCpInfo::m_syncManagerInstanceDesc = {
    "East-North",
    "East-South",
    "West-North",
    "West-South",
};

bool QmanCpInfo::parseSinglePacket(common::eCpParsingState state)
{
    bool isCpDmaState = (state == common::CP_PARSING_STATE_CP_DMAS);

    // Ping-pong is verified using a different mechanism then the Monitor-SM & Fence-SM ones
    bool ignoreFenceConfig = (state == common::CP_PARSING_STATE_FENCE_SET);

    if (m_bufferSize < MINIMAL_GAUDI_PACKET_SIZE)
    {
        LOG_GCP_FAILURE("printPacket: buffer-size left 0x{:x} is smaller than minimal packet-size ({})",
                        m_bufferSize,
                        MINIMAL_GAUDI_PACKET_SIZE);

        return false;
    }

    unsigned  opcode       = ((packet_msg_short*)m_hostAddress)->opcode;
    uint32_t* pHostAddress = (uint32_t*)m_hostAddress;

    if ((isCpDmaState) && (opcode != PACKET_CP_DMA) && (opcode != PACKET_ARB_POINT))
    {
        LOG_GCP_VERBOSE("  Work-Completion content:");
    }

    bool status = true;
    switch (opcode)
    {
        case PACKET_ARB_POINT:
            status = parseArbPoint();
            break;

        case PACKET_CP_DMA:
            status = parseCpDma();
            break;

        case PACKET_LIN_DMA:
            status = parseLinDma();
            break;

        case PACKET_MSG_LONG:
            status = parseMsgLong();
            break;

        case PACKET_MSG_SHORT:
            status = parseMsgShort();
            break;

        case PACKET_MSG_PROT:
            status = parseMsgProt();
            break;

        case PACKET_WREG_32:
            status = parseWreg32();
            break;

        case PACKET_WREG_BULK:
            status = parseWregBulk();
            break;

        case PACKET_NOP:
            status = parseNop();
            break;

        case PACKET_STOP:
            status = parseStop();
            break;

        case PACKET_WAIT:
            status = parseWait();
            break;

        case PACKET_FENCE:
            status = parseFence(ignoreFenceConfig);
            break;

        case PACKET_REPEAT:
            status = parseRepeat();
            break;

        case PACKET_LOAD_AND_EXE:
            status = parseLoadAndExecute();
            break;

        default:
            LOG_GCP_FAILURE("Unsupported opcode {} (0x{:x} : [{:#x}])",
                            opcode,
                            m_hostAddress,
                            fmt::join(&pHostAddress[0], &pHostAddress[2], " "));

            return false;
    }

    packet_id pktId = (packet_id)opcode;

    // In case opcode is invalid, we will not reach here
    // We might parse a packet and then declare it as invalid
    // This way, we will have its info
    if (!isValidPacket(pktId))
    {
        return false;
    }

    // Here we are sure that it is a valid packetId
    m_currentPacketId = pktId;
    m_packetIndex++;

    return status;
}

uint16_t QmanCpInfo::parseControlBlock(bool shouldPrintPred)
{
    // Common fields are at the same location for all
    packet_wreg32* pPacket   = (packet_wreg32*)m_hostAddress;
    uint16_t       predValue = (shouldPrintPred) ? pPacket->pred : INVALID_PRED_VALUE;

#ifndef DONT_PARSE_CONTROL_BLOCK
    if (shouldPrintPred)
    {
        LOG_GCP_VERBOSE("{}    Barriers: eng {} reg {} msg {} Predicate {}",
                        getCtrlBlockIndentation(),
                        pPacket->eng_barrier,
                        pPacket->reg_barrier,
                        pPacket->msg_barrier,
                        predValue);
    }
    else
    {
        LOG_GCP_VERBOSE("{}    Barriers: eng {} reg {} msg {}",
                        getCtrlBlockIndentation(),
                        pPacket->eng_barrier,
                        pPacket->reg_barrier,
                        pPacket->msg_barrier);
    }
#endif

    return predValue;
}

bool QmanCpInfo::parseLinDma()
{
    static const uint32_t packetSize = 6;
    if (!validatePacketSize(packetSize))
    {
        return false;
    }

    packet_lin_dma* pPacket      = (packet_lin_dma*)m_hostAddress;
    uint32_t*       pHostAddress = (uint32_t*)m_hostAddress;

    LOG_GCP_VERBOSE("{}{}: Lin-DMA packet (0x{:x} : [{:#x}]: src-address 0x{:x} "
                    "context-id 0x{:x}, 0x{:x} dst-address 0x{:x} size 0x{:x}",
                    getIndentation(),
                    getPacketIndexDesc(),
                    (uint64_t)m_hostAddress,
                    fmt::join(&pHostAddress[0], &pHostAddress[7], " "),
                    pPacket->src_addr,
                    pPacket->context_id_high,
                    pPacket->context_id_low,
                    pPacket->dst_addr,
                    pPacket->tsize);

    LOG_GCP_VERBOSE("{}                write-completion {} transpose {} dtype {} mem_set {}",
                    getIndentation(),
                    pPacket->wr_comp_en,
                    pPacket->transpose,
                    pPacket->dtype,
                    pPacket->mem_set);

    LOG_GCP_VERBOSE("{}                compress {} decompress {}",
                    getIndentation(),
                    pPacket->compress,
                    pPacket->decompress);

    parseControlBlock(true);

    updateNextBuffer(packetSize);

    return true;
}

bool QmanCpInfo::parseMsgLong()
{
    static const uint32_t packetSize = 4;
    if (!validatePacketSize(packetSize))
    {
        return false;
    }

    packet_msg_long* pPacket      = (packet_msg_long*)m_hostAddress;
    uint32_t*        pHostAddress = (uint32_t*)m_hostAddress;

    m_currentPacketAddressField = pPacket->addr;
    m_currentPacketValueField   = pPacket->value;

    eSyncManagerInstance sobjSyncMgrInstance    = gaudi::SYNC_MNGR_NUM;
    eSyncManagerInstance monitorSyncMgrInstance = gaudi::SYNC_MNGR_NUM;

    uint32_t syncObjectAddressOffset = 0;
    uint32_t monitorAddressOffset    = 0;

    bool status = true;

    bool isSobjAddress =
        getSyncObjectAddressOffset(m_currentPacketAddressField, syncObjectAddressOffset, sobjSyncMgrInstance);
    bool isMonitorAddress =
        getMonitorAddressOffset(m_currentPacketAddressField, monitorAddressOffset, monitorSyncMgrInstance);

    // Runtime only has recipes which configures the GC's SYNC-MGR
    if ((isSobjAddress || isMonitorAddress) && (monitorSyncMgrInstance != gaudi::SYNC_MNGR_GC) &&
        (sobjSyncMgrInstance != gaudi::SYNC_MNGR_GC))
    {
        LOG_GCP_FAILURE("{}{}: Msg-Long packet (0x{:x} : [{:#x}]): "
                        "Invalid address field 0x{:x} - {} {}",
                        getIndentation(),
                        getPacketIndexDesc(),
                        (uint64_t)m_hostAddress,
                        fmt::join(&pHostAddress[0], &pHostAddress[4], " "),
                        m_currentPacketAddressField,
                        _getSyncManagerDescription(monitorSyncMgrInstance),
                        isSobjAddress ? "SOBJ" : "Monitor");

        return false;
    }

    if (isSobjAddress)
    {
        // We only care about tThe first two Words, whch are the same in Msg-Long and Msg-Short
        status = parseSyncObjUpdate((packet_msg_short*)pPacket,
                                    syncObjectAddressOffset,
                                    m_currentPacketValueField,
                                    sobjSyncMgrInstance,
                                    false);
    }
    else if (isMonitorAddress)
    {
        // We only care about tThe first two Words, whch are the same in Msg-Long and Msg-Short
        status = parseMonitorPacket((packet_msg_short*)pPacket, monitorAddressOffset, monitorSyncMgrInstance, false);
    }
    else
    {
        LOG_GCP_VERBOSE("{}{}: Msg-Long packet (0x{:x} : [{:#x}]):"
                        " {} value 0x{:x} weakly-order {} no-snoop {} address 0x{:x}",
                        getIndentation(),
                        getPacketIndexDesc(),
                        (uint64_t)m_hostAddress,
                        fmt::join(&pHostAddress[0], &pHostAddress[4], " "),
                        ((pPacket->op) ? "write-timestamp" : "write-value"),
                        m_currentPacketValueField,
                        pPacket->weakly_ordered,
                        pPacket->no_snoop,
                        m_currentPacketAddressField);

        parseControlBlock(true);
    }

    if (!status)
    {
        return false;
    }

    updateNextBuffer(packetSize);

    return status;
}

bool QmanCpInfo::parseMsgShort()
{
    static const uint32_t packetSize = 2;
    if (!validatePacketSize(packetSize))
    {
        return false;
    }

    packet_msg_short* pPacket = (packet_msg_short*)m_hostAddress;

    m_currentPacketValueField = pPacket->value;

    uint32_t base             = pPacket->base;
    uint32_t operation        = pPacket->op;
    uint32_t msgAddressOffset = pPacket->msg_addr_offset;
    uint32_t value            = pPacket->value;

    if (pPacket->base >= 2)
    {
        parseMsgShortBasic(base, operation, msgAddressOffset, value);
    }
    else if (pPacket->base == 1)
    {
        if (!parseSyncObjUpdate(pPacket, msgAddressOffset, value, gaudi::SYNC_MNGR_GC, true))
        {
            return false;
        }
    }
    else if (pPacket->base == 0)
    {
        if (!parseMonitorPacket(pPacket, msgAddressOffset, gaudi::SYNC_MNGR_GC, true))
        {
            return false;
        }
    }

    updateNextBuffer(packetSize);

    return true;
}

bool QmanCpInfo::parseMsgProt()
{
    static const uint32_t packetSize = 4;
    if (!validatePacketSize(packetSize))
    {
        return false;
    }

    packet_msg_prot* pPacket      = (packet_msg_prot*)m_hostAddress;
    uint32_t*        pHostAddress = (uint32_t*)m_hostAddress;

    LOG_GCP_VERBOSE("{}{}: MSG-Prot packet (0x{:x} : [{:#x}]): address 0x{:x} value 0x{:x} {} weakly-order "
                    "{} no-snoop {}",
                    getIndentation(),
                    getPacketIndexDesc(),
                    (uint64_t)m_hostAddress,
                    fmt::join(&pHostAddress[0], &pHostAddress[4], " "),
                    pPacket->addr,
                    pPacket->value,
                    ((pPacket->op) ? "write-timestamp" : "write-value"),
                    pPacket->weakly_ordered,
                    pPacket->no_snoop);

    parseControlBlock(true);

    updateNextBuffer(packetSize);

    return true;
}

bool QmanCpInfo::parseWreg32()
{
    const uint32_t packetSize = 2;
    if (!validatePacketSize(packetSize))
    {
        return false;
    }

    packet_wreg32* pPacket      = (packet_wreg32*)m_hostAddress;
    uint32_t*      pHostAddress = (uint32_t*)m_hostAddress;

    LOG_GCP_VERBOSE("{}{}: WRG32 packet (0x{:x} : [{:#x}]): Reg-Offset 0x{:x} value 0x{:x}",
                    getIndentation(),
                    getPacketIndexDesc(),
                    (uint64_t)m_hostAddress,
                    fmt::join(&pHostAddress[0], &pHostAddress[2], " "),
                    pPacket->reg_offset,
                    pPacket->value);

    parseControlBlock(true);

    updateNextBuffer(packetSize);

    return true;
}

bool QmanCpInfo::parseWregBulk()
{
    packet_wreg_bulk* pPacket = (packet_wreg_bulk*)m_hostAddress;

    uint32_t packetSize = 2 + 2 * pPacket->size64;
    if (!validatePacketSize(packetSize))
    {
        return false;
    }

    uint32_t* pHostAddress = (uint32_t*)m_hostAddress;

    uint64_t currRegOffset = pPacket->reg_offset;

    LOG_GCP_VERBOSE("{}{}: WBulk packet (0x{:x} : [{:#x}]): Size 0x{:x} Reg-Offset 0x{:x}",
                    getIndentation(),
                    getPacketIndexDesc(),
                    (uint64_t)m_hostAddress,
                    fmt::join(&pHostAddress[0], &pHostAddress[2], " "),
                    pPacket->size64,
                    currRegOffset);

    const unsigned numOfBulkFieldsPerLine  = 4;
    const unsigned numOfFieldsPerBulkField = 2;
    const unsigned totalNumOfFieldsPerLine = numOfBulkFieldsPerLine * numOfFieldsPerBulkField;
    uint32_t*      pValues                 = (uint32_t*)pPacket->values;
    uint64_t       quadAmount              = pPacket->size64 / numOfBulkFieldsPerLine;
    for (uint64_t i = 0, valueIndex = 0; i < quadAmount; i++)
    {
        LOG_GCP_VERBOSE("{}   currRegOffset 0x{:x} (+ 0x{:x}): Value [{:#x}]",
                        getIndentation(),
                        currRegOffset,
                        valueIndex,
                        fmt::join(&pValues[0], &pValues[8], " "));

        pValues += totalNumOfFieldsPerLine;
        currRegOffset += totalNumOfFieldsPerLine * sizeof(uint32_t);
        valueIndex += numOfBulkFieldsPerLine;
    }

    const unsigned numOfReminderFieldsPerLine = 2;
    uint64_t       quadReminder               = pPacket->size64 % numOfBulkFieldsPerLine;
    for (uint64_t i = 0, valueIndex = 0; i < quadReminder; i++, valueIndex++)
    {
        LOG_GCP_VERBOSE("{}   currRegOffset 0x{:x}: Value [{:#x}]",
                        getIndentation(),
                        currRegOffset,
                        fmt::join(&pValues[0], &pValues[2], " "));
        pValues += numOfReminderFieldsPerLine;
        currRegOffset += numOfReminderFieldsPerLine * sizeof(uint32_t);
    }

    parseControlBlock(true);

    updateNextBuffer(packetSize);

    return true;
}

bool QmanCpInfo::parseNop()
{
    const uint32_t packetSize = 2;
    if (!validatePacketSize(packetSize))
    {
        return false;
    }

    uint32_t* pHostAddress = (uint32_t*)m_hostAddress;

    LOG_GCP_VERBOSE("{}{}: NOP packet (0x{:x} : [{:#x}]):",
                    getIndentation(),
                    getPacketIndexDesc(),
                    (uint64_t)m_hostAddress,
                    fmt::join(&pHostAddress[0], &pHostAddress[2], " "));

    parseControlBlock(false);

    updateNextBuffer(packetSize);

    return true;
}

bool QmanCpInfo::parseStop()
{
    const uint32_t packetSize = 2;
    if (!validatePacketSize(packetSize))
    {
        return false;
    }

    uint32_t* pHostAddress = (uint32_t*)m_hostAddress;

    LOG_GCP_VERBOSE("{}{}: Stop packet (0x{:x} : [{:#x}]):",
                    getIndentation(),
                    getPacketIndexDesc(),
                    (uint64_t)m_hostAddress,
                    fmt::join(&pHostAddress[0], &pHostAddress[2], " "));

    parseControlBlock(false);

    updateNextBuffer(packetSize);

    return true;
}

bool QmanCpInfo::parseFence(bool ignoreFenceConfig)
{
    const uint32_t packetSize = 2;
    if (!validatePacketSize(packetSize))
    {
        return false;
    }

    packet_fence* pPacket      = (packet_fence*)m_hostAddress;
    uint32_t*     pHostAddress = (uint32_t*)m_hostAddress;
    uint32_t      fenceIndex   = m_cpIndex * FENCE_IDS_PER_CP + pPacket->id;

    LOG_GCP_VERBOSE("{}{}: Fence packet (0x{:x} : [{:#x}]): Fence-Index {} Dec-Val {} Target-Val {}",
                    getIndentation(),
                    getPacketIndexDesc(),
                    (uint64_t)m_hostAddress,
                    fmt::join(&pHostAddress[0], &pHostAddress[2], " "),
                    fenceIndex,
                    pPacket->dec_val,
                    pPacket->target_val);

    uint16_t predValue = parseControlBlock(true);
    if (!ignoreFenceConfig && !shouldIgnoreCommand(predValue))
    {
        if (!m_cpFenceSm[fenceIndex].fenceCommand(pPacket->target_val, pPacket->dec_val))
        {
            return false;
        }
    }

    updateNextBuffer(packetSize);

    return true;
}

bool QmanCpInfo::parseWait()
{
    const uint32_t packetSize = 2;
    if (!validatePacketSize(packetSize))
    {
        return false;
    }

    packet_wait* pPacket      = (packet_wait*)m_hostAddress;
    uint32_t*    pHostAddress = (uint32_t*)m_hostAddress;

    LOG_GCP_VERBOSE("{}{}: Wait packet (0x{:x} : [{:#x}]): ID {} Cycles {} Inc-Val {}",
                    getIndentation(),
                    getPacketIndexDesc(),
                    (uint64_t)m_hostAddress,
                    fmt::join(&pHostAddress[0], &pHostAddress[2], " "),
                    pPacket->id,
                    pPacket->num_cycles_to_wait,
                    pPacket->inc_val);

    parseControlBlock(false);

    updateNextBuffer(packetSize);

    return true;
}

bool QmanCpInfo::parseRepeat()
{
    const uint32_t packetSize = 2;
    if (!validatePacketSize(packetSize))
    {
        return false;
    }

    packet_repeat* pPacket      = (packet_repeat*)m_hostAddress;
    uint32_t*      pHostAddress = (uint32_t*)m_hostAddress;

    LOG_GCP_VERBOSE("{}{}: Repeat packet (0x{:x} : [{:#x}]): Is-Start {} Is-Outter {} Jump-PTR 0x{:x}",
                    getIndentation(),
                    getPacketIndexDesc(),
                    (uint64_t)m_hostAddress,
                    fmt::join(&pHostAddress[0], &pHostAddress[2], " "),
                    pPacket->sore,
                    pPacket->o,
                    pPacket->jmp_ptr);

    parseControlBlock(true);

    updateNextBuffer(packetSize);

    return true;
}

bool QmanCpInfo::parseLoadAndExecute()
{
    const uint32_t packetSize = 4;
    if (!validatePacketSize(packetSize))
    {
        return false;
    }

    packet_load_and_exe* pPacket      = (packet_load_and_exe*)m_hostAddress;
    uint32_t*            pHostAddress = (uint32_t*)m_hostAddress;

    LOG_GCP_VERBOSE("{}{}: Load-And-Execute packet (0x{:x}: [{:#x}]): address 0x{:x} operation {} {} {} {}",
                    getIndentation(),
                    getPacketIndexDesc(),
                    (uint64_t)m_hostAddress,
                    fmt::join(&pHostAddress[0], &pHostAddress[4], " "),
                    pPacket->src_addr,
                    pPacket->load,
                    pPacket->exe,
                    ((pPacket->load) ? ((pPacket->dst) ? "load-scalar" : "load-predicates") : ""),
                    ((pPacket->exe) ? ((pPacket->etype) ? "execute-upper-RFs" : "execute-lower-RFs") : ""));

    parseControlBlock(true);

    updateNextBuffer(packetSize);

    return true;
}

void QmanCpInfo::parseMsgShortBasic(uint32_t base, uint32_t operation, uint32_t msgAddressOffset, uint32_t value)
{
    uint32_t* pHostAddress = (uint32_t*)m_hostAddress;

    LOG_GCP_VERBOSE("{}{}: MSG-Short packet (0x{:x} : [{:#x}]): {} (Value 0x{:x})",
                    getIndentation(),
                    getPacketIndexDesc(),
                    (uint64_t)m_hostAddress,
                    fmt::join(&pHostAddress[0], &pHostAddress[2], " "),
                    ((operation) ? "write-timestamp" : "write-value"),
                    value);

    LOG_GCP_VERBOSE("{}                  base {} MsgAddrOffset 0x{:x}", getIndentation(), base, msgAddressOffset);

    parseControlBlock(false);
}

bool QmanCpInfo::parseSyncObjUpdate(void*                pPacketBuffer,
                                    uint32_t             sobjAddressOffset,
                                    uint32_t             value,
                                    eSyncManagerInstance syncMgrInstance,
                                    bool                 isMsgShort)
{
    packet_msg_short* pPacket = (packet_msg_short*)pPacketBuffer;

    uint32_t* pHostAddress = (uint32_t*)m_hostAddress;

    static const uint32_t syncObjEntrySize      = sizeof(sob_objs::reg_sob_obj);
    static const uint32_t syncObjGroupSize      = 8;
    static const uint32_t syncObjGroupEntrySize = syncObjGroupSize * syncObjEntrySize;

    uint32_t whichGroup        = sobjAddressOffset / syncObjGroupEntrySize;
    uint32_t whichEntryInGroup = (sobjAddressOffset - (whichGroup * syncObjGroupEntrySize)) / syncObjEntrySize;
    uint32_t which             = sobjAddressOffset / syncObjEntrySize;

    if (which >= SOBJS_AMOUNT)
    {
        LOG_GCP_FAILURE("{}{}: MSG-{} packet (SO-Update) (0x{:x}): Invalid SOBJ-ID ({})",
                        getIndentation(),
                        getPacketIndexDesc(),
                        isMsgShort ? "Short" : "Long",
                        (uint64_t)m_hostAddress,
                        which);

        return false;
    }

    bool     isAdd               = (bool)pPacket->so_upd.mode;
    bool     isTraceEventEnabled = (bool)pPacket->so_upd.te;
    uint32_t syncValue           = pPacket->so_upd.sync_value;

    if (isMsgShort)
    {
        LOG_GCP_VERBOSE("{}{}: MSG-Short packet (SO-Update) (0x{:x} : [{:#x}]):"
                        " {} SO-Id {} (GroupId 0x{:x} Mask 0x{:x}) syncValue 0x{:x} {}",
                        getIndentation(),
                        getPacketIndexDesc(),
                        (uint64_t)m_hostAddress,
                        fmt::join(&pHostAddress[0], &pHostAddress[2], " "),
                        (isAdd) ? "Add" : "Set",
                        which,
                        whichGroup,
                        (uint8_t) ~(1 << whichEntryInGroup),
                        syncValue,
                        (isTraceEventEnabled) ? "Trace-events enabled" : "");
    }
    else
    {
        LOG_GCP_VERBOSE("{}{}: MSG-Long packet (SO-Update) (0x{:x} : [{:#x}]):"
                        " {} SYNC-MGR {} SO-Id {} (GroupId 0x{:x} Mask 0x{:x}) syncValue 0x{:x} {}",
                        getIndentation(),
                        getPacketIndexDesc(),
                        (uint64_t)m_hostAddress,
                        fmt::join(&pHostAddress[0], &pHostAddress[4], " "),
                        (isAdd) ? "Add" : "Set",
                        _getSyncManagerDescription(syncMgrInstance),
                        which,
                        whichGroup,
                        (uint8_t) ~(1 << whichEntryInGroup),
                        syncValue,
                        (isTraceEventEnabled) ? "Trace-events enabled" : "");
    }

    parseControlBlock(false);

    return true;
}

bool QmanCpInfo::parseMonitorPacket(void*                pPacketBuffer,
                                    uint32_t             monitorAddressOffset,
                                    eSyncManagerInstance syncMgrInstance,
                                    bool                 isMsgShort)
{
    packet_msg_short* pPacket = (packet_msg_short*)pPacketBuffer;

    if (monitorAddressOffset >= MONITOR_ARM_BLOCK_BASE)
    {
        unsigned monitorId = (monitorAddressOffset - MONITOR_ARM_BLOCK_BASE) / MONITOR_ARM_SIZE;

        if (monitorId > MONITORS_AMOUNT)
        {
            LOG_GCP_FAILURE("Invalid Monitor-ID ({}) [Monitor-ARM]", monitorId);
            return false;
        }

        return parseMonitorArm(pPacket, monitorId, syncMgrInstance, isMsgShort);
    }
    else
    {
        return parseMonitorSetup(pPacket, monitorAddressOffset, syncMgrInstance, isMsgShort);
    }

    return true;
}

bool QmanCpInfo::parseMonitorArm(void*                pPacketBuffer,
                                 uint32_t             monitorId,
                                 eSyncManagerInstance syncMgrInstance,
                                 bool                 isMsgShort)
{
    packet_msg_short* pPacket = (packet_msg_short*)pPacketBuffer;

    uint32_t* pHostAddress = (uint32_t*)m_hostAddress;

    if (isMsgShort)
    {
        LOG_GCP_VERBOSE("{}{}: MSG-Short packet (Monitor-Arm) (0x{:x} : [{:#x}]):"
                        " Monitor-ID {} Sync GroupId 0x{:x} Mask 0x{:x} {} Value 0x{:x}",
                        getIndentation(),
                        getPacketIndexDesc(),
                        (uint64_t)m_hostAddress,
                        fmt::join(&pHostAddress[0], &pHostAddress[2], " "),
                        monitorId,
                        pPacket->mon_arm_register.sync_group_id,
                        pPacket->mon_arm_register.mask,
                        (pPacket->mon_arm_register.mode) ? "Equal" : "GEQ",
                        pPacket->mon_arm_register.sync_value);
    }
    else
    {
        LOG_GCP_VERBOSE("{}{}: MSG-Long packet (Monitor-Arm) (0x{:x} : [{:#x}]):"
                        " SYNC-MGR {} Monitor-ID {} Sync GroupId 0x{:x} Mask 0x{:x} {} Value 0x{:x}",
                        getIndentation(),
                        getPacketIndexDesc(),
                        (uint64_t)m_hostAddress,
                        fmt::join(&pHostAddress[0], &pHostAddress[4], " "),
                        _getSyncManagerDescription(syncMgrInstance),
                        monitorId,
                        pPacket->mon_arm_register.sync_group_id,
                        pPacket->mon_arm_register.mask,
                        (pPacket->mon_arm_register.mode) ? "Equal" : "GEQ",
                        pPacket->mon_arm_register.sync_value);
    }

    SyncManagerInfo&     syncManagerInfo = (*m_pSyncManagerInfoDb)[gaudi::SYNC_MNGR_GC];
    MonitorStateMachine& monitorSM       = syncManagerInfo.getMonitorSM(monitorId, &m_cpFenceSm);

    uint16_t predValue = parseControlBlock(!isMsgShort);
    if (!shouldIgnoreCommand(predValue))
    {
        if (!monitorSM.monitorArmCommand())
        {
            return false;
        }
    }

    return true;
}

bool QmanCpInfo::parseMonitorSetup(void*                pPacketBuffer,
                                   uint32_t             msgAddressOffset,
                                   eSyncManagerInstance syncMgrInstance,
                                   bool                 isMsgShort)
{
    packet_msg_short* pPacket = (packet_msg_short*)pPacketBuffer;

    std::string description("Monitor-Setup ");
    std::string monitorSyncMgrDesc = _getSyncManagerDescription(syncMgrInstance);

    unsigned baseAddress = 0;
    unsigned monitorId   = 0;

    eMonitorSetupPhase monSetupPhase = MONITOR_SETUP_PHASE_NOT_SET;

    if (msgAddressOffset < MONITOR_PAYLOAD_HIGH_ADDRESS_BLOCK_BASE)
    {
        baseAddress   = MONITOR_PAYLOAD_LOW_ADDRESS_BLOCK_BASE;
        monitorId     = (msgAddressOffset - baseAddress) / MONITOR_PAYL_LOW_SIZE;
        monSetupPhase = MONITOR_SETUP_PHASE_LOW;

        m_monitorSetupInfo.m_monitorSetupAddressLow = m_currentPacketValueField;
    }
    else if (msgAddressOffset < MONITOR_PAYLOAD_DATA_BLOCK_BASE)
    {
        baseAddress   = MONITOR_PAYLOAD_HIGH_ADDRESS_BLOCK_BASE;
        monitorId     = (msgAddressOffset - baseAddress) / MONITOR_PAYL_HIGH_SIZE;
        monSetupPhase = MONITOR_SETUP_PHASE_HIGH;

        m_monitorSetupInfo.m_monitorSetupAddressHigh = m_currentPacketValueField;
    }
    else
    {
        baseAddress   = MONITOR_PAYLOAD_DATA_BLOCK_BASE;
        monitorId     = (msgAddressOffset - baseAddress) / MONITOR_PAYL_DATA_SIZE;
        monSetupPhase = MONITOR_SETUP_PHASE_DATA;

        m_monitorSetupInfo.m_monitorSetupData = m_currentPacketValueField;
    }
    std::string monitorSetupPhastDesc = _getMonitorSetupPhaseDescription(monSetupPhase);
    description.append(monitorSetupPhastDesc);

    uint32_t* pHostAddress = (uint32_t*)m_hostAddress;

    // Current packet
    if (isMsgShort)
    {
        LOG_GCP_VERBOSE("{}{}: MSG-Short packet ({} for Monitor-ID {})"
                        " (0x{:x} : [{:#x}]): Weakly-Order {} No-Snoop {}",
                        getIndentation(),
                        getPacketIndexDesc(),
                        description,
                        monitorId,
                        (uint64_t)m_hostAddress,
                        fmt::join(&pHostAddress[0], &pHostAddress[2], " "),
                        pPacket->weakly_ordered,
                        pPacket->no_snoop);
    }
    else
    {
        LOG_GCP_VERBOSE("{}{}: MSG-Long packet ({} for SYNC-MGR {} Monitor-ID {})"
                        " (0x{:x} : [{:#x}]): Weakly-Order {} No-Snoop {}",
                        getIndentation(),
                        getPacketIndexDesc(),
                        description,
                        monitorSyncMgrDesc,
                        monitorId,
                        (uint64_t)m_hostAddress,
                        fmt::join(&pHostAddress[0], &pHostAddress[4], " "),
                        pPacket->weakly_ordered,
                        pPacket->no_snoop);
    }

    uint16_t predValue = parseControlBlock(!isMsgShort);
    if (shouldIgnoreCommand(predValue))
    {
        return true;
    }

    if (m_monitorSetupInfo.m_monitorSetupState & monSetupPhase)
    {
        LOG_GCP_FAILURE("Re-configure of {} for Monitor-ID ({}) [Monitor-Setup]", monitorSetupPhastDesc, monitorId);

        return false;
    }
    m_monitorSetupInfo.m_monitorSetupState =
        (eMonitorSetupPhase)(m_monitorSetupInfo.m_monitorSetupState | monSetupPhase);

    if (monitorId > MONITORS_AMOUNT)
    {
        LOG_GCP_FAILURE("Invalid Monitor-ID ({}) [Monitor-Setup]", monitorId);
        return false;
    }

    if (m_monitorSetupInfo.m_monitorSetupMonitorId == common::INVALID_MONITOR_ID)
    {
        m_monitorSetupInfo.m_monitorSetupMonitorId = monitorId;
    }
    else if (m_monitorSetupInfo.m_monitorSetupMonitorId != monitorId)
    {
        LOG_GCP_FAILURE("{}{}: MSG-{} packet ({}) (0x{:x}): Invalid Monitor-ID (expected {} found {})",
                        getIndentation(),
                        getPacketIndexDesc(),
                        isMsgShort ? "Short" : "Long",
                        description,
                        (uint64_t)m_hostAddress,
                        m_monitorSetupInfo.m_monitorSetupMonitorId,
                        monitorId);

        return false;
    }

    if (m_monitorSetupInfo.isMonitorReady())
    {
        SyncManagerInfo&     syncManagerInfo = (*m_pSyncManagerInfoDb)[gaudi::SYNC_MNGR_GC];
        MonitorStateMachine& monitorSM       = syncManagerInfo.getMonitorSM(monitorId, &m_cpFenceSm);

        uint64_t monitorSetupAddress = ((uint64_t)m_monitorSetupInfo.m_monitorSetupAddressHigh) << 32 |
                                       m_monitorSetupInfo.m_monitorSetupAddressLow;

        std::string       monitorAddressDescription;
        std::stringstream descriptionStream;

        eSyncManagerInstance monitorSetupSyncMgrInstance = gaudi::SYNC_MNGR_EAST_NORTH;

        // Offset with respect to the SOBJ_BASE_ADDRESS or relevant SYNC_MGR instance
        uint32_t syncObjectAddressOffset = 0;

        bool isSyncObject =
            getSyncObjectAddressOffset(monitorSetupAddress, syncObjectAddressOffset, monitorSetupSyncMgrInstance);
        if (isSyncObject)
        {
            descriptionStream << "SYNC-MGR " << _getSyncManagerDescription(monitorSetupSyncMgrInstance) << " ";

            // GC's SYNC_MGR_E_N Sync-Object
            uint32_t syncObjIndex = syncObjectAddressOffset / SYNC_OBJECT_SIZE;
            descriptionStream << "Sync-Object " << syncObjIndex;
            monitorAddressDescription.append(descriptionStream.str());
            LOG_GCP_VERBOSE("{}==> Monitor-Setup: SYNC-MGR {} Monitor-ID {} Address 0x{:x} ({}) Data 0x{:x}",
                            getIndentation(),
                            monitorSyncMgrDesc,
                            monitorId,
                            monitorSetupAddress,
                            monitorAddressDescription,
                            m_monitorSetupInfo.m_monitorSetupData);  // std::string parseMonitorSetupData() => "Data
                                                                     // 0x{:x}" or "Inc by {}" use parseSyncObjUpdate

            if (!monitorSM.monitorSetupCommandSobj(syncObjIndex))
            {
                return false;
            }
        }
        else
        {  // QMAN's Fence
            uint64_t  qmanSize        = 0;
            uint64_t  qmanBaseAddress = 0;
            eQmanType qmanType        = common::QMAN_TYPE_TPC;

            if ((monitorSetupAddress >= TPC_QMAN_BASE_ADDRESS) && (monitorSetupAddress <= TPC_QMAN_END_ADDRESS))
            {
                qmanSize        = TPC_QMAN_SIZE;
                qmanBaseAddress = TPC_QMAN_BASE_ADDRESS;
                qmanType        = common::QMAN_TYPE_TPC;
                descriptionStream << "TPC";
            }
            else if ((monitorSetupAddress >= DMA_QMAN_BASE_ADDRESS) && (monitorSetupAddress <= DMA_QMAN_END_ADDRESS))
            {
                qmanSize        = DMA_QMAN_SIZE;
                qmanBaseAddress = DMA_QMAN_BASE_ADDRESS;
                qmanType        = common::QMAN_TYPE_DMA;
                descriptionStream << "DMA";
            }
            else if ((monitorSetupAddress >= MME_QMAN_BASE_ADDRESS) && (monitorSetupAddress <= MME_QMAN_END_ADDRESS))
            {
                qmanSize        = MME_QMAN_SIZE;
                qmanBaseAddress = MME_QMAN_BASE_ADDRESS;
                qmanType        = common::QMAN_TYPE_MME;
                descriptionStream << "MME";
            }
            else
            {
                LOG_GCP_FAILURE("{}Monitor-Setup (0x{:x}): SYNC-MGR {} Monitor-ID {} Invalid-Address(1) 0x{:x}",
                                getIndentation(),
                                (uint64_t)m_hostAddress,
                                monitorSyncMgrDesc,
                                monitorId,
                                monitorSetupAddress);

                return false;
            }

            uint32_t qmanTypeIndex    = (monitorSetupAddress - qmanBaseAddress) / qmanSize;
            uint64_t fenceBaseAddress = (monitorSetupAddress - qmanBaseAddress) % qmanSize;

            uint32_t finalQmanTypeIndex = qmanTypeIndex;
            // Converts into "actual" QMAN's Index
            // Due to MME special definition MME-0 is MME-2, MME-2 is MME-0, War is peace, freedom is slavery...
            if (qmanType == common::QMAN_TYPE_MME)
            {
                finalQmanTypeIndex += 2 * (1 - qmanTypeIndex);
            }

            // The fenceBaseAddress now needs to be in the QM block boundary
            if ((fenceBaseAddress < FENCE_BLOCK_OFFSET) ||
                (fenceBaseAddress >= FENCE_BLOCK_OFFSET + FENCE_BLOCK_TOTAL_SIZE))
            {
                LOG_GCP_FAILURE("{}Monitor-Setup (0x{:x}): SYNC-MGR {} Monitor-ID {} Invalid-Address(3) 0x{:x}",
                                getIndentation(),
                                (uint64_t)m_hostAddress,
                                monitorSyncMgrDesc,
                                monitorId,
                                monitorSetupAddress);

                return false;
            }

            // Getting the offset from the fence block (from "offset from QM" to "offset from fence-block")
            fenceBaseAddress -= FENCE_BLOCK_OFFSET;
            uint32_t qmanIndex = 0;
            if (!_getQmanIndex(qmanIndex, qmanType, finalQmanTypeIndex))
            {
                LOG_GCP_FAILURE("{}Monitor-Setup (0x{:x}): SYNC-MGR {} Monitor-ID {}"
                                " Invalid QMAN index (QmanTypeIndex {})",
                                getIndentation(),
                                (uint64_t)m_hostAddress,
                                monitorSyncMgrDesc,
                                monitorId,
                                finalQmanTypeIndex);

                return false;
            }

            uint32_t qmanFenceIndex = fenceBaseAddress / SINGLE_CP_RDATA_SIZE;

            uint32_t fenceIdInCp   = 0;
            uint32_t cpIndexInQman = 0;
            // No need to check status, as we vaildated that fenceBaseAddress is part of FANCE block
            _getFenceInfo(fenceIdInCp, cpIndexInQman, qmanFenceIndex);

            descriptionStream << "-" << finalQmanTypeIndex << " FENCE-ID " << fenceIdInCp << " CP-index "
                              << cpIndexInQman << " RDATA";
            monitorAddressDescription.append(descriptionStream.str());

            LOG_GCP_VERBOSE("{}==> Monitor-ID {} (SYNC-MGR {}): Monitor-Setup Address 0x{:x} ({}) Data 0x{:x}",
                            getIndentation(),
                            monitorId,
                            monitorSyncMgrDesc,
                            monitorSetupAddress,
                            monitorAddressDescription,
                            m_monitorSetupInfo.m_monitorSetupData);

            uint32_t cpIndex    = (qmanIndex * CPS_PER_QMAN) + cpIndexInQman;
            uint32_t fenceIndex = (cpIndex * FENCE_IDS_PER_CP) + fenceIdInCp;

            bool status = monitorSM.monitorSetupCommandFence(fenceIndex, m_monitorSetupInfo.m_monitorSetupData);
            if (!status)
            {
                return false;
            }
        }

        m_monitorSetupInfo.clear();
    }

    return true;
}

bool QmanCpInfo::getSyncObjectAddressOffset(uint64_t              fullAddress,
                                            uint32_t&             syncObjectAddressOffset,
                                            eSyncManagerInstance& syncMgrInstance)
{
    if ((fullAddress >= EAST_SOUTH_SOBJ_BASE_ADDRESS) && (fullAddress < EAST_SOUTH_SOBJ_LAST_ADDRESS))
    {
        syncObjectAddressOffset = fullAddress - EAST_SOUTH_SOBJ_BASE_ADDRESS;
        syncMgrInstance         = gaudi::SYNC_MNGR_EAST_SOUTH;
        return true;
    }
    else if ((fullAddress >= WEST_SOUTH_SOBJ_BASE_ADDRESS) && (fullAddress < WEST_SOUTH_SOBJ_LAST_ADDRESS))
    {
        syncObjectAddressOffset = fullAddress - WEST_SOUTH_SOBJ_BASE_ADDRESS;
        syncMgrInstance         = gaudi::SYNC_MNGR_WEST_SOUTH;
        return true;
    }
    else if ((fullAddress >= GC_SOBJ_BASE_ADDRESS) && (fullAddress < GC_SOBJ_LAST_ADDRESS))
    {
        syncObjectAddressOffset = fullAddress - GC_SOBJ_BASE_ADDRESS;
        syncMgrInstance         = gaudi::SYNC_MNGR_EAST_NORTH;
        return true;
    }
    else if ((fullAddress >= WEST_NORTH_SOBJ_BASE_ADDRESS) && (fullAddress < WEST_NORTH_SOBJ_LAST_ADDRESS))
    {
        syncObjectAddressOffset = fullAddress - WEST_NORTH_SOBJ_BASE_ADDRESS;
        syncMgrInstance         = gaudi::SYNC_MNGR_WEST_NORTH;
        return true;
    }

    return false;
}

bool QmanCpInfo::getMonitorAddressOffset(uint64_t              fullAddress,
                                         uint32_t&             monitorAddressOffset,
                                         eSyncManagerInstance& syncMgrInstance)
{
    if ((fullAddress >= EAST_SOUTH_MONITOR_BASE_ADDRESS) && (fullAddress < EAST_SOUTH_MONITOR_LAST_ADDRESS))
    {
        monitorAddressOffset = fullAddress - EAST_SOUTH_MONITOR_BASE_ADDRESS;
        syncMgrInstance      = gaudi::SYNC_MNGR_EAST_SOUTH;
        return true;
    }
    else if ((fullAddress >= WEST_SOUTH_MONITOR_BASE_ADDRESS) && (fullAddress < WEST_SOUTH_MONITOR_LAST_ADDRESS))
    {
        monitorAddressOffset = fullAddress - WEST_SOUTH_MONITOR_BASE_ADDRESS;
        syncMgrInstance      = gaudi::SYNC_MNGR_WEST_SOUTH;
        return true;
    }
    else if ((fullAddress >= GC_MONITOR_BASE_ADDRESS) && (fullAddress < GC_MONITOR_LAST_ADDRESS))
    {
        monitorAddressOffset = fullAddress - GC_MONITOR_BASE_ADDRESS;
        syncMgrInstance      = gaudi::SYNC_MNGR_EAST_NORTH;
        return true;
    }
    else if ((fullAddress >= WEST_NORTH_MONITOR_BASE_ADDRESS) && (fullAddress < WEST_NORTH_MONITOR_LAST_ADDRESS))
    {
        monitorAddressOffset = fullAddress - WEST_NORTH_MONITOR_BASE_ADDRESS;
        syncMgrInstance      = gaudi::SYNC_MNGR_WEST_NORTH;
        return true;
    }

    return false;
}

std::string QmanCpInfo::getPacketName(uint64_t packetId)
{
    if (packetId >= sizeof(m_packetsName) / sizeof(std::string))
    {
        return "Invalid";
    }

    return m_packetsName[packetId];
}

uint64_t QmanCpInfo::getFenceIdsPerCp()
{
    return FENCE_IDS_PER_CP;
}

bool QmanCpInfo::_getQmanIndex(uint32_t& qmanIndex, eQmanType qmanType, uint32_t qmanTypeIndex)
{
    bool status = true;

    uint32_t masterQmanTypeIndex = qmanTypeIndex;

    qmanIndex = 0;
    do
    {
        if (qmanType == common::QMAN_TYPE_DMA)
        {
            status = (qmanTypeIndex < DMA_QMANS_AMOUNT);
            break;
        }
        qmanIndex += DMA_QMANS_AMOUNT;

        if (qmanType == common::QMAN_TYPE_MME)
        {
            status              = (qmanTypeIndex < MME_QMANS_AMOUNT);
            masterQmanTypeIndex = qmanTypeIndex / 2;  // Each MME has Master-Slave couple
            break;
        }
        qmanIndex += MME_MASTER_QMANS_AMOUNT;

        status = (qmanTypeIndex < TPC_QMANS_AMOUNT);
    } while (0);  // Do once

    qmanIndex += masterQmanTypeIndex;

    return status;
}

bool QmanCpInfo::_getFenceInfo(uint32_t& fenceId, uint32_t& cpIndex, uint32_t fenceIndex)
{
    // The block is defined as {FENCE X {CPs}} - going over the CPs over a given FENCE-ID
    fenceId = fenceIndex / CPS_PER_QMAN;
    cpIndex = (fenceIndex % CPS_PER_QMAN);

    if ((fenceId >= FENCE_IDS_PER_CP) || (cpIndex >= CPS_PER_QMAN))
    {
        return false;
    }

    return true;
}

std::string QmanCpInfo::_getSyncManagerDescription(eSyncManagerInstance syncMgrInstanceId)
{
    if (syncMgrInstanceId >= m_syncManagerInstanceDesc.size())
    {
        return "Invalid";
    }

    return m_syncManagerInstanceDesc[syncMgrInstanceId];
}