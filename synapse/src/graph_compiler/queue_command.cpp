#include <atomic>

#include "queue_command.h"


std::atomic<uint64_t> QueueCommand::m_nextCommandId(INVALID_CONTAINER_ID);

QueueCommand::QueueCommand() :
        m_commandId(++m_nextCommandId),
        m_packetType(INVALID_PACKET_TYPE)
{
}

QueueCommand::QueueCommand(uint64_t commandId) :
        m_commandId(commandId),
        m_packetType(INVALID_PACKET_TYPE)
{
}

QueueCommand::QueueCommand(uint32_t packetType, uint64_t commandId) :
        m_commandId(commandId),
        m_packetType(packetType)
{
}

QueueCommand::QueueCommand(uint32_t packetType) :
        m_commandId(++m_nextCommandId),
        m_packetType(packetType)
{
}

QueueCommand::~QueueCommand()
{
}

void QueueCommand::SetContainerInfo(const BasicFieldsContainerInfo& afContainerInfo)
{
    m_addressContainerInfo = afContainerInfo;
    m_addressContainerInfo.updateContainerId(m_commandId);

    const AddressFieldInfoSet& addressFieldInfoSet = m_addressContainerInfo.retrieveAddressFieldInfoSet();
    if (LOG_LEVEL_AT_LEAST_TRACE(QMAN) && addressFieldInfoSet.size() > 0)
    {
        for (auto addressInfoPair : addressFieldInfoSet)
        {
            AddressFieldInfoSharedPtr pAddressInfo = addressInfoPair.second;

            LOG_TRACE(QMAN,
                      "{}: m_commandId=0x{:x}, memoryID={}, sectionName={}, targetAddress=0x{:x}, "
                      "fieldEngineId=0x{:x}, addressPart={}, fieldIndexOffset=0x{:x}",
                      HLLOG_FUNC,
                      m_commandId,
                      pAddressInfo->getMemorySectionId(),
                      pAddressInfo->getSectionName(),
                      pAddressInfo->getTargetAddress(),
                      pAddressInfo->getEngineFieldId(),
                      pAddressInfo->getAddressPart(),
                      pAddressInfo->getFieldIndexOffset());
        }
    }
}

const BasicFieldsContainerInfo& QueueCommand::getBasicFieldsContainerInfo() const
{
    return m_addressContainerInfo;
}

uint64_t QueueCommand::writeInstruction(void* whereTo) const
{
    LOG_ERR(QMAN, "{}: default implementation should not be invoked", HLLOG_FUNC);
    HB_ASSERT(false, "default implementation should not be invoked");
    return 0;
}

void QueueCommand::prepareFieldInfos()
{
    LOG_ERR(QMAN, "{}: default implementation should not be invoked", HLLOG_FUNC);
    HB_ASSERT(false, "default implementation should not be invoked");
}

void QueueCommand::setAsBlobCommitter(int64_t nodeExeIdx)
{
    m_isBlobCommitter = true;
    setNodeExecutionIndex(nodeExeIdx);
}
