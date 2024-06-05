#pragma once

#include <atomic>
#include <sstream>
#include <memory>
#include "types.h"
#include "address_fields_container_info.h"
#include "sync/sync_types.h"
#include "params_file_manager.h"
#include "settable.h"

#define UNDEFINED_NODE_EXE_INDEX -1

static unsigned const SEND_SYNC_BIT = 30;

namespace gc_recipe
{
    class CommandBuffer;
    class generic_packets_container;
}


// Base class for commands that can be pushed into command queues (Goya and Gaudi)
class QueueCommand
{
public:
    QueueCommand();
    QueueCommand(uint32_t packetType);
    virtual ~QueueCommand();

    virtual void      Print() const = 0;
    virtual unsigned  GetBinarySize() const = 0;
    virtual void      WritePB(gc_recipe::generic_packets_container* pktCon) = 0;
    virtual void      WritePB(gc_recipe::generic_packets_container* pktCon, ParamsManager* params) = 0;
    virtual void      SetContainerInfo(const BasicFieldsContainerInfo& afContainerInfo);
    uint64_t          GetCommandId()    {return m_commandId;}
    static uint64_t   GetNewCommandId() {return ++m_nextCommandId;}

    virtual uint64_t  writeInstruction(void* whereTo) const; // returns how many bytes were written
    virtual void      prepareFieldInfos();
    virtual void      setAsExe()                     { m_isExe = true; }
    virtual bool      isExe() const                  { return m_isExe; }
    virtual bool      isMonitorArm() const           { return false; }
    virtual bool      isBlobCommitter() const        { return m_isBlobCommitter; }
    virtual bool      isDynamic() const              { return m_isDynamic; }
    virtual bool      invalidateHistory() const      { return false; }

    virtual const BasicFieldsContainerInfo& getBasicFieldsContainerInfo() const;

    virtual void    setAsBlobCommitter(int64_t nodeExeIdx); // node exe index is needed for stage submission
    virtual void    setNodeExecutionIndex(int64_t idx) { m_nodeExecutionIndex = idx; }
    virtual int64_t getNodeExecutionIndex() const { return m_nodeExecutionIndex; }

    virtual void setSwitchCQ()      = 0;
    virtual void resetSwitchCQ()    = 0;
    virtual void toggleSwitchCQ()   = 0;
    virtual bool isSwitchCQ() const = 0;

protected:
    QueueCommand(uint64_t commandId);
    QueueCommand(uint32_t packetType, uint64_t commandId);

    BasicFieldsContainerInfo m_addressContainerInfo;
    const uint64_t           m_commandId;  // id for command that has patching-info, applicable to all packets in the command
    const uint32_t           m_packetType;
    static const uint32_t    INVALID_PACKET_TYPE = 0xFFFFFFFF;
    bool                     m_isExe             = false;
    bool                     m_isBlobCommitter   = false;
    bool                     m_isDynamic         = false;

private:
    QueueCommand(const QueueCommand&)   = delete;
    void operator=(const QueueCommand&) = delete;

    static std::atomic<uint64_t> m_nextCommandId;
    int64_t                      m_nodeExecutionIndex = UNDEFINED_NODE_EXE_INDEX;
};

using QueueCommandPtr = std::unique_ptr<QueueCommand>;
