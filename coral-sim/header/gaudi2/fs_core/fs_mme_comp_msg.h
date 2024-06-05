#pragma once

#include <stdint.h>

#include <list>

#include "gaudi2/mme.h"

#include "fs_mme_condition_variable.h"
#include "fs_mme_mem_access.h"
#include "fs_mme_queue.h"
#include "fs_mme_queue_structs.h"
#include "fs_mme_thread.h"
#include "fs_mme_utils.h"

namespace Gaudi2
{
namespace Mme
{
class CompMsg : public Gaudi2::Mme::Thread
{
   public:
    CompMsg(const unsigned color, FS_Mme* mme = nullptr, const std::string& name = std::string());

    virtual ~CompMsg() override {}

    void setConnectivity(Gaudi2::Mme::Queue<Gaudi2::Mme::MmeSyncObject>* wbc0ToCompMsgQueue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::MmeSyncObject>* wbc1ToCompMsgQueue,
                         Gaudi2::Mme::Queue<bool>*                       remoteCompMsgQueue);

    void setDebugMemAccess(MmeCommon::MemAccess* memAccess) { m_memAccess = memAccess; }

    void fence(uint64_t signalsNr);

    void setName(const std::string& name);

   protected:
    virtual void execute() override;

   private:
    const unsigned                                  m_color;
    Gaudi2::Mme::Queue<Gaudi2::Mme::MmeSyncObject>* m_wbc0ToCompMsgQueue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::MmeSyncObject>* m_wbc1ToCompMsgQueue;
    Gaudi2::Mme::Queue<bool>*                       m_remoteCompMsgQueue;
    uint64_t                                        m_smCnt;
    MmeCommon::MemAccess*                           m_memAccess;
    std::mutex                                      m_mutex;
    Gaudi2::Mme::ConditionVariable                  m_cond;

    void sendSyncMessage(const uint8_t color, const Gaudi2::Mme::MmeSyncObject& syncObject);
};
} // namespace Mme
} // namespace Gaudi2
