#pragma once

#include <stdint.h>

#include "gaudi2/mme.h"

#include "fs_mme_md.h"
#include "fs_mme_mem_access.h"
#include "fs_mme_queue.h"
#include "fs_mme_queue_structs.h"
#include "fs_mme_thread.h"
#include "fs_mme_utils.h"
#include "mme_half.h"

namespace Gaudi2
{
namespace Mme
{
class WBC : public Gaudi2::Mme::Thread
{
   public:
    WBC(FS_Mme* mme = nullptr, const std::string& name = std::string());

    virtual ~WBC() override {}

    void setConnectivity(const Gaudi2::Mme::EMmeBrainIdx                 aguId,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Agu2Sb>*     agu2wbcQueue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Ap2Wbc>*     ap2wbcQueue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::MmeSyncObject>* wbc2compMsg0Queue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::MmeSyncObject>* wbc2compMsg1Queue);

    void setDebugMemAccess(MmeCommon::MemAccess* memAccess);
    void waitIdle();

   protected:
    virtual void execute() override;

   private:
    Gaudi2::Mme::EMmeBrainIdx                       m_aguId;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Agu2Sb>*     m_agu2wbcQueue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Ap2Wbc>*     m_ap2wbcQueue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::MmeSyncObject>* m_wbc2compMsg0Queue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::MmeSyncObject>* m_wbc2compMsg1Queue;
    unsigned                                        m_debugSignalCtr;
    MmeCommon::MemAccess*                           m_memAccess;
    unsigned                                        m_cmdCtrWbc;
};
} // namespace Mme
} // namespace Gaudi2
