#pragma once

#include <stdint.h>

#include <list>

#include "gaudi2/mme.h"

#include "fs_mme_queue.h"
#include "fs_mme_queue_structs.h"
#include "fs_mme_thread.h"
#include "fs_mme_utils.h"

namespace Gaudi2
{
namespace Mme
{
class AccApBrain : public Gaudi2::Mme::Thread
{
   public:
    static const unsigned c_mme_eus_max_common_dim = 128;

    AccApBrain(FS_Mme* mme = nullptr, const std::string& name = std::string());

    virtual ~AccApBrain() {}

    void setConnectivity(Gaudi2::Mme::Queue<Gaudi2::Mme::QS_AccApBrain>* accApBrain2accApQueue);

   protected:
    virtual void execute();

   private:
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_AccApBrain>* m_accApBrain2accApQueue;
    std::list<Gaudi2::Mme::ACCAPDescriptor>         m_descQ;
    unsigned                                        m_accId;
    unsigned                                        m_descCnt;

    void genDescs(const Gaudi2::Mme::Desc* desc);

    inline uint8_t encodeChunkSize(unsigned chunkSize) { return chunkSize == 32 ? 0 : (chunkSize == 64 ? 1 : 2); }

    inline uint8_t encodeapDim1Stride(unsigned chunkSize) { return chunkSize == 32 ? 0 : (chunkSize == 64 ? 1 : 2); }

    void dumpDescIO(const Gaudi2::Mme::Desc* desc);
};
} // namespace Mme
} // namespace Gaudi2
