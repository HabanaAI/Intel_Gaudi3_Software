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
class EUBrain : public Gaudi2::Mme::Thread
{
   public:
    static const unsigned c_mme_eus_max_common_dim = 128;

    EUBrain(FS_Mme* mme = nullptr, const std::string& name = std::string());

    virtual ~EUBrain() override {}

    void setConnectivity(Gaudi2::Mme::Queue<Gaudi2::Mme::QS_EusBrain>* eusBrain2eusQueue);

    void genDescs(const Gaudi2::Mme::Desc* desc);

   protected:
    virtual void execute() override;

   private:
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_EusBrain>* m_eusBrain2eusQueue;
    std::list<Gaudi2::Mme::EUSDescriptor>         m_descQ;
    unsigned                                      m_accId;
    unsigned                                      m_descCnt;

    inline void validateBias(unsigned bias)
    {
        switch (Gaudi2::Mme::EMmeFP8LegalBias(bias)) {
            case e_mme_fp8_143_bias_3:
            case e_mme_fp8_143_bias_7:
            case e_mme_fp8_143_bias_11:
            case e_mme_fp8_143_bias_15: return;
            default: FS_ASSERT_MSG(0, "Illegal FP8 bias");
        }
    }

    unsigned adjustFCD(const unsigned fcd, const Gaudi2::Mme::EMmeDataType dataType);
    void     dumpDescIO(const Gaudi2::Mme::Desc* desc);
};
} // namespace Mme
} // namespace Gaudi2
