#pragma once

#include <stdint.h>

#include "gaudi2/mme.h"

#include "fs_mme_md.h"
#include "fs_mme_mem_access.h"
#include "fs_mme_queue.h"
#include "fs_mme_queue_structs.h"
#include "fs_mme_thread.h"
#include "fs_mme_utils.h"

namespace Gaudi2
{
namespace Mme
{
class SB : public Gaudi2::Mme::Thread
{
   public:
    static const unsigned c_sb_cache_size  = 4;
    static const unsigned c_sb_cache_banks = 2;

    SB(FS_Mme* mme = nullptr, const std::string& name = std::string());

    virtual ~SB() override {}

    void setConnectivity(Gaudi2::Mme::EMmeInputOperand               operandType,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Agu2Sb>* agu2sbQueue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sb2Te>*  sb2teQueue);

    void setDebugMemAccess(MmeCommon::MemAccess* memAccess) { m_memAccess = memAccess; }

    Gaudi2::Mme::HbwRdMaster* m_hbwRdMstr;

   protected:
    virtual void execute() override;

   private:
    struct SBCacheEntry
    {
        uint64_t reuseWindowCtr;
        uint64_t addr;
        bool     valid;
        uint8_t  data[Gaudi2::Mme::c_cl_size];
    };

    bool readFromInput(unsigned& availInputs);

    bool writeToOutput(unsigned& availOutputSpace);

    Gaudi2::Mme::EMmeInputOperand               m_operandType;
    Gaudi2::Mme::EMmeBrainIdx                   m_aguId;
    unsigned                                    m_sbSize;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Agu2Sb>* m_agu2sbQueue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sb2Te>*  m_sb2teQueue;
    uint8_t                                     m_buffer[Gaudi2::Mme::c_mme_sb_size][Gaudi2::Mme::c_cl_size];
    Gaudi2::Mme::MetaData::SB                   m_md[Gaudi2::Mme::c_mme_sb_size];
    bool                                        m_valid[Gaudi2::Mme::c_mme_sb_size];
    SBCacheEntry                                m_cache[c_sb_cache_banks][c_sb_cache_size];
    uint64_t                                    m_cacheMissCtr[c_sb_cache_banks];
    uint64_t                                    m_axiRd;
    uint64_t                                    m_axiWr;
    uint64_t                                    m_reuseRd;
    uint64_t                                    m_reuseWindowCtr;
    bool                                        m_miss;
    Gaudi2::Mme::MetaData::SB                   m_missMD;
    MmeCommon::MemAccess*                       m_memAccess;
};
} // namespace Mme
} // namespace Gaudi2
