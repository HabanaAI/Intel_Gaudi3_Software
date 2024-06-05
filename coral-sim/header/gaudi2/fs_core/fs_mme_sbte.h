#pragma once
#include <string.h>

#include <list>
#include <queue>

#include "gaudi2/mme.h"

#include "fs_mme_agu.h"
#include "fs_mme_dec.h"
#include "fs_mme_md.h"
#include "fs_mme_queue.h"
#include "fs_mme_sb.h"
#include "fs_mme_te.h"
#include "fs_mme_unit.h"

namespace Gaudi2
{
namespace Mme
{

class FS_SbTe : public Unit, public Cbb_Base
{
   public:
    static const unsigned c_comp_if_queue_depth = 4;

    FS_SbTe(coral_module_name m_name, FS_Mme* mme, const std::string& name = std::string());

    virtual ~FS_SbTe();
    void delete_queues();
    void create_queues(FS_Mme* mme);

    void setConnectivity(Gaudi2::Mme::EMmeBrainIdx aguId, Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* sbte2eus_Q);

    void setDebugMemAccess(MmeCommon::MemAccess* memAccess) { m_sb->setDebugMemAccess(memAccess); }
    void launch();
    void terminate(bool queue_reset = false);
    void setName(const std::string& name);
    void get_coresight_params(uint8_t& stream_id, bool& stm_en);
    void connect_to_lbw(LBWHub* lbw, Specs* specs, uint32_t itr) override;
    void connect_stm_etf_to_lbw(LBWHub* lbw, Specs* specs, uint32_t itr, unsigned idx);

    Gaudi2::Mme::AGU* m_agu;

   private:
    Gaudi2::Mme::EMmeBrainIdx m_aguId;
    std::string               m_name;

    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Agu2Sb>* m_agu2sb_Q;
    Gaudi2::Mme::SB*                            m_sb;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sb2Te>*  m_sb2te_Q;
    Gaudi2::Mme::TE*                            m_te;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Te2Dec>* m_te2dec_Q;
    Gaudi2::Mme::DEC*                           m_dec;
};
} // namespace Mme
} // namespace Gaudi2
