#pragma once
#include <stdint.h>

#include <queue>
#include <string>

#include "fs_mme_accap.h"
#include "fs_mme_dataq.h"
#include "fs_mme_dec.h"
#include "fs_mme_eu_fp.h"
#include "fs_mme_md.h"
#include "fs_mme_queue.h"
#include "fs_mme_queue_structs.h"
#include "fs_mme_thread.h"
#include "mme_half.h"

namespace Gaudi2
{
namespace Mme
{
class EUS : public Gaudi2::Mme::Thread
{
   public:
    static const unsigned c_matrix_width_in_elements =
        Gaudi2::Mme::c_mme_dcore_matrix_width_in_bytes / sizeof(uint16_t);
    static const unsigned c_matrix_height_in_elements =
        Gaudi2::Mme::c_mme_dcore_matrix_height_in_bytes / sizeof(uint16_t);
    static const unsigned c_matrix_phy_width = c_matrix_width_in_elements / Gaudi2::Mme::EU_fp::c_fma_adder_depth;

    EUS(FS_Mme* mme = nullptr, const std::string& name = std::string());

    virtual ~EUS() override;

    void setConnectivity(FS_Mme*                                       remote_mme,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_EusBrain>* eusBrainQueue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* sb0ToEusQueue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* sb1ToEusQueue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* sb2ToEusQueue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* sb3ToEusQueue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* sb4ToEusQueue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* sbROut0ToEusQueue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* sbROut1ToEusQueue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Eus2Acc>*  eus2acc,
                         Gaudi2::Mme::EU_fp*                           euFp);

    unsigned getCoalescingStep(const Gaudi2::Mme::EMmeDataType& dataType);

   protected:
    virtual void execute() override;

   private:
    typedef enum
    {
        e_eus_routing_a0 = 0x0,
        e_eus_routing_a1 = 0x1,
        e_eus_routing_a2 = 0x2,
        e_eus_routing_a3 = 0x3,
        e_eus_routing_b0 = 0x4,
        e_eus_routing_b1 = 0x5,
        e_eus_routing_b2 = 0x6,
        e_eus_routing_b3 = 0x7,
    } EMmeEusRouting;

    union bufLine
    {

        uint8_t  b[Gaudi2::Mme::c_mme_dcore_matrix_width_in_bytes];
        uint16_t w[Gaudi2::Mme::c_mme_dcore_matrix_width_in_bytes / sizeof(uint16_t)];
    };
    struct chCoal
    {
        bufLine buf[Gaudi2::Mme::EU_fp::c_fma_fp8_adder_depth];
    };

    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_EusBrain>* m_eusBrainQueue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* m_sb0ToEusQueue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* m_sb1ToEusQueue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* m_sb2ToEusQueue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* m_sb3ToEusQueue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* m_sb4ToEusQueue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* m_sbRin0ToEusQueue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* m_sbRin1ToEusQueue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* m_eusToRout0Queue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sbte2Eus>* m_eusToRout1Queue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Eus2Acc>*  m_eus2acc;
    Gaudi2::Mme::EU_fp*                           m_euFp;
    chCoal*                                       m_aBuf;
    chCoal*                                       m_bBuf;
    Gaudi2::Mme::QS_Eus2Acc*                      m_rollup2acc;
    Gaudi2::Mme::EU_fp::RollupVec                 m_rollupFp[Gaudi2::Mme::EU_fp::c_matrix_height];
    unsigned                                      m_rollupCnt;

    Gaudi2_MME_Half::MME::PerfUoW::Ucmd m_ucmd;
    Gaudi2_MME_Half::MME*               coral_mme                = NULL;
    Gaudi2_MME_Half::MME*               coral_partner_mme        = NULL;
    uint64_t                            prev_cmd_id              = 0;
    uint64_t                            ucmd_id                  = 0;
    uint64_t                            cmd_sb0_start_id         = 0;
    uint64_t                            cmd_sb1_start_id         = 0;
    uint64_t                            cmd_sb2_start_id         = 0;
    uint64_t                            cmd_sb3_start_id         = 0;
    uint64_t                            cmd_sb4_start_id         = 0;
    uint64_t                            cmd_partner_sb0_start_id = 0;
    uint64_t                            cmd_partner_sb1_start_id = 0;
    uint64_t                            cmd_partner_sb2_start_id = 0;
    uint64_t                            cmd_partner_sb3_start_id = 0;
    uint64_t                            cmd_partner_sb4_start_id = 0;
    uint64_t                            sb0Reuse_end_id          = -1;
    uint64_t                            sb1Reuse_end_id          = -1;
    uint64_t                            sb2Reuse_end_id          = -1;
    uint64_t                            sb3Reuse_end_id          = -1;
    uint64_t                            sb4Reuse_end_id          = -1;
    uint64_t                            partner_sb0Reuse_end_id  = -1;
    uint64_t                            partner_sb1Reuse_end_id  = -1;
    uint64_t                            partner_sb2Reuse_end_id  = -1;
    uint64_t                            partner_sb3Reuse_end_id  = -1;
    uint64_t                            partner_sb4Reuse_end_id  = -1;

    void update_start_id(Gaudi2_MME_Half::MME::PerfUoW::SBEndUcmd& ucmd_sb,
                         uint64_t&                                 cmd_sb_start_id,
                         uint64_t                                  cmd_id,
                         uint64_t&                                 sbReuse_end_id,
                         std::string                               sb_str);

    bool channelCoalescing(const Gaudi2::Mme::EUSDescriptor& desc);
    void prepareOperandA(const Gaudi2::Mme::EUSDescriptor& desc, Gaudi2::Mme::EU_fp::VecA& vecA, const bool hx2Iter);
    void prepareOperandB(const Gaudi2::Mme::EUSDescriptor& desc,
                         Gaudi2::Mme::EU_fp::VecB&         vecB,
                         const unsigned                    cyc,
                         const bool                        hx2Iter);

    void     validateDesc(const Gaudi2::Mme::EUSDescriptor& desc);
    void     deInterleaveTE(QS_Sbte2Eus& vec);
    bool     getA(bool                      get4ports,
                  const Mme::EUSDescriptor& desc,
                  QS_Sbte2Eus&              a0,
                  QS_Sbte2Eus&              a1,
                  QS_Sbte2Eus&              a2,
                  QS_Sbte2Eus&              a3);
    bool     getB(bool                      get4ports,
                  const Mme::EUSDescriptor& desc,
                  QS_Sbte2Eus&              b0,
                  QS_Sbte2Eus&              b1,
                  QS_Sbte2Eus&              b2,
                  QS_Sbte2Eus&              b3);

    bool opA4ports(const Gaudi2::Mme::MmeRouting routing);
    bool opB4ports(const Gaudi2::Mme::MmeRouting routing);
};
} // namespace Mme
} // namespace Gaudi2
