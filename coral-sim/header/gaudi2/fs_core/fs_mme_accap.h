#pragma once
#include <stdint.h>

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "gaudi2/mme.h"

#include "cbb.h"
#include "fs_mme_md.h"
#include "fs_mme_queue.h"
#include "fs_mme_queue_structs.h"
#include "fs_mme_thread.h"
#include "fs_mme_utils.h"
#include "lbw_hub.h"
#include "specs.h"

namespace Gaudi2
{
namespace Mme
{
class AccAp : public Gaudi2::Mme::Thread, public Cbb_Base
{
   public:
    static const unsigned c_acc_height      = 128;
    static const unsigned c_acc_width       = 256;
    static const unsigned c_mme_num_of_lfsr = Gaudi2::Mme::c_mme_lfsr_seeds_nr;
    static const unsigned c_mme_num_of_wb   = 2;

    AccAp(coral_module_name m_name, FS_Mme* mme = nullptr, const std::string& name = std::string());

    virtual ~AccAp() override
    {
        delete m_accMem;
        delete m_rollupFromEus;
    }
    void reset();
    void setConnectivity(Gaudi2::Mme::Queue<Gaudi2::Mme::QS_AccApBrain>* accapBrainQueue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Eus2Acc>*    eus2acc,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Ap2Wbc>*     ap2wbc0Queue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Ap2Wbc>*     ap2wbc1Queue);

    inline void setLfsrSeed(uint32_t idx, uint32_t val) { m_lfsr[idx] = val; }
    inline uint32_t getLfsrSeed(uint32_t idx) { return m_lfsr[idx]; }

    inline void setLfsrPolynom(uint32_t val) { m_lfsrPolynom = val; }

    struct cacheLine
    {
        int8_t data[Gaudi2::Mme::c_cl_size];
    };

    void get_coresight_params(uint8_t& stream_id, bool& stm_en);
    void connect_to_lbw(LBWHub* lbw, Specs* specs, uint32_t itr) override;

   protected:
    virtual void execute() override;

   private:
    typedef union
    {
        float acc2x[Gaudi2::Mme::c_mme_accums_nr][c_acc_height][c_acc_width];
        float acc[Gaudi2::Mme::c_mme_2x_accums_nr][c_acc_height / 2][c_acc_width];
    } AccMem;

    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_AccApBrain>* m_accapBrainQueue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Eus2Acc>*    m_eus2acc;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Ap2Wbc>*     m_ap2wbc0Queue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Ap2Wbc>*     m_ap2wbc1Queue;

    uint32_t                          m_lfsr[c_mme_num_of_lfsr];
    uint32_t                          m_lfsrPolynom;
    Gaudi2::Mme::ACCAPDescriptor      m_currDesc;
    AccMem*                           m_accMem;
    unsigned                          m_accLineIdx;
    std::vector<std::vector<int32_t>> m_currAccLine;
    std::queue<cacheLine>             m_CoutQ[c_mme_num_of_wb];
    Gaudi2::Mme::QS_Eus2Acc*          m_rollupFromEus;
    unsigned                          m_outCnt[c_mme_num_of_wb];
    uint64_t                          ucmd_id  = -1;
    uint64_t                          cmd_id   = -1;
    uint8_t                           accum_id = -1;

    bool writeToOutput(const unsigned wbIdx, unsigned& availOutputSpace);

    bool write2acc(const Gaudi2::Mme::ACCAPDescriptor& desc, const uint8_t& accIdx);

    void resizeAccLine();

    void prepareNewOutput();

    void prepareAccLine(const uint8_t& accIdx);

    void activationPipe();

    void doRelu();
};
} // namespace Mme
} // namespace Gaudi2
