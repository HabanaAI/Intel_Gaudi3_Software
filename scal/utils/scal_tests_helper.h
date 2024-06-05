#include "gaudi3/asic_reg_structs/pdma_ch_a_regs.h"

static constexpr unsigned c_pdma_ch_block_size = 4 * 1024;

class PDMA_internals_helper
{
    public:
        PDMA_internals_helper() {}
        ~PDMA_internals_helper();

        void init_baseA_block(int fd, unsigned qid);
        void set_CI(unsigned ci);
        unsigned getQidFromChannelId(unsigned chid);
   protected:
        gaudi3::block_pdma_ch_a *m_chBlock = nullptr;
        unsigned m_allocatedSize;
};

