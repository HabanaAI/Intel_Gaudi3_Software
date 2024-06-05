#pragma once
#include <stdint.h>

namespace Gaudi2
{
namespace Mme
{
struct MetaData
{
    typedef enum
    {
        e_bgemm_en = (1 << 0),
        e_dec_en   = (1 << 1),

        e_opcode_bypass   = 0,
        e_opcode_sparsity = e_dec_en,
        e_opcode_bgemm    = e_dec_en | e_bgemm_en,
    } EDecOpcode;

    typedef enum
    {
        e_mme_size_8bits  = 1,
        e_mme_size_16bits = 2,
    } EMmeTEDataType;

    union AWUSER
    {
        struct
        {
            uint32_t resereved1 : 13; // 0-12
            uint32_t rmw0 : 9; // 13-21
            uint32_t reserved2 : 4; // 22-25
            uint32_t max_0_add : 1; // 26
            uint32_t reserved3 : 5; // 27-31
        };
        uint32_t dw;
    };

    struct SB
    {
        struct TE
        {
            struct DEC
            {
                struct Instruction
                {
                    EDecOpcode opcode : 2;
                    uint16_t   size : 6;
                };

                Instruction instr;
            };

            struct
            {
                uint32_t       transEn : 1;
                uint32_t       wrNum : 7;
                uint32_t       rdNum : 7;
                EMmeTEDataType dataType : 2;
                uint32_t       sparsity : 1;
            };
            DEC dec;
        };

        uint32_t padValue;
        TE       te;
        struct
        {
            uint64_t mpad : 8;
            uint64_t lpad : 8;
            uint64_t repeat : 8;
            uint64_t rlTokens : 8;
            uint64_t perfEvtCtx : 16;
            uint64_t fenceReq0 : 1;
            uint64_t fenceReq1 : 1;
            AWUSER   awuserRMW;
            uint64_t convEnd : 1;
            uint64_t cacheInvalidate : 1;
            uint64_t cacheDisable : 1;
            uint64_t perfEvetStart : 1;
            uint64_t perfEvetEnd : 1;
            uint64_t noData : 1;
            uint64_t duplicate : 1;
            uint64_t colorSet0 : 1;
            uint64_t colorSet1 : 1;
            uint64_t lastInDesc : 1;
            uint64_t spare : 16;
        };
        Gaudi2::Mme::MmeSyncObject so;
        unsigned                   cmdIdx;
        uint64_t                   addrId;
    };

    SB sb;
};

} // namespace Mme
} // namespace Gaudi2
