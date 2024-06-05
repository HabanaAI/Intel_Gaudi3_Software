#pragma once

#include <stdint.h>

#include <cstring>
#include <list>
#include <string>

#include "gaudi2/mme.h"

#include "fs_assert.h"
#include "fs_mme_dataq.h"
#include "fs_mme_md.h"
#include "fs_mme_queue.h"
#include "fs_mme_queue_structs.h"
#include "fs_mme_thread.h"
#include "fs_mme_utils.h"

namespace Gaudi2
{
namespace Mme
{
class DEC : public Gaudi2::Mme::Thread
{
   public:
    static const unsigned c_max_cl_for_8b           = 7;
    static const unsigned c_max_cl_for_16b          = 15;
    static const unsigned c_max_cl_for_32b          = 31;
    static const unsigned c_nz_size_bytes_for_8b    = 16;
    static const unsigned c_nz_size_bytes_for_16b   = 8;
    static const unsigned c_nz_size_bytes_for_32b   = 4;
    static const unsigned c_nz_offset_bytes_for_8b  = 16;
    static const unsigned c_nz_offset_bytes_for_16b = 8;
    static const unsigned c_nz_offset_bytes_for_32b = 4;

    DEC(FS_Mme* mme = nullptr, const std::string& name = std::string());

    virtual ~DEC() override {}

    void setConnectivity(Gaudi2::Mme::EMmeInputOperand                operandType,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Te2Dec>*  te2decQueue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Dec2Eus>* dec2eusQueue);

   protected:
    virtual void execute() override;

   private:
    typedef enum
    {
        e_state_header,
        e_state_data_stream,
    } EPacketDecoderState;

    typedef enum
    {
        e_decomp_mode_none = 0,
        e_decomp_mode_8b   = 1,
        e_decomp_mode_16b  = 2,
        e_decomp_mode_32b  = 3,
    } ECompressionMode;

    typedef struct
    {
        uint8_t data[Gaudi2::Mme::c_cl_size];
    } CacheLine;

    union NonZeros
    {
        uint8_t nz8[c_max_cl_for_8b * c_nz_size_bytes_for_8b + 12]; // NZ for 8-bit mode (7*16 = 112B)
        uint8_t nz16[c_max_cl_for_16b * c_nz_size_bytes_for_16b + 4]; // NZ for 16-bit mode (15*8 = 120B)
        uint8_t nz32[c_max_cl_for_32b * c_nz_size_bytes_for_32b]; // NZ for 32-bit mode (31*4 = 124B)
        uint8_t data[124];
    };
    struct DecHeader
    {
        ECompressionMode mode;
        uint8_t          cheight; // # of CL of compressed (input) data (stream-size)
        uint8_t          height; // # of CL of uncompressed (output) data (dense-size)
        NonZeros         nz; // Non-zeros
        uint32_t         streamPtr;
    };

    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Te2Dec>*  m_te2decQueue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Dec2Eus>* m_dec2eusQueue;
    Gaudi2::Mme::EMmeInputOperand                m_operandType;
    EPacketDecoderState                          m_packetDecSt;
    DecHeader                                    m_header; // Header+Instruction Queue
    std::list<CacheLine>                         m_outBuffer;
    std::list<CacheLine>                         m_packetBuf;
    std::list<uint64_t>                          m_outAddrId;

    bool readFromInput(unsigned& availInputs);
    bool writeToOutput(unsigned& availOutputSpace);
    void headerDecoder(DecHeader& header, CacheLine& cl);
    template <typename TYPE>
    void rotateSparsity(const DecHeader& header);
    void rotateBgemm(const CacheLine& cl, const uint8_t size);
};
} // namespace Mme
} // namespace Gaudi2
