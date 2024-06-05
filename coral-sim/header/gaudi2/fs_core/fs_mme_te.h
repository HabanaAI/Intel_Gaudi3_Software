#pragma once
#include <stdint.h>

#include <string>

#include "gaudi2/mme.h"

#include "fs_mme_md.h"
#include "fs_mme_queue.h"
#include "fs_mme_queue_structs.h"
#include "fs_mme_thread.h"
#include "fs_mme_utils.h"

namespace Gaudi2
{
namespace Mme
{
class TE : public Gaudi2::Mme::Thread
{
   public:
    static const unsigned c_te_height = Gaudi2::Mme::c_cl_size;
    static const unsigned c_te_width  = Gaudi2::Mme::c_cl_size;

    TE(FS_Mme* mme = nullptr, const std::string& name = std::string());

    virtual ~TE() override {}

    void setConnectivity(Gaudi2::Mme::EMmeInputOperand               operandType,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sb2Te>*  sb2teQueue,
                         Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Te2Dec>* te2decQueue);

   protected:
    virtual void execute() override;

   private:
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Sb2Te>*  m_sb2teQueue;
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Te2Dec>* m_te2decQueue;

    Gaudi2::Mme::EMmeInputOperand m_operandType;
    uint8_t                       m_buffer[c_te_height][c_te_width];
    uint8_t                       m_outBuffer[c_te_height][c_te_width];
    Gaudi2::Mme::MetaData::SB::TE m_metaData;
    Gaudi2::Mme::MetaData::SB::TE m_outMetaData;
    uint64_t                      m_rdIdx;
    uint64_t                      m_wrIdx;
    bool                          m_packetPending;
    bool                          m_outPacketPending;
    uint64_t                      m_addrId;
    uint64_t                      m_outAddrId;

    bool readFromInput(unsigned& availInputs);

    bool writeToOutput(unsigned& availOutputSpace);

    void prepareNewOutput();

    uint32_t getPacketSize(bool outSize, const Gaudi2::Mme::MetaData::SB::TE& md);
};
} // namespace Mme
} // namespace Gaudi2
