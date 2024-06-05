#pragma once

#include <stdint.h>

#include <list>

#include "gaudi2/mme.h"

#include "fs_mme_queue.h"
#include "fs_mme_queue_structs.h"
#include "fs_mme_thread.h"
#include "fs_mme_utils.h"

#include "mme_half.h"

namespace Gaudi2
{
namespace Mme
{
class AGU : public Gaudi2::Mme::Thread
{
   public:
    AGU(FS_Mme* mme = nullptr, const std::string& name = std::string());

    virtual ~AGU() override {}

    void     setConnectivity(Gaudi2::Mme::EMmeBrainIdx aguId, Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Agu2Sb>* agu2sbQueue);
    uint64_t getFenceCtr(const unsigned idx) const { return m_fenceCtrs[idx]; }

    void genAddresses(const Gaudi2::Mme::Desc* desc);

   protected:
    virtual void execute() override;

   private:
    static constexpr unsigned int c_max_agu_inc = 1;

    struct FixedParams
    {
        Gaudi2::Mme::MmeTensorDesc    tensorDesc;
        unsigned                      spatialSizeMinus1;
        uint64_t                      baseAddr;
        uint64_t                      baseAddr1; // for Cout only
        int                           loopMask;
        int                           accumMask;
        int                           signalMask0;
        int                           signalMask1;
        int                           partialHeightLoopMask;
        int                           fcdLoopMask;
        int                           logElementSize;
        int                           height;
        int                           heightLast;
        unsigned                      fcd;
        unsigned                      fcdLast;
        bool                          lower;
        uint32_t                      paddingValue;
        unsigned                      loopsNum;
        Gaudi2::Mme::MmePerfEvt       perfEvt;
        uint16_t                      spare;
        Gaudi2::Mme::MmeUserData      usr;
        unsigned                      rlVal;
        Gaudi2::Mme::MetaData::SB::TE teMD;
        bool                          duplicate;
        int                           colorSet0;
        int                           colorSet1;
        Gaudi2::Mme::MmeSyncObject    syncObject;
        unsigned                      matSize;
        unsigned                      numOfTEBlocks;
        bool                          fp32nonTransWalk;
        bool                          noData;
        bool                          cacheDisable;
        uint8_t                       bgemmFp8Size;
        bool                          signalOnly;
    };

    struct SOInfo
    {
        uint32_t sync_object_address;
        uint32_t sync_object_data;
    };
    Gaudi2::Mme::Queue<Gaudi2::Mme::QS_Agu2Sb>* m_agu2sbQueue;
    Gaudi2::Mme::EMmeBrainIdx                   m_aguId;
    Gaudi2::Mme::EMmeInputOperand               m_operandType;
    std::list<Gaudi2::Mme::QS_Agu2Sb>           m_addrQ;
    Gaudi2_MME_Half::SB::PerfUoW*               m_perf_sb_uow = NULL;
    unsigned                                    m_descCnt;
    uint16_t                                    m_perfEvtCtr;
    uint64_t                                    m_fenceCtrs[2];
    uint64_t                                    m_currentAddrId = 0;

    inline unsigned getNumOfElementsInChannel(const Gaudi2::Mme::EMmeDataType dataType)
    {
        return (Mme::c_cl_size / getElementSize(dataType));
    }

    inline uint16_t getSpareBits(const Gaudi2::Mme::Desc* desc, const bool master)
    {
        uint16_t res;
        uint8_t  mstrIdx = !master;
        switch (m_aguId) {
            case e_mme_agu0_idx: res = desc->spare[mstrIdx].sb0; break;
            case e_mme_agu1_idx: res = desc->spare[mstrIdx].sb1; break;
            case e_mme_agu2_idx: res = desc->spare[mstrIdx].sb2; break;
            case e_mme_agu3_idx: res = desc->spare[mstrIdx].sb3; break;
            case e_mme_agu4_idx: res = desc->spare[mstrIdx].sb4; break;
            case e_mme_agu_cout0_idx:
            case e_mme_agu_cout1_idx: res = desc->spare[mstrIdx].Out; break;
            default: FS_ASSERT(0);
        }
        return res;
    }

    bool pushAddr2SbQ(uint32_t elmToKeep = 0);
    void genGemmAddresses(const FixedParams* fp,
                          const bool         advance,
                          const int64_t*     startOffsets, // The current dim offsets.
                          int64_t*           nextStartOffsets, // The next dim offsets.
                          const int64_t*     roiBase,
                          unsigned           loopStartMask,
                          unsigned           loopEndMask);
    void popAddr(Gaudi2::Mme::QS_Agu2Sb* paddr);
    void dumpDescIO(const Gaudi2::Mme::Desc* desc);
    void dumpSyncForNullDesc(const Gaudi2::Mme::MmeSyncObject* syncObject);
    void swapBaseAndOffset(Gaudi2::Mme::MmeAguCoreDesc* aguDesc, Gaudi2::Mme::MmeTensorDesc* tensorDesc);
    void getAssociatedDims(unsigned* associatedDim, const Gaudi2::Mme::Desc* desc);
    void getSlaveSyncObject(const Gaudi2::Mme::Desc* desc, Gaudi2::Mme::MmeSyncObject* syncObject);
};
} // namespace Mme
} // namespace Gaudi2
