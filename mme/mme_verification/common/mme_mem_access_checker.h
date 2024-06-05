#pragma once

#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_set>
#include "sync/overlap.h"
#include "fs_core/fs_mme_mem_access.h"

namespace MmeCommon
{
class MmeMemAccessChecker : public MemAccess
{
public:
    MmeMemAccessChecker();
    virtual ~MmeMemAccessChecker();

    void write(const uint64_t base, const uint64_t size, const unsigned cmdIdx, const unsigned signal) override;
    void read(const uint64_t base, const uint64_t size, const unsigned cmdIdx) override;
    void signal(const unsigned colorSetIdx) override;

    void addDesc(const OverlapDescriptor& desc, unsigned descColor = 0, bool ofNullDesc = false);
    static const unsigned COLORS_NR = 2;
    using signalsCntArr = std::array<unsigned, COLORS_NR>;

protected:
    static const unsigned REMOTE = 1;
    static const unsigned LOCAL = 0;

    struct CmdColorSetMemAccessTracker
    {
        bool valid;
        unsigned signalBase;
        unsigned signalsNr;
        unsigned maxRelSignal;
        std::map<std::thread::id, SegmentsSpace<unsigned>> readAccess;
        std::map<std::thread::id, SegmentsSpace<unsigned>> writeAccess;
        SegmentsSpace<unsigned> allowedReadAccess;
        SegmentsSpace<unsigned> expectedReadAccess;
        SegmentsSpace<unsigned> expectedWriteAccess;
    };

    struct CmdMemAccessTracker
    {
        unsigned cmdIdx;
        bool ofNullDesc=false;
        struct CmdColorSetMemAccessTracker colors[2]; // remote and local
    };

    class AULock
    {
    public:
        AULock() :  m_pendingAdminNr(0), m_userNr(0), m_activeAdmin(false) {}
        void userBegin();
        void userEnd();
        void adminBegin();
        void adminEnd();

    private:
        std::mutex m_mutex;
        std::condition_variable m_cond;
        unsigned m_pendingAdminNr;
        unsigned m_userNr;
        bool m_activeAdmin;
    };

    template <int IS_WRITE>
    static void mergeAndNormThreads(
        CmdMemAccessTracker* tracker,
        const unsigned color,
        SegmentsSpace<unsigned> *segSpace);

    static void compareSegSpacesRead(
        SegmentsSpace<unsigned> *actual,
        const SegmentsSpace<unsigned> *expected,
        const SegmentsSpace<unsigned> *allowed);

    static void compareSegSpacesWrite(const SegmentsSpace<unsigned>* actualWriteSegment,
                                      const SegmentsSpace<unsigned>* expectedWriteSegment);

    static void testTracker(CmdMemAccessTracker* tracker, const unsigned color);

    template <int ALLOW_OVERLAP, int IS_INPUT>
    static void addSubRoisToTracker(
        const std::list<OverlapRoi> *rois,
        CmdMemAccessTracker * tracker,
        const unsigned descColor);

    std::pair<SegmentsSpace<unsigned> *, unsigned> getAccessSegSpace(
        const unsigned cmdIdx,
        uint64_t base,
        uint64_t size,
        const bool isWrite);

    unsigned calcSignalsNr(unsigned colorSetIdx) const;
    void gatherAllSubRoisToOpenReads(const OverlapDescriptor& desc,
                                     std::map<unsigned int, CmdMemAccessTracker>::mapped_type& newTracker,
                                     unsigned descColor);

    void initTracker(const OverlapDescriptor& desc,
                     unsigned int cmdIdx,
                     std::map<unsigned int, CmdMemAccessTracker>::mapped_type& newTracker,
                     bool ofNullDesc = false);
    void mergeAssignedSubRois(unsigned int color, unsigned int nextSignal);

    unsigned m_signalsCtr[2];
    AULock m_auLock;
    std::map<unsigned, CmdMemAccessTracker> m_cmd2Trackers;
    std::unordered_set<unsigned> m_usedCmdIds;
    std::multimap<unsigned, CmdMemAccessTracker*> m_desc2Trackers[2];
    SegmentsSpace<unsigned> openReads[2];
    unsigned m_numOfDesc = 0;
    std::array<unsigned, 2> m_segmentTracker = {0};
};
static MmeMemAccessChecker::signalsCntArr segmentCnt = {0, 0};
}  // namespace MmeCommon