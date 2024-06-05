#pragma once
#include "runtime/qman/common/wcm/wcm_cs_querier_interface.hpp"

class WcmCsQuerier : public WcmCsQuerierInterface
{
public:
    WcmCsQuerier(int fd);

    virtual ~WcmCsQuerier() = default;

    virtual synStatus query(hlthunk_wait_multi_cs_in* inParams, hlthunk_wait_multi_cs_out* outParams) override;

    virtual void dump() const override;

    virtual void report() override;

private:
    struct WcmCsQuerierStats
    {
        uint64_t m_cntTotalQueries;
        uint64_t m_cntTotalCs;
        uint64_t m_cntTotalFailures;
        uint64_t m_cntTotalSuccesses;
        uint64_t m_cntTotalCompleted;
    };

    struct WcmCsQuerierStatsReported : public WcmCsQuerierStats
    {
        uint64_t m_cntTotalQueriesReported;
    };

    static void dumpQueryStatistics(const WcmCsQuerierStats& rStats);

    const int      m_fd;
    const uint64_t m_reportAmount;

    WcmCsQuerierStatsReported m_stats;
};