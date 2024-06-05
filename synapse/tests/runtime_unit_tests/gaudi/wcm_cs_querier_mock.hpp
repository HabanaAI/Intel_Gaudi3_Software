#pragma once
#include "runtime/qman/common/wcm/wcm_cs_querier_interface.hpp"
#include "synapse_common_types.h"
#include "hlthunk.h"
#include <future>
#include <set>
#include <array>
#include <deque>

class WcmCsQuerierCheckerMock : public WcmCsQuerierInterface
{
public:
    WcmCsQuerierCheckerMock(const std::deque<std::array<std::set<uint64_t>, 32>>& rQueryPermutations);

    virtual synStatus query(hlthunk_wait_multi_cs_in* inParams, hlthunk_wait_multi_cs_out* outParams) override;

    virtual void dump() const override {};

    virtual void report() override {};

    std::deque<std::array<std::set<uint64_t>, 32>> m_queryPermutations;
};

class WcmCsQuerierIncompleteMock : public WcmCsQuerierInterface
{
public:
    WcmCsQuerierIncompleteMock(uint64_t csHandle, uint64_t queryAmount);

    virtual synStatus query(hlthunk_wait_multi_cs_in* inParams, hlthunk_wait_multi_cs_out* outParams) override;

    virtual void dump() const override {};

    virtual void report() override {};

    const uint64_t    m_csHandle;
    uint64_t          m_queryAmount;

    enum CsStatus
    {
        PENDING_FIRST_QUERY,
        RETRYING,
        PENDING_LAST_QUERY,
        NA
    };

    CsStatus m_csStatus;
};

class WcmCsQuerierStatusMock : public WcmCsQuerierInterface
{
public:
    WcmCsQuerierStatusMock(uint64_t queryAmount, synStatus status);
    ~WcmCsQuerierStatusMock();

    virtual synStatus query(hlthunk_wait_multi_cs_in* inParams, hlthunk_wait_multi_cs_out* outParams) override;

    virtual void dump() const override {};

    virtual void report() override {};

    uint64_t          m_queryAmount;
    synStatus         m_status;
};

class WcmCsQuerierRecorderMock : public WcmCsQuerierInterface
{
public:
    WcmCsQuerierRecorderMock(unsigned queryAmount = 1);

    virtual synStatus query(hlthunk_wait_multi_cs_in* inParams, hlthunk_wait_multi_cs_out* outParams) override;

    virtual void dump() const override {};

    virtual void report() override {};

    std::promise<hlthunk_wait_multi_cs_in> m_inParams;
    unsigned                               m_queryAmount;
};

class WcmCsQuerierPause2Mock : public WcmCsQuerierInterface
{
public:
    WcmCsQuerierPause2Mock(std::array<std::future<hlthunk_wait_multi_cs_out>, 2>& rQueryPausers);

    virtual synStatus query(hlthunk_wait_multi_cs_in* inParams, hlthunk_wait_multi_cs_out* outParams) override;

    virtual void dump() const override {};

    virtual void report() override {};

    std::array<std::future<hlthunk_wait_multi_cs_out>, 2>& m_rQueryPausers;
    std::promise<hlthunk_wait_multi_cs_in>                 m_inParams[2];
    unsigned                                               queryIndex;
};