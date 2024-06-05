#pragma once

#include "device_info_interface.hpp"

#include "synapse_common_types.h"

#include "runtime/scal/common/infra/scal_includes.hpp"
#include "runtime/scal/common/infra/scal_types.hpp"

#include <vector>
#include <atomic>

struct ScalEvent;
class ScalCompletionGroup;

namespace common
{
class ScalStream;
}

/*************************************************************/
/**************    ScalCompletionGroupBase    ****************/
/*************************************************************/
class ScalCompletionGroupBase
{
public:
    ScalCompletionGroupBase(scal_handle_t                      devHndl,
                            const std::string&                 name,
                            const common::DeviceInfoInterface* pDeviceInfoInterface);

    virtual ~ScalCompletionGroupBase() = default;

    // Virtual impl methods
    virtual synStatus init();

    virtual uint32_t   getIndexInScheduler() const { return 0; };
    virtual bool       getCgTdrInfo(const CgTdrInfo*& cgTdrInfo) const;
    virtual TdrRtn     tdr(TdrType tdrType);
    const std::string& getName() const { return m_name;}

    uint64_t getLongSoAddress();

    // local methods
    ScalLongSyncObject getIncrementedLongSo(bool isUserReq, uint64_t targetOffset = 1);

    ScalLongSyncObject getTargetLongSo(uint64_t targetOffset) const;

    synStatus
    longSoWait(const ScalLongSyncObject& rLongSo, uint64_t timeoutMicroSec, bool alwaysWaitForInterrupt = false) const;

    synStatus longSoWaitForLast(bool isUserReq, uint64_t timeoutMicroSec) const;

    ScalLongSyncObject getLastTarget(bool isUserReq) const;

    synStatus getCurrentCgInfo(scal_completion_group_infoV2_t& cgInfo);

    bool isForceOrdered();

    void longSoRecord(bool isUserReq, ScalLongSyncObject& rLongSo) const;

    synStatus eventRecord(bool isUserReq, ScalEvent& scalEvent) const;

    // inline methods
    inline uint64_t getCompletionTarget() const { return m_completionTarget; }

    inline const scal_completion_group_infoV2_t& getCgInfo() const { return m_cgInfo; }

protected:
    virtual std::string _getAdditionalPrintInfo() const = 0;

    void _setExpectedCounter() const;

    const scal_handle_t            m_devHndl;
    const std::string              m_name;
    scal_comp_group_handle_t       m_cgHndl;
    scal_completion_group_infoV2_t m_cgInfo;
    std::atomic<uint64_t>          m_completionTarget;
    std::atomic<uint64_t>          m_lastUserCompletionTarget;
    CgTdrInfo                      m_tdrInfo;


private:
    const common::DeviceInfoInterface* m_pDeviceInfoInterface;
};
