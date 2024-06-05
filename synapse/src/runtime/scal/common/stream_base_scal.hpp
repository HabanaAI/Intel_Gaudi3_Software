#pragma once

#include "define_synapse_common.hpp"

#include "runtime/common/queues/queue_base.hpp"

#include "runtime/scal/common/infra/scal_types.hpp"

#include <mutex>

class ScalStreamCopyInterface;

class QueueBaseScal : public QueueBase
{
public:
    QueueBaseScal(const BasicQueueInfo& rBasicQueueInfo);

    virtual ~QueueBaseScal() = default;

    virtual synStatus eventQuery(const EventInterface& rEventInterface) = 0;

    virtual synStatus eventSynchronize(const EventInterface& rEventInterface) = 0;
};

class QueueBaseScalCommon : public QueueBaseScal
{
public:
    QueueBaseScalCommon(const BasicQueueInfo& rBasicQueueInfo, ScalStreamCopyInterface* scalStream);

    virtual ~QueueBaseScalCommon() = default;

    virtual uint32_t getPhysicalQueueOffset() const override { return 0; }

    inline ScalStreamCopyInterface* getScalStream() { return m_scalStream; }

    virtual synStatus eventQuery(const EventInterface& rEventInterface) override;

    virtual synStatus eventSynchronize(const EventInterface& rEventInterface) override;

    virtual synStatus waitForLastLongSo(bool isUserReq);

    virtual synStatus addCompletionAfterWait();

    TdrRtn tdr(TdrType tdrType);

    virtual void finalize() override {};

    void dfaInfo(DfaReq dfaReq, uint64_t csSeq) override;

protected:
    ScalStreamCopyInterface* m_scalStream;

    std::timed_mutex m_userOpLock;

private:
    virtual void dfaUniqStreamInfo(bool               showOldestRecipeOnly,
                                   uint64_t           currentLongSo,
                                   bool               dumpRecipe,
                                   const std::string& callerMsg);  // overridden for compute only

    virtual std::set<ScalStreamCopyInterface*> dfaGetQueueScalStreams() = 0;
};
