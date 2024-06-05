#pragma once

#include "queue_interface.hpp"
#include "basic_queue_info.hpp"

class QueueBase : public QueueInterface
{
public:
    QueueBase(const BasicQueueInfo& rBasicQueueInfo);

    virtual ~QueueBase();

    virtual const BasicQueueInfo& getBasicQueueInfo() const override { return m_basicQueueInfo; }

    // HCL
    virtual synStatus createHclStream() override { return synUnsupported; }

    virtual synStatus destroyHclStream() override { return synUnsupported; }

    virtual hcl::hclStreamHandle getHclStreamHandle() const override { return nullptr; }

protected:
    const BasicQueueInfo m_basicQueueInfo;
};
