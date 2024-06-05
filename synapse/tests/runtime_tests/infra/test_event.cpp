#include "test_event.hpp"
#include "synapse_api.h"

TestEvent::TestEvent(synEventHandle eventHandle) : m_eventHandle(eventHandle) {}

TestEvent::TestEvent(TestEvent&& other) : m_eventHandle(other.m_eventHandle)
{
    other.m_eventHandle = nullptr;
}

TestEvent::~TestEvent()
{
    try
    {
        destroy();
    }
    catch (...)
    {
    }
}

void TestEvent::synchronize() const
{
    ASSERT_NE(m_eventHandle, nullptr) << "Invalid event handle";
    const synStatus status = synEventSynchronize(m_eventHandle);
    ASSERT_EQ(status, synSuccess) << "synEventSynchronize failed";
}

void TestEvent::query(synStatus& status) const
{
    ASSERT_NE(m_eventHandle, nullptr) << "Invalid event handle";
    status = synEventQuery(m_eventHandle);
}

void TestEvent::mapTensor(size_t                        mappingAmount,
                          const synLaunchTensorInfoExt* mappedLaunchTensorsInfo,
                          const synRecipeHandle&        recipeHandle)
{
    synStatus status = synEventMapTensorExt(&m_eventHandle, mappingAmount, mappedLaunchTensorsInfo, recipeHandle);
    ASSERT_EQ(status, synSuccess) << "Failed to map event tensor";
}

void TestEvent::destroy()
{
    if (m_eventHandle != nullptr)
    {
        const synStatus status = synEventDestroy(m_eventHandle);
        ASSERT_EQ(status, synSuccess) << "synEventDestroy failed";
        m_eventHandle = nullptr;
    }
}

synStatus TestEvent::query() const
{
    if (m_eventHandle != nullptr)
    {
        return synEventQuery(m_eventHandle);
    }
    return synFail;
}