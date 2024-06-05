#pragma once

#include "../infra/test_types.hpp"

class TestEvent final
{
public:
    TestEvent(synEventHandle eventHandle);
    TestEvent(const TestEvent&) = delete;
    TestEvent(TestEvent&& other);
    ~TestEvent();
    operator synEventHandle() const { return m_eventHandle; }

    void      synchronize() const;
    synStatus query() const;

    void query(synStatus& status) const;

    void mapTensor(size_t                        mappingAmount,
                   const synLaunchTensorInfoExt* mappedLaunchTensorsInfo,
                   const synRecipeHandle&        recipeHandle);

private:
    void destroy();

    synEventHandle m_eventHandle;
};
