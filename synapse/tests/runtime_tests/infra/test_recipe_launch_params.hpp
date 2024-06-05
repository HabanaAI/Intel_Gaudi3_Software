#pragma once

#include "synapse_api.h"
#include "../infra/test_types.hpp"
#include "syn_test_filter_factory.hpp"

#include "test_host_buffer_alloc.hpp"
#include "test_tensor_memory.hpp"

#include <cstdint>
#include <deque>
#include <initializer_list>
#include <limits>
#include <vector>

// Per launch
typedef std::vector<synLaunchTensorInfoExt> SynLaunchTensorInfoVec;

class RecipeLaunchParams
{
public:
    RecipeLaunchParams(TestDeviceBufferAlloc&& rWorkspace,
                       LaunchTensorMemory&&    rLaunchTensorMemory,
                       SynLaunchTensorInfoVec& rSynLaunchTensorInfoVec);

    void clear();

    uint64_t getWorkspace() const;

    const TestHostBuffer&        getHostInput(unsigned tensorIndex) const;
    const TestDeviceBufferAlloc& getDeviceInput(unsigned tensorIndex) const;

    const TestHostBuffer&        getHostOutput(unsigned tensorIndex) const;
    const TestDeviceBufferAlloc& getDeviceOutput(unsigned tensorIndex) const;

    const LaunchTensorMemory& getLaunchTensorMemory() const;

    const SynLaunchTensorInfoVec& getSynLaunchTensorInfoVec() const;

private:
    TestDeviceBufferAlloc  m_workspace;
    LaunchTensorMemory     m_launchTensorMemory;
    SynLaunchTensorInfoVec m_synLaunchTensorInfoVec;
};

typedef std::vector<RecipeLaunchParams> RecipeLaunchParamsVec;