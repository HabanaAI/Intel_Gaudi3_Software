#include "test_recipe_launch_params.hpp"

void ASSERT_LT_TENSOR(unsigned tensorIndex, const TensorMemoryVec& tensorMemoryVec)
{
    ASSERT_LT(tensorIndex, tensorMemoryVec.size());
}

RecipeLaunchParams::RecipeLaunchParams(TestDeviceBufferAlloc&& rWorkspace,
                                       LaunchTensorMemory&&    rLaunchTensorMemory,
                                       SynLaunchTensorInfoVec& rSynLaunchTensorInfoVec)
: m_workspace(std::move(rWorkspace)),
  m_launchTensorMemory(std::move(rLaunchTensorMemory)),
  m_synLaunchTensorInfoVec(rSynLaunchTensorInfoVec)
{
}

void RecipeLaunchParams::clear()
{
    m_launchTensorMemory.m_tensorInfoVecInputs.clear();
    m_launchTensorMemory.m_tensorInfoVecOutputs.clear();
    m_synLaunchTensorInfoVec.clear();
}

uint64_t RecipeLaunchParams::getWorkspace() const
{
    return m_workspace.getBuffer();
}

const TestHostBuffer& RecipeLaunchParams::getHostInput(unsigned tensorIndex) const
{
    ASSERT_LT_TENSOR(tensorIndex, m_launchTensorMemory.m_tensorInfoVecInputs);
    return m_launchTensorMemory.m_tensorInfoVecInputs[tensorIndex].getTestHostBuffer();
}

const TestDeviceBufferAlloc& RecipeLaunchParams::getDeviceInput(unsigned tensorIndex) const
{
    ASSERT_LT_TENSOR(tensorIndex, m_launchTensorMemory.m_tensorInfoVecInputs);
    return m_launchTensorMemory.m_tensorInfoVecInputs[tensorIndex].getTestDeviceBuffer();
}

const TestHostBuffer& RecipeLaunchParams::getHostOutput(unsigned tensorIndex) const
{
    ASSERT_LT_TENSOR(tensorIndex, m_launchTensorMemory.m_tensorInfoVecOutputs);
    return m_launchTensorMemory.m_tensorInfoVecOutputs[tensorIndex].getTestHostBuffer();
}

const TestDeviceBufferAlloc& RecipeLaunchParams::getDeviceOutput(unsigned tensorIndex) const
{
    ASSERT_LT_TENSOR(tensorIndex, m_launchTensorMemory.m_tensorInfoVecOutputs);
    return m_launchTensorMemory.m_tensorInfoVecOutputs[tensorIndex].getTestDeviceBuffer();
}

const LaunchTensorMemory& RecipeLaunchParams::getLaunchTensorMemory() const
{
    return m_launchTensorMemory;
}

const SynLaunchTensorInfoVec& RecipeLaunchParams::getSynLaunchTensorInfoVec() const
{
    return m_synLaunchTensorInfoVec;
}
