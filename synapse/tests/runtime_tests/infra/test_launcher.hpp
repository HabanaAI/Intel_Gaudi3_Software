#pragma once

#include "synapse_api.h"
#include "../infra/test_types.hpp"
#include "syn_test_filter_factory.hpp"

#include "test_device_buffer_alloc.hpp"
#include "test_host_buffer_alloc.hpp"
#include "test_tensor_info.hpp"
#include "test_recipe_launch_params.hpp"
#include "test_tensor_init.hpp"
#include "test_tensor_dimensions.hpp"

#include <cstdint>
#include <deque>
#include <initializer_list>
#include <limits>
#include <vector>

class TestDevice;
class TestEvent;
class TestStream;
class TestRecipeInterface;

class TestLauncher
{
public:
    TestLauncher(const TestDevice& rTestDevice) : m_rTestDevice(rTestDevice) {}
    ~TestLauncher() = default;

    static void download(const TestStream&          rStream,
                         const TestRecipeInterface& rRecipe,
                         const RecipeLaunchParams&  rRecipeLaunchParams);

    static void launch(const TestStream&          rStream,
                       const TestRecipeInterface& rRecipe,
                       const RecipeLaunchParams&  rRecipeLaunchParams);

    static void launchWithExternalEvents(const TestStream&          rStream,
                                         const TestRecipeInterface& rRecipe,
                                         const RecipeLaunchParams&  rRecipeLaunchParams,
                                         std::vector<TestEvent>&    sfgEvents);

    static void upload(const TestStream&          rStream,
                       const TestRecipeInterface& rRecipe,
                       const RecipeLaunchParams&  rRecipeLaunchParams);

    static void execute(const TestStream&          rStream,
                        const TestRecipeInterface& rRecipe,
                        const RecipeLaunchParams&  rRecipeLaunchParams);

    synStatus downloadConstSections(const TestStream&                rStream,
                                    const TestRecipeInterface&       rRecipe,
                                    const std::vector<synSectionId>& rConstSectionsIdDB,
                                    AllocDeviceBuffersVec&           rmanagedDeviceBuffers,
                                    std::vector<uint64_t>&           rConstSectionsHostAddresses);

    RecipeLaunchParams createRecipeLaunchParams(const TestRecipeInterface&   rRecipe,
                                                TensorInitInfo               tensorInitInfo,
                                                const TestTensorsDimensions& rTestTensorsDimensions = {});

    RecipeLaunchParams createRecipeLaunchParams(const TestRecipeInterface&   rRecipe,
                                                const AllocDeviceBuffersVec& constSectionsDeviceBuffers,
                                                TensorInitInfo               tensorInitInfo,
                                                const TestTensorsDimensions& rTestTensorsDimensions = {});

    void generateLaunchTensorsWithTensorsMemory(const TestRecipeInterface& rRecipe,
                                                const LaunchTensorMemory&  rLaunchTensorMemory,
                                                SynLaunchTensorInfoVec&    rSynLaunchTensorInfoVec,
                                                TensorInitInfo             tensorInitInfo);

private:
    static void generateLaunchTensors(const TestRecipeInterface&   rRecipe,
                                      const LaunchTensorMemory&    rLaunchTensorMemory,
                                      SynLaunchTensorInfoVec&      rSynLaunchTensorInfoVec,
                                      const AllocDeviceBuffersVec& constSectionsDeviceBuffers,
                                      TensorInitInfo               tensorInitInfo,
                                      const TestTensorsDimensions& rTestTensorsDimensions);

    void allocateMemory(const TestRecipeInterface&   rRecipe,
                        LaunchTensorMemory&            rLaunchTensorMemory,
                        const TestTensorsDimensions&   rTestTensorsDimensions) const;

    void allocateSectionMemory(const TestRecipeInterface&     rRecipe,
                               LaunchTensorMemory&            rLaunchTensorMemory) const;

    const TestDevice& m_rTestDevice;
};
