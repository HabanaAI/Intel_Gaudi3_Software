#include "test_launcher.hpp"

#include "filesystem.h"

#include "test_recipe_interface.hpp"

#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "test_tensor_init.hpp"
#include "../infra/test_types.hpp"

#include "test_device.hpp"
#include "test_event.hpp"
#include "test_utils.h"
#include "test_stream.hpp"

#include <cstdint>
#include <string>


void TestLauncher::download(const TestStream&          rStream,
                            const TestRecipeInterface& rRecipe,
                            const RecipeLaunchParams&  rRecipeLaunchParams)
{
    const uint64_t tensorInfoVecSizeInput = rRecipe.getTensorInfoVecSizeInput();
    uint64_t       src[tensorInfoVecSizeInput];
    uint64_t       size[tensorInfoVecSizeInput];
    uint64_t       dst[tensorInfoVecSizeInput];
    uint64_t       numCopies = 0;

    for (unsigned index = 0; index < tensorInfoVecSizeInput; index++)
    {
        if (rRecipe.isInputOnConstSection(index))
        {
            continue;
        }

        src[numCopies]  = (uint64_t)rRecipeLaunchParams.getHostInput(index).getBuffer();
        size[numCopies] = rRecipe.getTensorSizeInput(index);
        dst[numCopies]  = rRecipeLaunchParams.getDeviceInput(index).getBuffer();
        numCopies++;
    }

    rStream.memcopyAsyncMultiple(&src[0], &size[0], &dst[0], HOST_TO_DRAM, numCopies);
}

void TestLauncher::launch(const TestStream&          rStream,
                          const TestRecipeInterface& rRecipe,
                          const RecipeLaunchParams&  rRecipeLaunchParams)
{
    rStream.launch(rRecipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                   rRecipe.getTensorInfoVecSize(),
                   rRecipeLaunchParams.getWorkspace(),
                   rRecipe.getRecipe(),
                   0);
}

void TestLauncher::launchWithExternalEvents(const TestStream&          rStream,
                                            const TestRecipeInterface& rRecipe,
                                            const RecipeLaunchParams&  rRecipeLaunchParams,
                                            std::vector<TestEvent>&    sfgEvents)
{
    rStream.launchWithExternalEvents(rRecipeLaunchParams.getSynLaunchTensorInfoVec().data(),
                                     rRecipe.getTensorInfoVecSize(),
                                     rRecipeLaunchParams.getWorkspace(),
                                     rRecipe.getRecipe(),
                                     0 /* flags */,
                                     sfgEvents);
}

void TestLauncher::upload(const TestStream&          rStream,
                          const TestRecipeInterface& rRecipe,
                          const RecipeLaunchParams&  rRecipeLaunchParams)
{
    for (unsigned outputIndex = 0; outputIndex < rRecipe.getTensorInfoVecSizeOutput(); outputIndex++)
    {
        // Cannot be, but just in case...
        if (rRecipe.isOutputOnConstSection(outputIndex))
        {
            continue;
        }

        const auto& hostOutput   = rRecipeLaunchParams.getHostOutput(outputIndex);
        const auto& deviceOutput = rRecipeLaunchParams.getDeviceOutput(outputIndex);

        unsigned                      tensorSize = 1;
        const synLaunchTensorInfoExt& currentOutputLaunchTensorInfo =
            rRecipeLaunchParams.getSynLaunchTensorInfoVec().data()[rRecipe.getTensorInfoVecSizeInput() + outputIndex];
        for (unsigned i = 0; i < rRecipe.getTensorInfoOutput(outputIndex)->m_dimsAmount; i++)
        {
            tensorSize *= currentOutputLaunchTensorInfo.tensorSize[i];
        }
        tensorSize *= dataTypeSizeInBytes(rRecipe.getTensorInfoOutput(outputIndex)->m_dataType);

        rStream.memcopyAsync(deviceOutput.getBuffer(), tensorSize, (uint64_t)hostOutput.getBuffer(), DRAM_TO_HOST);
    }
}

void TestLauncher::execute(const TestStream&          rStream,
                           const TestRecipeInterface& rRecipe,
                           const RecipeLaunchParams&  rRecipeLaunchParams)
{
    download(rStream, rRecipe, rRecipeLaunchParams);
    launch(rStream, rRecipe, rRecipeLaunchParams);
    upload(rStream, rRecipe, rRecipeLaunchParams);
}

synStatus TestLauncher::downloadConstSections(const TestStream&                rStream,
                                              const TestRecipeInterface&       rRecipe,
                                              const std::vector<synSectionId>& rConstSectionsIdDB,
                                              AllocDeviceBuffersVec&           rAllocDeviceBuffers,
                                              std::vector<uint64_t>&           rConstSectionsHostAddresses)
{
    synStatus status(synSuccess);

    rConstSectionsHostAddresses.clear();

    // Mapping owner (will unmap upon return from method)
    MappedHostBuffersVec copyHostBuffers;

    std::vector<uint64_t> dstDeviceBuffers;
    std::vector<uint64_t> copySizes;
    for (synSectionId singleConstSectionId : rConstSectionsIdDB)
    {
        bool     isConst         = false;
        uint64_t sectionSize     = 0;
        uint64_t hostSectionData = 0;

        status = rRecipe.getConstSectionProperties(isConst, sectionSize, hostSectionData, singleConstSectionId);
        if (status != synSuccess)
        {
            // In case clearing the pointer, this will fail
            return status;
        }

        if (!isConst)
        {
            continue;
        }

        // Allocate memory on DRAM for the const section
        TestDeviceBufferAlloc constSectionDeviceAddress =
            m_rTestDevice.allocateDeviceBuffer(sectionSize, 0 /* flags */);
        dstDeviceBuffers.push_back(constSectionDeviceAddress.getBuffer());
        rAllocDeviceBuffers.push_back(std::move(constSectionDeviceAddress));

        // Map host buffer
        TestHostBufferMap mappedHostBuffer = m_rTestDevice.mapHostBuffer((void*)hostSectionData, sectionSize);
        copyHostBuffers.push_back(std::move(mappedHostBuffer));

        rConstSectionsHostAddresses.push_back(hostSectionData);
        copySizes.push_back(sectionSize);
    }

    // Copy all sections' data from host to device
    rStream.memcopyAsyncMultiple(rConstSectionsHostAddresses.data(),
                                 copySizes.data(),
                                 dstDeviceBuffers.data(),
                                 HOST_TO_DRAM,
                                 rConstSectionsHostAddresses.size());

    rStream.synchronize();

    return status;
}

RecipeLaunchParams TestLauncher::createRecipeLaunchParams(const TestRecipeInterface&   rRecipe,
                                                          TensorInitInfo               tensorInitInfo,
                                                          const TestTensorsDimensions& rTestTensorsDimensions)
{
    static const AllocDeviceBuffersVec defaultDeviceBuffers;

    return createRecipeLaunchParams(rRecipe, defaultDeviceBuffers, tensorInitInfo, rTestTensorsDimensions);
}

RecipeLaunchParams TestLauncher::createRecipeLaunchParams(const TestRecipeInterface&   rRecipe,
                                                          const AllocDeviceBuffersVec& constSectionsDeviceBuffers,
                                                          TensorInitInfo               tensorInitInfo,
                                                          const TestTensorsDimensions& rTestTensorsDimensions)
{
    TestDeviceBufferAlloc workspace;
    const uint64_t        workspaceSize = rRecipe.getWorkspaceSize();
    if (workspaceSize > 0)
    {
        workspace = m_rTestDevice.allocateDeviceBuffer(workspaceSize, 0);
    }

    LaunchTensorMemory launchTensorMemory;
    allocateMemory(rRecipe,
                   launchTensorMemory,
                   rTestTensorsDimensions);

    SynLaunchTensorInfoVec synLaunchTensorInfoVec;
    generateLaunchTensors(rRecipe,
                          launchTensorMemory,
                          synLaunchTensorInfoVec,
                          constSectionsDeviceBuffers,
                          tensorInitInfo,
                          rTestTensorsDimensions);

    RecipeLaunchParams recipeLaunchParams(std::move(workspace), std::move(launchTensorMemory), synLaunchTensorInfoVec);

    return recipeLaunchParams;
}

void TestLauncher::allocateSectionMemory(const TestRecipeInterface&     rRecipe,
                                         LaunchTensorMemory&            rLaunchTensorMemory) const
{
    const std::vector<TensorInfo>& rLaunchInfoInputs  = rRecipe.getTensorInfoVecInputs();
    const std::vector<TensorInfo>& rLaunchInfoOutputs = rRecipe.getTensorInfoVecOutputs();
    ASSERT_EQ(rLaunchTensorMemory.m_tensorInfoVecInputs.size(), 0) << "rLaunchInfoInputs is not empty";
    ASSERT_EQ(rLaunchTensorMemory.m_tensorInfoVecOutputs.size(), 0) << "m_tensorInfoVecOutputs is not empty";

    // Allocate memory for sections
    std::map<uint32_t, uint64_t>  sectionsSizes;
    // 1st compute sections sizes
    for (unsigned tensorIndex = 0; tensorIndex < rLaunchInfoInputs.size(); tensorIndex++)
    {
        TensorInfo tensorInfo(rLaunchInfoInputs[tensorIndex]);

        if ((tensorInfo.m_tensorFlags != (uint32_t)TensorFlag::CONST_SECTION))
        {
            unsigned sectionID = rRecipe.getSectionID(tensorInfo);
            if (sectionsSizes.count(sectionID) > 0) // key exists
            {
                sectionsSizes[sectionID] =
                    std::max(sectionsSizes[sectionID], tensorInfo.m_tensorSize + tensorInfo.m_sectionOffset);
            }
            else
            {
                sectionsSizes[sectionID] = tensorInfo.m_tensorSize + tensorInfo.m_sectionOffset;
            }
        }
    }
    for (unsigned tensorIndex = 0; tensorIndex < rLaunchInfoOutputs.size(); tensorIndex++)
    {
        TensorInfo tensorInfo(rLaunchInfoOutputs[tensorIndex]);
        if ((tensorInfo.m_tensorFlags != (uint32_t)TensorFlag::CONST_SECTION))
        {
            unsigned sectionID = rRecipe.getSectionID(tensorInfo);
            if (sectionsSizes.count(sectionID) > 0) // key exists
            {
                sectionsSizes[sectionID] =
                    std::max(sectionsSizes[sectionID], tensorInfo.m_tensorSize + tensorInfo.m_sectionOffset);
            }
            else
            {
                sectionsSizes[sectionID] = tensorInfo.m_tensorSize + tensorInfo.m_sectionOffset;
            }
        }
    }
    // 2nd - allocate the host and device buffers
    // Input and output need to be mapped to the device as they are copied from / to
    for (auto& item : sectionsSizes)
    {
        // item.first is sectionID
        uint64_t sectionSize = sectionsSizes[item.first];
        rLaunchTensorMemory.m_sectionsData[item.first].host = m_rTestDevice.allocateHostBuffer(sectionSize, 0); // Take ownership
        rLaunchTensorMemory.m_sectionsData[item.first].dev = m_rTestDevice.allocateDeviceBuffer(sectionSize, 0);  // Take ownership
    }
}

void TestLauncher::allocateMemory(const TestRecipeInterface&     rRecipe,
                                  LaunchTensorMemory&            rLaunchTensorMemory,
                                  const TestTensorsDimensions&   rTestTensorsDimensions) const
{
    // allocate memory for entire sections
    allocateSectionMemory(rRecipe, rLaunchTensorMemory);

    //  Set tensor addresses by offsets into the sections
    const std::vector<TensorInfo>& rLaunchInfoInputs  = rRecipe.getTensorInfoVecInputs();
    for (unsigned tensorIndex = 0; tensorIndex < rLaunchInfoInputs.size(); tensorIndex++)
    {
        TensorInfo tensorInfo(rLaunchInfoInputs[tensorIndex]);


        if (tensorInfo.m_tensorFlags == (uint32_t)TensorFlag::CONST_SECTION)
        {
            rLaunchTensorMemory.m_tensorInfoVecInputs.emplace_back(TestHostBufferMalloc(), TestDeviceBufferAlloc());
            continue;
        }

        const TensorDimensions* pTensorDimensions = rTestTensorsDimensions.getDimensions(true, tensorIndex);
        if (pTensorDimensions != nullptr)
        {
            std::memcpy(tensorInfo.m_tensorDimsSize, pTensorDimensions->data(), HABANA_DIM_MAX * sizeof(TSize));
        }
        unsigned sectionID = rRecipe.getSectionID(tensorInfo);
        rLaunchTensorMemory.m_tensorInfoVecInputs.emplace_back(
                TestHostBufferMalloc(rLaunchTensorMemory.m_sectionsData[sectionID].host, tensorInfo.m_sectionOffset),
                TestDeviceBufferAlloc(rLaunchTensorMemory.m_sectionsData[sectionID].dev, tensorInfo.m_sectionOffset, tensorInfo.m_tensorSize));
    }

    const std::vector<TensorInfo>& rLaunchInfoOutputs = rRecipe.getTensorInfoVecOutputs();
    for (unsigned tensorIndex = 0; tensorIndex < rLaunchInfoOutputs.size(); tensorIndex++)
    {
        TensorInfo tensorInfo(rLaunchInfoOutputs[tensorIndex]);

        if (tensorInfo.m_tensorFlags == (uint32_t)TensorFlag::CONST_SECTION)
        {
            rLaunchTensorMemory.m_tensorInfoVecOutputs.emplace_back(TestHostBufferMalloc(), TestDeviceBufferAlloc());
            continue;
        }

        const TensorDimensions* pTensorDimensions = rTestTensorsDimensions.getDimensions(false, tensorIndex);
        if (pTensorDimensions != nullptr)
        {
            std::memcpy(tensorInfo.m_tensorDimsSize, pTensorDimensions->data(), HABANA_DIM_MAX * sizeof(TSize));
        }
        unsigned sectionID = rRecipe.getSectionID(tensorInfo);
        rLaunchTensorMemory.m_tensorInfoVecOutputs.emplace_back(
                TestHostBufferMalloc(rLaunchTensorMemory.m_sectionsData[sectionID].host, tensorInfo.m_sectionOffset),
                TestDeviceBufferAlloc(rLaunchTensorMemory.m_sectionsData[sectionID].dev, tensorInfo.m_sectionOffset, tensorInfo.m_tensorSize));
    }
}

void TestLauncher::generateLaunchTensors(const TestRecipeInterface&   rRecipe,
                                         const LaunchTensorMemory&    rLaunchTensorMemory,
                                         SynLaunchTensorInfoVec&      rSynLaunchTensorInfoVec,
                                         const AllocDeviceBuffersVec& constSectionsDeviceBuffers,
                                         TensorInitInfo               tensorInitInfo,
                                         const TestTensorsDimensions& rTestTensorsDimensions)
{
    const uint64_t numberOfTensors = rRecipe.getTensorInfoVecSizeInput() + rRecipe.getTensorInfoVecSizeOutput();

    const std::vector<uint64_t>& tensorIds = rRecipe.retrieveTensorsId();
    ASSERT_EQ(tensorIds.size(), numberOfTensors) << "Invalid amount of IDs";

    ASSERT_EQ(rRecipe.getTensorInfoVecSizeInput(), rLaunchTensorMemory.m_tensorInfoVecInputs.size())
        << "unequal input tensors number";
    ASSERT_EQ(rRecipe.getTensorInfoVecSizeOutput(), rLaunchTensorMemory.m_tensorInfoVecOutputs.size())
        << "unequal output tensors number";

    rSynLaunchTensorInfoVec.resize(rRecipe.getTensorInfoVecSizeInput() + rRecipe.getTensorInfoVecSizeOutput());

    unsigned tensorIndex = 0;
    for (unsigned tensorIndexInput = 0; tensorIndexInput < rRecipe.getTensorInfoVecSizeInput();
         tensorIndexInput++, tensorIndex++)
    {
        if (rRecipe.getTensorInfoInput(tensorIndexInput)->m_isConst)
        {
            continue;
        }

        bool                    isOnConstSection     = rRecipe.isInputOnConstSection(tensorIndexInput);
        synLaunchTensorInfoExt& rSynLaunchTensorInfo = rSynLaunchTensorInfoVec[tensorIndex];
        TensorInfo              tensorInfo(rRecipe.getTensorInfoVecInputs()[tensorIndexInput]);
        const TensorDimensions* pTensorDimensions = rTestTensorsDimensions.getDimensions(true, tensorIndexInput);

        if (pTensorDimensions != nullptr)
        {
            std::memcpy(tensorInfo.m_tensorDimsSize, pTensorDimensions->data(), HABANA_DIM_MAX * sizeof(TSize));
        }

        rSynLaunchTensorInfo.tensorName =
            rRecipe.getTensorsName()[tensorIndex];  // Must match the name supplied at tensor creation
        rSynLaunchTensorInfo.tensorType = tensorInfo.m_tensorType;
        rSynLaunchTensorInfo.tensorId   = tensorIds[tensorIndex];

        if (!isOnConstSection)
        {
            rSynLaunchTensorInfo.pTensorAddress =
                rLaunchTensorMemory.m_tensorInfoVecInputs[tensorIndexInput].getTestDeviceBuffer().getBuffer();
        }
        else
        {
            HB_ASSERT(tensorInfo.m_sectionIndex < constSectionsDeviceBuffers.size(), "Invalid tensor's section-index");

            rSynLaunchTensorInfo.pTensorAddress =
                constSectionsDeviceBuffers[tensorInfo.m_sectionIndex].getBuffer() + tensorInfo.m_sectionOffset;
        }

        std::memcpy(rSynLaunchTensorInfo.tensorSize, tensorInfo.m_tensorDimsSize, HABANA_DIM_MAX * sizeof(TSize));

        if (!isOnConstSection)
        {
            // Initialize host memory
            initBufferValues(
                tensorInitInfo.input,
                tensorInfo.m_dataType,
                getNumberOfElements(tensorInfo.m_tensorDimsSize, tensorInfo.m_dimsAmount),
                const_cast<void*>(
                    rLaunchTensorMemory.m_tensorInfoVecInputs[tensorIndexInput].getTestHostBuffer().getBuffer()));
        }
    }

    for (unsigned tensorIndexOutput = 0; tensorIndexOutput < rRecipe.getTensorInfoVecSizeOutput();
         tensorIndexOutput++, tensorIndex++)
    {
        TensorInfo              tensorInfo(rRecipe.getTensorInfoVecOutputs()[tensorIndexOutput]);
        const TensorDimensions* pTensorDimensions = rTestTensorsDimensions.getDimensions(false, tensorIndexOutput);

        if (pTensorDimensions != nullptr)
        {
            std::memcpy(tensorInfo.m_tensorDimsSize, pTensorDimensions->data(), HABANA_DIM_MAX * sizeof(TSize));
        }

        synLaunchTensorInfoExt& rSynLaunchTensorInfo = rSynLaunchTensorInfoVec[tensorIndex];

        rSynLaunchTensorInfo.tensorName =
            rRecipe.getTensorsName()[tensorIndex];  // Must match the name supplied at tensor creation
        rSynLaunchTensorInfo.pTensorAddress =
            rLaunchTensorMemory.m_tensorInfoVecOutputs[tensorIndexOutput].getTestDeviceBuffer().getBuffer();
        rSynLaunchTensorInfo.tensorType = tensorInfo.m_tensorType;
        rSynLaunchTensorInfo.tensorId   = tensorIds[tensorIndex];

        std::memcpy(rSynLaunchTensorInfo.tensorSize, tensorInfo.m_tensorDimsSize, HABANA_DIM_MAX * sizeof(TSize));

        initBufferValues(
            tensorInitInfo.output,
            tensorInfo.m_dataType,
            getNumberOfElements(tensorInfo.m_tensorDimsSize, tensorInfo.m_dimsAmount),
            const_cast<void*>(
                rLaunchTensorMemory.m_tensorInfoVecOutputs[tensorIndexOutput].getTestHostBuffer().getBuffer()));
    }
}

void TestLauncher::generateLaunchTensorsWithTensorsMemory(const TestRecipeInterface& rRecipe,
                                                          const LaunchTensorMemory&  rLaunchTensorMemory,
                                                          SynLaunchTensorInfoVec&    rSynLaunchTensorInfoVec,
                                                          TensorInitInfo             tensorInitInfo)
{
    const uint64_t numberOfTensors = rRecipe.getTensorInfoVecSizeInput() + rRecipe.getTensorInfoVecSizeOutput();

    const std::vector<uint64_t>& tensorIds = rRecipe.retrieveTensorsId();
    ASSERT_EQ(tensorIds.size(), numberOfTensors) << "Invalid amount of IDs";

    ASSERT_EQ(rRecipe.getTensorInfoVecSizeInput(), rLaunchTensorMemory.m_tensorInfoVecInputs.size())
        << "unequal input tensors number";
    ASSERT_EQ(rRecipe.getTensorInfoVecSizeOutput(), rLaunchTensorMemory.m_tensorInfoVecOutputs.size())
        << "unequal output tensors number";

    rSynLaunchTensorInfoVec.resize(rRecipe.getTensorInfoVecSizeInput() + rRecipe.getTensorInfoVecSizeOutput());

    unsigned tensorIndex = 0;
    for (unsigned tensorIndexInput = 0; tensorIndexInput < rRecipe.getTensorInfoVecSizeInput();
         tensorIndexInput++, tensorIndex++)
    {
        if (rRecipe.getTensorInfoInput(tensorIndexInput)->m_isConst)
        {
            continue;
        }
        synLaunchTensorInfoExt& rSynLaunchTensorInfo = rSynLaunchTensorInfoVec[tensorIndex];

        rSynLaunchTensorInfo.tensorName =
            rRecipe.getTensorsName()[tensorIndex];  // Must match the name supplied at tensor creation
        rSynLaunchTensorInfo.pTensorAddress =
            rLaunchTensorMemory.m_tensorInfoVecInputs[tensorIndexInput].getTestDeviceBuffer().getBuffer();

        rSynLaunchTensorInfo.tensorType = rRecipe.getTensorInfoVecInputs()[tensorIndexInput].m_tensorType;
        rSynLaunchTensorInfo.tensorId   = tensorIds[tensorIndex];

        std::memcpy(rSynLaunchTensorInfo.tensorSize,
                    rRecipe.getTensorInfoVecInputs()[tensorIndexInput].m_tensorDimsSize,
                    HABANA_DIM_MAX * sizeof(uint32_t));

        // Initialize host memory
        initBufferValues(
            tensorInitInfo.input,
            rRecipe.getTensorInfoVecInputs()[tensorIndexInput].m_dataType,
            getNumberOfElements(rRecipe.getTensorInfoVecInputs()[tensorIndexInput].m_tensorDimsSize,
                                rRecipe.getTensorInfoVecInputs()[tensorIndexInput].m_dimsAmount),
            const_cast<void*>(
                rLaunchTensorMemory.m_tensorInfoVecInputs[tensorIndexInput].getTestHostBuffer().getBuffer()));
    }

    for (unsigned tensorIndexOutput = 0; tensorIndexOutput < rRecipe.getTensorInfoVecSizeOutput();
         tensorIndexOutput++, tensorIndex++)
    {
        synLaunchTensorInfoExt& rSynLaunchTensorInfo = rSynLaunchTensorInfoVec[tensorIndex];

        rSynLaunchTensorInfo.tensorName =
            rRecipe.getTensorsName()[tensorIndex];  // Must match the name supplied at tensor creation
        rSynLaunchTensorInfo.pTensorAddress =
            rLaunchTensorMemory.m_tensorInfoVecOutputs[tensorIndexOutput].getTestDeviceBuffer().getBuffer();

        rSynLaunchTensorInfo.tensorType = rRecipe.getTensorInfoVecOutputs()[tensorIndexOutput].m_tensorType;
        rSynLaunchTensorInfo.tensorId   = tensorIds[tensorIndex];

        std::memcpy(rSynLaunchTensorInfo.tensorSize,
                    rRecipe.getTensorInfoVecOutputs()[tensorIndexOutput].m_tensorDimsSize,
                    HABANA_DIM_MAX * sizeof(uint32_t));

        initBufferValues(
            tensorInitInfo.output,
            rRecipe.getTensorInfoVecOutputs()[tensorIndexOutput].m_dataType,
            getNumberOfElements(rRecipe.getTensorInfoVecOutputs()[tensorIndexOutput].m_tensorDimsSize,
                                rRecipe.getTensorInfoVecOutputs()[tensorIndexOutput].m_dimsAmount),
            const_cast<void*>(
                rLaunchTensorMemory.m_tensorInfoVecOutputs[tensorIndexOutput].getTestHostBuffer().getBuffer()));
    }
}
