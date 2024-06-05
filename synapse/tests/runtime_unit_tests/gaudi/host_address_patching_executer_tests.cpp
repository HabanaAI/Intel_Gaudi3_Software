#include "runtime/common/recipe/patching/host_address_patching_executer.hpp"
#include "runtime/common/recipe/recipe_patch_processor.hpp"

#include "test_dummy_recipe.hpp"

#include "define_synapse_common.hpp"
#include "runtime/common/habana_global_conf_runtime.h"
#include "runtime/common/syn_logging.h"
#include <gtest/gtest.h>

using namespace patching;

class UTGaudiHostAddressPatchingExecuterTests : public ::testing::Test
{
};

inline uint64_t div_round_up(uint64_t a, uint64_t b)
{
    assert(b != 0 && "Divider should be non zero");
    return (a + b - 1) / b;
}

inline uint64_t round_to_multiple(uint64_t a, uint64_t mul)
{
    return mul == 0 ? 0 : mul * div_round_up(a, mul);
}

struct validateSectionInfo
{
    uint64_t sectionId;
    uint64_t sectionSize;
};

TEST(UTGaudiHostPatchingTest, basicSobPatching)
{
    HostAddressPatchingExecuter ha;

    TestDummyRecipe dummyRecipe(RECIPE_TYPE_NORMAL, 0x1002, 0x5003, 0x3004, 0x2005, 0x806, 1);

    const recipe_t* recipe = dummyRecipe.getRecipe();

    const synStatus status = ha.verify(*recipe, recipe->sections_nr + 1);
    ASSERT_EQ(status, synSuccess);

    // check patching on buffer
    HostAddressPatchingInformation hostAddressPatchingInformation;
    hostAddressPatchingInformation.initialize(recipe->sections_nr + 1, recipe->sections_nr - 1);
    uint64_t*                               pCurrWorkspaceSize   = recipe->workspace_sizes;
    uint64_t                                currWorkspaceAddress = 0;
    std::map<uint64_t, validateSectionInfo> sortedSectionStart;  // [address, [section_id,size]]
    std::vector<uint64_t>                   sectionAddresses;    // [section_id, address]
    uint32_t                                sobBaseAddress = 0xbeef;
    //
    WorkSpacesInformation workspaceInfo;
    workspaceInfo.programDataAddress = 0x10;
    workspaceInfo.scratchPadAddress  = 0x20;
    workspaceInfo.programCodeAddress = 0x30;
    for (uint64_t i = 0; i < recipe->workspace_nr; i++, pCurrWorkspaceSize++)
    {
        if (i == MEMORY_ID_RESERVED_FOR_PROGRAM_DATA)
        {
            currWorkspaceAddress = round_to_multiple(workspaceInfo.programDataAddress, (1 << 13));
        }
        else if (i == MEMORY_ID_RESERVED_FOR_WORKSPACE)
        {
            currWorkspaceAddress = workspaceInfo.scratchPadAddress;
        }
        else if (i == MEMORY_ID_RESERVED_FOR_PROGRAM)
        {
            currWorkspaceAddress = workspaceInfo.programCodeAddress;
        }
        else
        {
            LOG_ERR(SYN_API, "{}: Failed to update workspace index {}", HLLOG_FUNC, i);
            hostAddressPatchingInformation.patchingAbort();
            ASSERT_EQ(true, false);
        }

        uint64_t sectionTypeId = 0;

        bool res = hostAddressPatchingInformation.setSectionHostAddress(i,
                                                                        sectionTypeId,
                                                                        currWorkspaceAddress,
                                                                        false,
                                                                        false,
                                                                        false);
        if (!res)
        {
            LOG_ERR(SYN_API,
                    "{}: Failed to setSectionHostAddress for section {} with address {:#x}",
                    HLLOG_FUNC,
                    i,
                    currWorkspaceAddress);
            hostAddressPatchingInformation.patchingAbort();
            ASSERT_EQ(true, false);
        }
    }
    bool res = ha.executePatchingInBuffer(&hostAddressPatchingInformation,
                                          recipe->patch_points,
                                          recipe->patch_points_nr,
                                          recipe->blobs,
                                          recipe->blobs_nr,
                                          sobBaseAddress);
    ASSERT_EQ(res, true);
    ASSERT_EQ(*((uint32_t*)(recipe->blobs[recipe->patch_points[0].blob_idx].data) +
                recipe->patch_points[0].dw_offset_in_blob),
              sobBaseAddress);
    // check patching on data chunks
    data_chunk_patch_point_t dataChunkPatchPoints[2];
    memcpy(dataChunkPatchPoints, recipe->patch_points, sizeof(patch_point_t));
    uint64_t sectionToHostAddressDb[recipe->sections_nr + 1];
    memset(sectionToHostAddressDb, 0, sizeof(sectionToHostAddressDb));
    std::vector<uint64_t> dataChunkHostAddresses(10, 0);
    uint64_t              buffer[64];
    buffer[0]                                                    = 0;
    sectionToHostAddressDb[0]                                    = 0;
    dataChunkHostAddresses[0]                                    = (uint64_t)buffer;
    uint64_t dcSize                                              = GCFG_STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_LOWER_CP.value() * 1024;
    dataChunkPatchPoints[0].node_exe_index                       = 0;
    dataChunkPatchPoints[1].node_exe_index                       = UINT32_MAX;
    dataChunkPatchPoints[0].offset_in_data_chunk                 = 0;
    dataChunkPatchPoints[0].type                                 = patch_point_t::SOB_PATCH_POINT;
    dataChunkPatchPoints[0].data_chunk_index                     = 0;
    dataChunkPatchPoints[0].memory_patch_point.effective_address = 0;
    dataChunkPatchPoints[0].memory_patch_point.section_idx       = 0;
    const data_chunk_patch_point_t* dataChunkPatchPoint          = &dataChunkPatchPoints[0];
    res                                                          = ha.executePatchingInDataChunks(sectionToHostAddressDb,
                                         dataChunkPatchPoint,
                                         dataChunkHostAddresses,
                                         dcSize,
                                         0,
                                         sobBaseAddress);
    ASSERT_EQ(res, true);
    ASSERT_EQ(buffer[0], sobBaseAddress);
}

TEST(UTGaudiHostPatchingTest, verifyPatchingInformationNoPatching)
{
    bool res = HostAddressPatchingExecuter::verifyPatchingInformation(nullptr, 0, nullptr, 0, 0, synDeviceGaudi);
    ASSERT_EQ(res, true);
}

TEST(UTGaudiHostPatchingTest, verifyPatchingInformationInvalidNumberOfBlobs)
{
    bool res = HostAddressPatchingExecuter::verifyPatchingInformation(nullptr, 1, nullptr, 0, 0, synDeviceGaudi);
    ASSERT_EQ(res, false);
}

TEST(UTGaudiHostPatchingTest, verifyPatchingInformationInvalidBlobs)
{
    bool res = HostAddressPatchingExecuter::verifyPatchingInformation(nullptr, 1, nullptr, 1, 0, synDeviceGaudi);
    ASSERT_EQ(res, false);
}

TEST(UTGaudiHostPatchingTest, verifyPatchingInformationInvalidPatchingPoints)
{
    blob_t blob = {0};
    bool   res = HostAddressPatchingExecuter::verifyPatchingInformation(nullptr, 1, &blob, 1, 0, synDeviceGaudi);
    ASSERT_EQ(res, false);
}
