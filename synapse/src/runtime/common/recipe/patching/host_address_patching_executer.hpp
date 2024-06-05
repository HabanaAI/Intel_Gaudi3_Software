#pragma once

#include "host_address_patcher.hpp"

struct data_chunk_patch_point_t;
struct DataChunkPatchPointsInfo;

namespace patching
{
class HostAddressPatchingInformation;

class HostAddressPatchingExecuter
{
public:
    static synStatus verify(const recipe_t& rRecipe, uint64_t maxSectionId);

    static bool executePatchingInDataChunks(const uint64_t* const            newAddressToBeSetDb,
                                            const data_chunk_patch_point_t*& currPatchPoint,
                                            const std::vector<uint64_t>&     dataChunksHostAddresses,
                                            const uint64_t                   dataChunkSize,
                                            const uint32_t                   lastNodeIndexToPatch,
                                            const uint32_t                   sobBaseAddr);

    static bool executePatchingInBuffer(const HostAddressPatchingInformation* pPatchingInfo,
                                        const patch_point_t*                  patchingPoints,
                                        const uint64_t                        numberOfPatchingPoints,
                                        blob_t*                               blobs,
                                        const uint64_t                        numberOfBlobs,
                                        const uint32_t                        sobBaseAddr);

    static bool verifyPatchingInformation(const patch_point_t* patchingPoints,
                                          uint64_t             numberOfPatchingPoints,
                                          const blob_t*        blobs,
                                          uint64_t             numberOfBlobs,
                                          uint64_t             sectionsDbSize,
                                          synDeviceType        deviceType);
};
}  // namespace patching