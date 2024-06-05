#pragma once

// eager includes (relative to src/eager/lib/)
#include "node_info/node_info_defs.h"
#include "utils/general_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/types.h"

// std includes
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace eager_mode
{
struct ProgramDataBlob
{
    ProgramDataBlob() = default;
    ProgramDataBlob(const char* ptr, deviceAddrOffset address, uint64_t size)
    : hostAddrPtr(ptr), deviceAddr(address), binSize(size)
    {
    }

    const char*      hostAddrPtr = nullptr;
    deviceAddrOffset deviceAddr  = 0;
    uint64_t         binSize     = 0;
};

struct KernelIdMapping
{
    KernelIdMapping() = default;
    KernelIdMapping(kernelID id, deviceAddrOffset address) : kernelId(id), deviceAddr(address) {}

    kernelID         kernelId   = 0;
    deviceAddrOffset deviceAddr = 0;
};

using ProgramDataBlobsVec = VecNodes<ProgramDataBlob>;
using ProgramDataBlobsSharedPtrs = VecNodes<std::shared_ptr<char>>;

class ProgramDataBlobManager
{
public:
    void registerProgramDataBlobForDownload(char* hostAddr, deviceAddrOffset deviceAddr, uint64_t size)
    {
        // it is possible the user data was copied into the tensor, but we do not bother
        // to optimize this use case at the moment. Futurewise, we can keep it using a unique pointer
        // and claim ownership here, such that if we end up with a single program data blob consisting of
        // only this single user data we can skip the copy into the recipe.
        m_copyRequired = true;
        m_programBlobs.emplace_back(hostAddr, deviceAddr, size);
    }

    deviceAddrOffset getKernelAddress(kernelID kid) const
    {
        auto iter = std::find_if(m_kernelIdToProgramBlobIdx.begin(),
                                 m_kernelIdToProgramBlobIdx.end(),
                                 [kid](const KernelIdMapping& mapping) { return kid == mapping.kernelId; });
        EAGER_ASSERT(iter != m_kernelIdToProgramBlobIdx.end(), "kernel id is not found, this is impossible");
        return iter->deviceAddr;
    }

    std::optional<deviceAddrOffset> getKernelAddressIfFound(char* hostAddr)
    {
        auto iter =
            std::find_if(m_programBlobs.begin(), m_programBlobs.end(), [hostAddr](const ProgramDataBlob& blobInfo) {
                return hostAddr == blobInfo.hostAddrPtr;
            });
        if (iter != m_programBlobs.end())
        {
            return iter->deviceAddr;
        }
        return {};
    }

    void registerExistingTPCProgramDataBlobForDownload(deviceAddrOffset deviceAddr, kernelID kid)
    {
        m_kernelIdToProgramBlobIdx.emplace_back(kid, deviceAddr);
    }

    void
    registerNewTPCProgramDataBlobForDownload(char* hostAddr, deviceAddrOffset deviceAddr, uint64_t size, kernelID kid)
    {
        m_programBlobs.emplace_back(hostAddr, deviceAddr, size);
        m_kernelIdToProgramBlobIdx.emplace_back(kid, deviceAddr);
    }

    void cacheBlobBuffer(std::shared_ptr<char>&& buffer) { m_programBlobsSharedPtrs.push_back(std::move(buffer)); };

    const ProgramDataBlobsVec& getProgramDataBlobs() const { return m_programBlobs; }

    const ProgramDataBlobsSharedPtrs& getProgramDataBlobsSharedPtrs() const { return m_programBlobsSharedPtrs; }

    bool isProgramDataBlobCopyRequired() const { return m_copyRequired || m_programBlobs.size() > 1; }

private:
    bool                m_copyRequired = false;
    ProgramDataBlobsVec m_programBlobs;
    // A vector mapping each tpc node unique id to the relevant kernel device address.
    // many to one relationship as for Eager the id is unique per tpc nodea and not per kernel.
    // graph mode relies upon crc to compare kernel contents (assuming there are no colisions).
    // Eager graph just assigns each tpc node a new incremental unique id.
    // uniqueness can be checked through the kernel binary pointers for Eager.
    VecNodes<KernelIdMapping> m_kernelIdToProgramBlobIdx;
    // A vector holding shared pointers to kernels requiring ownership.
    // In the general case there are 3 options:
    // case 1: The tpc node pointed to the Elf binary of tpc_kernel instantiation and
    // the kernel itself pointed to a location within this Elf binary.
    // case 2: The tpc node allocated shared pointer for Elf binary was copied into
    // such that the Elf binary data is owned by the tpc node and the kernel binary
    // just points into it. This was legacy code and likely no longer exists after
    // SW-110568.
    // case 3: The tpc node kernel binary is located in the shared pointer owned by tpc
    // node, meaning it was copied into. The Elf binary is less interesting in this case.
    // This probably corresponds to the tpc fuser use case where a few existing kernels
    // are fused into a new kernel, thus the new kernel is not part of the static Elf binary
    // and requires the tpc node to get a deep copy. I don't think this is relevant to Eager
    // and we do not seem to enter this code flow for current Bert or Resnet.
    // So likely the caching of the shared pointer bellow
    // (Elf binary for case 2 or Kernel binary for case 3) is not needed for Eager use case
    // but keeping it in here in case it does\ this is changed in the future.
    ProgramDataBlobsSharedPtrs m_programBlobsSharedPtrs;
};

}  // namespace eager_mode