#include "tpc_node_handler.h"

// eager includes (relative to src/eager/lib/)
#include "eager_graph.h"
#include "utils/general_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/kernel_db.h"
#include "graph_compiler/types.h"
#include "hal_reader/hal_reader.h"

// std includes
#include <optional>

namespace eager_mode::glue
{
bool loadKernelAndAllocAuxTensors(EagerGraph& g, TPCNode& n)
{
    if (auto initRes = n.init(deviceTypeToDeviceID(g.getDeviceType()), nullptr, g.getNextTPCKernelUniqueId());
        unlikely(initRes != tpc_lib_api::GLUE_SUCCESS))
    {
        LOG_ERR(EAGER,
                "TPC Node init failed with {}. Eager mode doesn't support retries.",
                KernelDB::parseReturnValue(initRes));
        return false;
    }

    ProgramDataBlobManager&         programDataBlobManager = g.getProgramDataBlobManager();
    KernelInfo                      kernelInfo             = n.getKernelInfo();
    std::optional<deviceAddrOffset> existingAddress =
        programDataBlobManager.getKernelAddressIfFound(kernelInfo.kernelBinary);
    if (existingAddress.has_value())
    {
        programDataBlobManager.registerExistingTPCProgramDataBlobForDownload(*existingAddress, kernelInfo.kernelId);
        n.setKernelOffsetInSection(*existingAddress);
    }
    else
    {
        auto&                      programDataAllocator = g.getProgramDataAllocator();
        Settable<deviceAddrOffset> newAddress =
            programDataAllocator.Allocate(kernelInfo.kernelSize, g.getHALReader()->getCacheLineSizeInBytes());
        if (!newAddress.is_set())
        {
            g.getGraphAnnotation().errors.memoryAllocationError = true;
            EAGER_REPORT_ERROR("Failed to allocate tpc kernel for Eager");
            return false;
        }
        n.setKernelOffsetInSection(newAddress.value());
        programDataBlobManager.registerNewTPCProgramDataBlobForDownload(kernelInfo.kernelBinary,
                                                                        *newAddress,
                                                                        kernelInfo.kernelSize,
                                                                        kernelInfo.kernelId);
        if (kernelInfo.cachedBinary)
        {
            programDataBlobManager.cacheBlobBuffer(std::move(kernelInfo.cachedBinary));
        }
    }
    return true;  // success!
}

}  // namespace eager_mode::glue
