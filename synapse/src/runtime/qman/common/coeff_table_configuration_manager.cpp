#include "coeff_table_configuration_manager.hpp"
#include "runtime/qman/common/master_qmans_definition_interface.hpp"
#include "queue_base_qman.hpp"
#include "syn_singleton.hpp"
#include "runtime/qman/gaudi/device_gaudi.hpp"

namespace gaudi
{
extern std::unique_ptr<CoeffTableConfManager> createCoeffTableConfManager(DeviceGaudi* device);
}

std::unique_ptr<CoeffTableConfManager> createCoeffTableConfManager(DeviceGaudi* device)
{
    switch (device->getDevType())
    {
        case synDeviceGaudi:
            return gaudi::createCoeffTableConfManager(device);
        default:
            return nullptr;
    }
}

CoeffTableConfManager::CoeffTableConfManager(DeviceGaudi* device) : m_device(device) {}

/*************************************************************************************************
 *   @brief  generateAndSendCoeffTableConfiguration() generates LBW write commands in order to
 *           configure coeff table addresses for all TPCs in compute_tpc cluster
 *
 *   @param  coeffTableAddr - uint64_t that hold the coeff table address on HBM
 *   @param  stream - From this stream we can get the information on the engines in cluster
 *
 *   @return status
 **************************************************************************************************/
synStatus CoeffTableConfManager::generateCoeffTableConfigurationPackets(char*&    pPackets,
                                                                        uint64_t& packetsSize,
                                                                        uint64_t  tableBaseAddr)
{
    if (pPackets != nullptr)
    {
        LOG_ERR(SYN_API, "{}: pPackets param must be nullptr", HLLOG_FUNC);
        return synFail;
    }

    // Allocate buffer for packets
    uint64_t signalPacketsSize                       = 0;
    uint64_t singleEngineCoeffTableConfigCommandSize = m_cmdBuffPktGenerator->getCoeffTableConfigCommandSize();
    uint64_t coeffTableConfigCommandTotalSize =
        singleEngineCoeffTableConfigCommandSize * m_halReader->getNumTpcEngines();

    if (m_isSyncWithExternalRequired)
    {
        signalPacketsSize = m_cmdBuffPktGenerator->getSignalCommandSize();
    }
    packetsSize = coeffTableConfigCommandTotalSize + signalPacketsSize;
    pPackets    = new char[packetsSize];

    ptrToInt addr;
    addr.u64 = tableBaseAddr;

    synStatus status = m_cmdBuffPktGenerator->generateCoeffTableConfigCommands(pPackets,
                                                                               addr,
                                                                               singleEngineCoeffTableConfigCommandSize);

    if (m_isSyncWithExternalRequired)
    {
        char* pTmpPackets = pPackets + coeffTableConfigCommandTotalSize;
        status = m_cmdBuffPktGenerator->generateSignalCommand(pTmpPackets, signalPacketsSize, 0, 1, 0, ALL_BARRIERS);
        LOG_TRACE(SYN_API, "Generate signal from external queue for coeff table CS");
    }

    if (status != synSuccess)
    {
        delete[] pPackets;
        return synFail;
    }

    return synSuccess;
}

/*************************************************************************************************
 *   @brief  submitCoeffTableConfiguration() is called from acquire()
 *           It should allocate coeff table in HBM and configure its address for each TPC
 *
 *   @return status
 **************************************************************************************************/
synStatus CoeffTableConfManager::submitCoeffTableConfiguration(const synDeviceType deviceType)
{
    if (deviceType != synDeviceGaudi)
    {
        return synSuccess;
    }

    synStatus status = synSuccess;

    /*
     * 1. Allocate host memory for coeff table
     */
    uint64_t tableAllocatedSize = getSpecialFuncCoeffTableAllocatedSize();
    uint64_t tableSize          = getSpecialFuncCoeffTableSize();

    if (tableAllocatedSize < tableSize)
    {
        LOG_ERR(SYN_API,
                "{}: Can't generate coeff table. Real table size {} is bigger than allocated size {}",
                HLLOG_FUNC,
                tableSize,
                tableAllocatedSize);
        return synFail;
    }
    uint64_t                bufferOnDevice = 0;
    std::unique_ptr<char[]> bufferOnHostUnique(new char[tableAllocatedSize]);
    void*                   bufferOnHost = (void*)(bufferOnHostUnique.get());
    std::string             mappingDescHost("Special Func Coeff Table (Host)");

    status = _SYN_SINGLETON_INTERNAL->m_deviceManager.mapBufferToDevice(tableAllocatedSize,
                                                                        bufferOnHost,
                                                                        false /* isUserRequest */,
                                                                        0,
                                                                        mappingDescHost);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_API, "{}: Can't generate coeff table. Failed to allocate and map memory on host", HLLOG_FUNC);
        return status;
    }
    memcpy(bufferOnHost, getSpecialFuncCoeffTableData(), tableSize);

    do
    {
        /*
         * 2. Allocate device memory for coeff table
         */
        std::string mappingDescDevice("Special Func Coeff Table (Device)");
        status = _SYN_SINGLETON_INTERNAL->m_deviceManager.allocateDeviceMemory((tableAllocatedSize),
                                                                               0,
                                                                               (void**)&bufferOnDevice,
                                                                               false /* isUserRequest */,
                                                                               0,
                                                                               mappingDescDevice);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_API, "{}: Can't generate coeff table. Failed to allocate and map memory on device", HLLOG_FUNC);
            break;
        }
        LOG_TRACE(SYN_API, "Allocate coeff table at addr 0x{:x}, size 0x{:x}", bufferOnDevice, tableAllocatedSize);

        /*
         * 3. Get DMA downstream to copy coeff table from host to device
         */
        QueueBaseQman* pStreamCopy = static_cast<QueueBaseQman*>(m_device->getDmaDownStream());
        if (pStreamCopy == nullptr)
        {
            LOG_ERR(SYN_API, "{}: Can't generate coeff table. Failed to get dma downstream", HLLOG_FUNC);
            status = synFail;
            break;
        }

        /*
         * 4. Copy coeff table from host to device
         */
        internalMemcopyParams memcpyParams {
            {.src = (uint64_t)bufferOnHost, .dst = bufferOnDevice, .size = (tableSize)}};
        status = pStreamCopy->memcopy(memcpyParams,
                                      MEMCOPY_HOST_TO_DRAM,
                                      false, /* isUserRequest */
                                      nullptr,
                                      0 /* overrideMemsetVal */,
                                      false /* inspectCopiedContent */,
                                      nullptr /* pRecipeProgramBuffer */,
                                      0);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_API, "{}: Can't generate coeff table. Failed to copy table from host to device", HLLOG_FUNC);
            break;
        }
        LOG_TRACE(SYN_API,
                  "Copy coeff table from host addr 0x{:p} to device addr 0x{:x}",
                  bufferOnHost,
                  bufferOnDevice);

        status = pStreamCopy->synchronize(nullptr, false /* isUserRequest */);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_API, "{}: Can't generate coeff table. Can't synchronize stream", HLLOG_FUNC);
            break;
        }

        status = _SYN_SINGLETON_INTERNAL->m_deviceManager.unmapBufferFromDevice(bufferOnHost, false);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_API, "{}: Can't generate coeff table. Failed to unmap memory on host", HLLOG_FUNC);
            return status;
        }

        /*
         * 5. Generate configuration packets (msgLong + signal) for all TPCs
         */
        char*       pPackets    = nullptr;
        uint64_t    packetsSize = 0;
        uint32_t    qmanId      = m_qmanDefs->getArbitratorMasterQueueIdForCompute();
        std::string description("Coeff table configuration");

        status = generateCoeffTableConfigurationPackets(pPackets, packetsSize, bufferOnDevice);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_API, "{}: Can't generate packets for {}", HLLOG_FUNC, description);
            break;
        }

        /*
         * 6. Send packets to TPC0
         *    Send packets using the arb-master as the payload stream and stream-master as work-completion stream
         */
        status = _SYN_SINGLETON_INTERNAL->submitTrainingConfigurationCS(deviceType,
                                                                        pPackets,
                                                                        packetsSize,
                                                                        description,
                                                                        qmanId,
                                                                        m_isConfigOnInternal,
                                                                        m_isSyncWithExternalRequired);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_API, "{}: Can't submit CS", HLLOG_FUNC);
            break;
        }
    } while (0);

    return status;
}
