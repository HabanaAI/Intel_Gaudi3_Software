#include "coeff_table_configuration_manager.hpp"
#include "defenders.h"
#include "device_scal.hpp"
#include "runtime/scal/common/stream_compute_scal.hpp"
#include "runtime/scal/common/entities/scal_stream_copy_interface.hpp"

using namespace common;

CoeffTableConf::CoeffTableConf(common::DeviceScal* device) : m_device(device) {}

/*************************************************************************************************
 *   @brief  submitCoeffTableConfiguration() is called from acquire()
 *           It should allocate coeff table in HBM and configure its address for each TPC
 *
 *   @return status
 **************************************************************************************************/
synStatus CoeffTableConf::submitCoeffTableConfiguration()
{
    synStatus status = synSuccess;

    /*
     * 1. Allocate host memory for coeff table
     */
    std::string mappingDescHost("Special Func Coeff Table (Host)");
    uint64_t    tableSize = getSpecialFuncCoeffTableSize();
    void*       bufferOnHost;

    status = m_device->allocateMemory(tableSize, synMemFlags::synMemHost, &bufferOnHost, false /* isUserRequest */, 0, mappingDescHost);

    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "{}: Can not configure coeff table. Failed to allocate and map memory on host", HLLOG_FUNC);
        return status;
    }
    std::memcpy(bufferOnHost, getSpecialFuncCoeffTableData(), tableSize);

    /*
     * 2. Allocate device memory for coeff table
     */
    std::string mappingDescDevice("Special Func Coeff Table (Device)");
    uint64_t    bufferOnDevice = 0;

    status = m_device->allocateMemory(tableSize, synMemFlags::synMemDevice, (void**)&bufferOnDevice, false /* isUserRequest */, 0, mappingDescDevice);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE,
                "{}: Can not configure coeff table. Failed to allocate and map memory on device",
                HLLOG_FUNC);
        return status;
    }

    QueueInterface* copyHostToDeviceQueueInterface;
    QueueInterface* computeQueueInterface;
    /*
     * 3. Create a user stream to copy coeff table from host to device
     */
    LOG_DEBUG_T(SYN_DEVICE, "{}: create host to device to copy coeff table", HLLOG_FUNC);
    do // Do once
    {
        status = m_device->createStreamQueue(QUEUE_TYPE_COPY_HOST_TO_DEVICE, 0, false, copyHostToDeviceQueueInterface);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE, "{}: Can not configure coeff table. Failed to create copyHostToDeviceStreamHandle stream", HLLOG_FUNC);
            return status;
        }
        HB_ASSERT_PTR(copyHostToDeviceQueueInterface);

        status = m_device->createStreamQueue(QUEUE_TYPE_COMPUTE, 0, false, computeQueueInterface);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE, "{}: Can not configure coeff table. Failed to create computeStreamHandle stream", HLLOG_FUNC);
            break;
        }
        HB_ASSERT_PTR(computeQueueInterface);

        /*
        * 4. Copy coeff table from host to device
        */
        internalMemcopyParams memcpyParams {{.src = (uint64_t)bufferOnHost, .dst = bufferOnDevice, .size = tableSize}};

        status = copyHostToDeviceQueueInterface->memcopy(memcpyParams,
                                                         MEMCOPY_HOST_TO_DRAM,
                                                         false /* isUserRequest */,
                                                         nullptr,
                                                         0 /* overrideMemsetVal */,
                                                         false /* inspectCopiedContent */,
                                                         nullptr /* pRecipeProgramBuffer */,
                                                         0);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE,
                    "{}: Can not configure coeff table. Failed to copy table from host to device",
                    HLLOG_FUNC);
            break;
        }
        LOG_DEBUG(SYN_DEVICE, "Copy coeff table from host addr 0x{:p} to device addr 0x{:x}", bufferOnHost, bufferOnDevice);

        status = copyHostToDeviceQueueInterface->synchronize(nullptr, false /* isUserRequest */);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE, "{}: Can not generate coeff table. Failed to synchronize user stream", HLLOG_FUNC);
            break;
        }

        status = m_device->deallocateMemory(bufferOnHost, synMemFlags::synMemHost, false /* isUserRequest */);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE, "{}: Failed to release coeff table on host.", HLLOG_FUNC);
            break;
        }

        /*
        * 5. Generate configuration LBW commands for all TPCs and send to scheduler
        */
        QueueBaseScalCommon* pComputeStreamBaseScalCommon = dynamic_cast<QueueBaseScalCommon*>(computeQueueInterface);
        CHECK_POINTER(SYN_DEVICE, pComputeStreamBaseScalCommon, "QueueBaseScal", synInvalidArgument);

        status = generateAndSendCoeffTableConfiguration(bufferOnDevice, pComputeStreamBaseScalCommon);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE,
                    "{}: Can not generated coeff table. Failed to generate and send coeff table configuration",
                    HLLOG_FUNC);
        }
    } while (0); // Do once - end

    synStatus destroyStatus = m_device->destroyStreamQueue(copyHostToDeviceQueueInterface);
    if (destroyStatus != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "{}: Failed to destroy copyHostToDeviceStreamHandle stream", HLLOG_FUNC);
        status = destroyStatus;
    }

    destroyStatus = m_device->destroyStreamQueue(computeQueueInterface);
    if (destroyStatus != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "{}: Failed to destroy computeStreamHandle stream", HLLOG_FUNC);
        status = destroyStatus;
    }

    return status;
}

/*************************************************************************************************
 *   @brief  generateAndSendCoeffTableConfiguration() generates LBW write commands in order to
 *           configure coeff table addresses for all TPCs in compute_tpc cluster
 *
 *   @param  coeffTableAddr - uint64_t that hold the coeff table address on HBM
 *   @param  stream - From this stream we can get the information on the engines in cluster
 *
 *   @return status
 **************************************************************************************************/
synStatus CoeffTableConf::generateAndSendCoeffTableConfiguration(uint64_t coeffTableAddr, QueueBaseScalCommon* pStream)
{
    synStatus status = synSuccess;

    // Get cluster information - we need to get a list of all engines in tpc_compute cluster
    scal_cluster_info_t clusterInfo;
    status = m_device->getClusterInfo(clusterInfo, (char*)COMPUTE_TPC_CLUSTER_NAME);
    if (status != synSuccess)
    {
        if (status == synUnavailable)
        {
            // We want to support configurations without compute tpc cluster
            LOG_WARN(SYN_DEVICE, "Cluster {} was not found", (char*)COMPUTE_TPC_CLUSTER_NAME);
            return synSuccess;
        }
        LOG_ERR(SYN_DEVICE, "Failed to get cluster info. status: {}", status);
        return synFail;
    }

    bool send = false;
    // For each TPC engine, send LBW command to setup 4 coeff table addresses
    // (each has 2 regs for Hi and LO addr)
    for (uint32_t engineId = 0; engineId < clusterInfo.numEngines; engineId++)
    {
        bool isLastEngine = (engineId == clusterInfo.numEngines - 1);

        // Get specific TPC engine info
        scal_control_core_info_t engineInfo;
        scal_control_core_get_info(clusterInfo.engines[engineId], &engineInfo);

        // Get base address of TPC# CFG block
        uint32_t tpcCfgBaseAddress = getTpcCfgBaseAddress(engineInfo.idx);

        const uint64_t* pTpcCoeffHbmOffset             = getTpcCoeffHbmAddressOffsetTable();
        const uint32_t* pTpcCoeffAddrRegOffset         = getTpcCoeffAddrRegOffsetTable();
        uint32_t        numOfRegsPerTpcCoeffTable      = getNumOfRegsPerTpcCoeffTable();
        uint32_t        numOfTpcCoeffTables            = getNumOfTpcCoeffTables();
        uint32_t        totalAmoungOfTpcCoeffTableRegs = numOfRegsPerTpcCoeffTable * numOfTpcCoeffTables;

        VERIFY_IS_NULL_POINTER(SYN_DEVICE, pTpcCoeffHbmOffset, "TPC COEFF HBM-Offset iterator");
        VERIFY_IS_NULL_POINTER(SYN_DEVICE, pTpcCoeffAddrRegOffset, "TPC COEFF address-register's offset iterator");

        for (uint32_t table = 0, i = 0; i < totalAmoungOfTpcCoeffTableRegs;
             table++, i += numOfRegsPerTpcCoeffTable, pTpcCoeffHbmOffset++, pTpcCoeffAddrRegOffset++)
        {
            bool isLastTable = (table == (numOfTpcCoeffTables - 1));

            ptrToInt tableAddr;
            // Calc the offset of the specific table
            tableAddr.u64 = coeffTableAddr + *pTpcCoeffHbmOffset;

            // Configure LO addr of coeff table
            uint32_t regAddr = tpcCfgBaseAddress + *pTpcCoeffAddrRegOffset;
            status           = pStream->getScalStream()->addLbwWrite(regAddr, (uint32_t)tableAddr.u32[0], false, send, false);
            if (status != synSuccess)
            {
                LOG_ERR(SYN_DEVICE, "Failed to LbWrite for event {}", status);
                return synFail;
            }
            pTpcCoeffAddrRegOffset++;

            send = isLastEngine && isLastTable;
            // Configure HI addr of coeff table
            regAddr = tpcCfgBaseAddress + *pTpcCoeffAddrRegOffset;
            status  = pStream->getScalStream()->addLbwWrite(regAddr, (uint32_t)tableAddr.u32[1], false, send, false);
            if (status != synSuccess)
            {
                LOG_ERR(SYN_DEVICE, "Failed to LbWrite for event {}", status);
                return synFail;
            }
        }
    }

    return status;
}