#pragma once

#include "define_synapse_common.hpp"
#include "synapse_common_types.h"

#include "runtime/scal/common/infra/scal_includes.hpp"

struct PoolMemoryStatus;

class ScalMemoryPool
{
public:
    ScalMemoryPool(scal_handle_t devHndl, const std::string& name);

    virtual ~ScalMemoryPool() = default;

    synStatus init();

    synStatus allocateDeviceMemory(uint64_t memSize, scal_buffer_handle_t& bufHndl);

    synStatus getDeviceMemoryAddress(const scal_buffer_handle_t& bufHndl, uint32_t& coreAddr, uint64_t& devAddr);

    synStatus releaseDeviceMemory(const scal_buffer_handle_t& bufHndl);

    synStatus getMemoryStatus(PoolMemoryStatus& poolMemoryStatus) const;

private:
    const scal_handle_t   m_devHndl;
    const std::string     m_name;
    scal_pool_handle_t    m_mpHndl;
    scal_memory_pool_info m_mpInfo;
    uint64_t              m_memReserved;
};
