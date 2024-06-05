#include <assert.h>
#include <cstring>
#include <string>
#include "scal_base.h"
#include "logger.h"
#include "hlthunk.h"
#include "scal_qman_program.h"

using namespace Qman;

Scal::DeviceDmaBuffer::DeviceDmaBuffer()
: m_hostAddr(nullptr)
, m_base(Allocator::c_bad_alloc)
, m_pool(nullptr)
, m_size(0)
, m_alignment(0)
, m_initialized(false)
, m_scal(nullptr)
, m_mappedHostAddr(0)
{
}

Scal::DeviceDmaBuffer::DeviceDmaBuffer(Pool * pool, const uint64_t size, const uint64_t alignment)
: m_hostAddr(nullptr)
, m_base(Allocator::c_bad_alloc)
, m_pool(pool)
, m_size(size)
, m_alignment(alignment)
, m_initialized(true)
, m_scal(pool ? pool->scal : nullptr)
, m_mappedHostAddr(0)
{
    if (!m_pool || !m_size || !m_alignment)
    {
        m_initialized = false;
        LOG_ERR(SCAL,"{}: illegal params", __FUNCTION__);
        assert(0);
    }
}

Scal::DeviceDmaBuffer::DeviceDmaBuffer(const uint64_t deviceAddr, Scal * scal, const uint64_t size, const uint64_t alignment)
: m_hostAddr(nullptr)
, m_base(deviceAddr)
, m_pool(nullptr)
, m_size(size)
, m_alignment(alignment)
, m_initialized(true)
, m_scal(scal)
, m_mappedHostAddr(0)
{
    if (!m_base || !m_size || !m_alignment || !m_scal)
    {
        m_initialized = false;
        LOG_ERR(SCAL,"{}: illegal params", __FUNCTION__);
        assert(0);
    }
}

Scal::DeviceDmaBuffer::~DeviceDmaBuffer()
{
    if (m_initialized)
    {
        if (m_mappedHostAddr)
        {
            bool umapErr = hlthunk_memory_unmap(m_scal->m_fd, m_mappedHostAddr);
            if (umapErr)
            {
                LOG_ERR(SCAL, "{}: failed to unmap host address from the device. fd={}, addr: {:#x}", __FUNCTION__, m_scal->m_fd, m_mappedHostAddr);
                assert(0);
            }
        }

        if ((m_base != Allocator::c_bad_alloc) && m_pool)
        {
            m_pool->allocator->free(m_base);
        }

        if (m_hostAddr && (!m_pool || (m_pool->type == Pool::Type::HBM)))
        {
            free(m_hostAddr);
        }
    }
}

void *Scal::DeviceDmaBuffer::getHostAddress()
{
    if (m_initialized)
    {
        if (!m_hostAddr)
        {
            if (!m_pool || (m_pool->type == Pool::Type::HBM))
            {
                const size_t alignedSize = alignSizeUpPowerOf2Unsigned<c_host_page_size>(m_size);
                if (alignedSize == -1)
                {
                    LOG_ERR(SCAL,"{}: failed to allocate host address. Not initialized", __FUNCTION__);
                    assert(0);
                    return nullptr;
                }
                m_hostAddr = aligned_alloc(c_host_page_size, alignedSize);
            }
            else
            {
                m_base = m_pool->allocator->alloc(m_size, m_alignment);
                if (m_base != Allocator::c_bad_alloc)
                {
                    m_hostAddr = ((uint8_t *) m_pool->hostBase) + m_base;
                }
            }
        }
        return m_hostAddr;
    }
    else
    {
        LOG_ERR(SCAL,"{}: failed to allocate host address. Not initialized", __FUNCTION__);
        assert(0);
        return nullptr;
    }
}

uint32_t Scal::DeviceDmaBuffer::getCoreAddress()
{
    return (uint32_t)getDeviceAddress();
}

uint64_t Scal::DeviceDmaBuffer::getDeviceAddress()
{
    if (m_initialized)
    {
        if (m_pool)
        {
            if (m_base == Allocator::c_bad_alloc)
            {
                m_base = m_pool->allocator->alloc(m_size, m_alignment);
            }

            return (m_base == Allocator::c_bad_alloc) ? 0 : m_pool->deviceBase + m_base;
        }
        else
        {
            return m_base;
        }
    }
    else
    {
        LOG_ERR(SCAL,"{}: failed to allocate device address. Not initialized", __FUNCTION__);
        assert(0);
        return 0;
    }
}

bool Scal::DeviceDmaBuffer::init(Pool * pool, const uint64_t size, const uint64_t alignment)
{
    if (m_initialized)
    {
        LOG_ERR(SCAL, "{}: already initialized", __FUNCTION__);
        assert(0);
        return false;
    }

    m_pool = pool;
    m_size = size;
    m_alignment = alignment;
    m_initialized = m_pool && m_size && m_alignment;
    if (!m_initialized)
    {
        m_initialized = false;
        LOG_ERR(SCAL, "{}: failed to init DeviceDmaBuffer", __FUNCTION__);
        assert(0);
        return false;
    }

    m_scal = m_pool->scal;

    return true;
}

bool Scal::DeviceDmaBuffer::init(const uint64_t deviceAddr, Scal * scal, const uint64_t size, const uint64_t alignment)
{
    if (m_initialized)
    {
        LOG_ERR(SCAL, "{}: already initialized", __FUNCTION__);
        assert(0);
        return false;
    }

    m_scal = scal;
    m_base = deviceAddr;
    m_size = size;
    m_alignment = alignment;
    m_initialized = m_base && m_size && m_alignment && m_scal;
    if (!m_initialized)
    {
        m_initialized = false;
        LOG_ERR(SCAL, "{}: failed to init DeviceDmaBuffer", __FUNCTION__);
        assert(0);
        return false;
    }

    return true;
}


bool Scal::DeviceDmaBuffer::commit(Qman::Workload *wkld)
{
    bool submitRet = true;
    bool umapErr = false;

    if (!m_initialized)
    {
        LOG_ERR(SCAL,"{}: failed to copy dma buff. Not initialized", __FUNCTION__);
        assert(0);
        return false;
    }

    if (!m_pool || (m_pool->type == Pool::Type::HBM))
    {
        uint64_t dst = getDeviceAddress();
        if (!dst)
        {
            LOG_ERR(SCAL,"{}: failed to allocate device address. fd={}", __FUNCTION__, m_scal->m_fd);
            assert(0);
            return false;
        }

        void * srcHost = getHostAddress();
        if (!srcHost)
        {
            LOG_ERR(SCAL,"{}: failed to allocate host address. fd={}", __FUNCTION__, m_scal->m_fd);
            assert(0);
            return false;
        }

        uint64_t srcDevice = hlthunk_host_memory_map(m_scal->m_fd, srcHost, 0, m_size);
        if (!srcDevice)
        {
            LOG_ERR(SCAL,"{}: failed to map host address to the device. fd={}", __FUNCTION__, m_scal->m_fd);
            assert(0);
            return false;
        }

        if (wkld)
        {
            m_mappedHostAddr = srcDevice;
            submitRet = wkld->addPDmaTransfer(srcDevice, dst, m_size);
            if (!submitRet)
            {
                LOG_ERR(SCAL, "{}: failed to submit the linDma wkld to the device. fd={}", __FUNCTION__, m_scal->m_fd);
                assert(0);
            }
        }
        else
        {
            Qman::Workload localWkld;
            submitRet = localWkld.addPDmaTransfer(srcDevice, dst, m_size);
            if (submitRet)
            {
                submitRet = m_scal->submitQmanWkld(localWkld);
                if (!submitRet)
                {
                    LOG_ERR(SCAL, "{}: failed to submit the linDma wkld to the device. fd={}", __FUNCTION__, m_scal->m_fd);
                    assert(0);
                }
            }
            else
            {

                LOG_ERR(SCAL, "{}: failed to add transfer request to the program. fd={}", __FUNCTION__, m_scal->m_fd);
                assert(0);
            }

            umapErr = hlthunk_memory_unmap(m_scal->m_fd, srcDevice);
            if (umapErr)
            {
                LOG_ERR(SCAL, "{}: failed to unmap host address from the device. fd={}", __FUNCTION__, m_scal->m_fd);
                assert(0);
            }

            free(srcHost);
            m_hostAddr = nullptr;
        }
    }

    return submitRet && !umapErr;
}
    