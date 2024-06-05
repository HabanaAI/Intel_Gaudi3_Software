/*****************************************************************************
 * Copyright (C) 2016 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 * Authors:
 * Tzachi Cohen <tcohen@gaudilabs.com>
 * Oded Gabbay <ogabbay@gaudilabs.com>
 ******************************************************************************
 */

#ifndef COMMAND_BUFFER_HPP
#define COMMAND_BUFFER_HPP

#include <map>
#include <mutex>

#include "settable.h"
#include "synapse_types.h"

struct hl_cs_chunk;

/*
 * *************************************************************************************************
 *
 *  @brief CommandBuffer class implements the following logics:
 *  1.) Abstracts OS details about command buffers.
 *  2.) Creates one long command buffer which may be comprised out of
 *      a set of small distinct command buffers;
 * *************************************************************************************************
 */

class CommandBuffer
{
public:
    static const uint32_t     INVALID_QUEUE_INDEX  = std::numeric_limits<uint32_t>::max();
    static const unsigned int c_defaultCbSize      = 1024 * 4;

    CommandBuffer();
    virtual ~CommandBuffer() {};

    virtual synStatus InitializeCommandBuffer(uint32_t size = c_defaultCbSize) = 0;
    virtual synStatus DestroyCommandBuffer()                                   = 0;
    virtual synStatus MapBuffer()                                              = 0;
    virtual synStatus UnmapBuffer()                                            = 0;
    virtual synStatus FillCBChunk(hl_cs_chunk& args, uint32_t queueOffset = 0);

    synStatus SetBufferToCB(const void* pBuffer, unsigned cb_size, uint64_t* bufferOffset = nullptr);
    synStatus GetPacketFromCB(void* packet, unsigned packet_size, unsigned pkt_offset);
    synStatus WaitForQueueToFinish(unsigned long timeout = 0);

    synStatus FreeBuffer();
    synStatus ClearCB();
    void      SetQueueIndex(uint32_t queueIndex);

    // This method will allow the user to update the mapped-buffer, and update the occupied size
    synStatus UpdateOccupiedSize(uint64_t additionalUsedSize);

    unsigned char* GetBufferMap() const;
    uint32_t       GetOccupiedSize() const;
    uint64_t       GetCbHandle() const;
    uint32_t       GetQueueIndex() const;
    synStatus      GetQueueIndex(uint32_t& queueIndex) const;

protected:
    CommandBuffer(const CommandBuffer& orig) = delete;
    CommandBuffer& operator=(const CommandBuffer& other) = delete;

    // todo: place all variables below under an stl container to support unlimited CB size
    uint64_t           m_cbHandle;
    uint32_t           m_occupiedSize;
    uint32_t           m_totalSize;
    Settable<uint32_t> m_queueIndex;
    unsigned char*     m_bufferMap;

    // For simplicity and readability
    bool m_isCbCreated;
    bool m_isBufferMapped;

    const uint32_t m_pageSize;

    // The buffrer-size that the CP fetches on each call
    // This also defined the size of each packet to be a multiple of that size.
    // Hence, that a buffer must be a multiple of that size
    static const uint64_t m_cpFetchSize;
};

class ExternalCommandBuffer : public CommandBuffer
{
public:
    virtual synStatus InitializeCommandBuffer(uint32_t size = c_defaultCbSize) override;
    virtual synStatus DestroyCommandBuffer() override;
    virtual synStatus MapBuffer() override;
    virtual synStatus UnmapBuffer() override;
};

class MmuMappedCommandBuffer : public CommandBuffer
{
public:
    virtual synStatus InitializeCommandBuffer(uint32_t size = c_defaultCbSize) override;
    virtual synStatus DestroyCommandBuffer() override;
    virtual synStatus MapBuffer() override;
    virtual synStatus UnmapBuffer() override;
    virtual synStatus FillCBChunk(hl_cs_chunk& args, uint32_t queueOffset = 0) override;
};

/*
 * *************************************************************************************************
 *
 *  @brief CommandBufferMap class implements the following operations:
 *  1.) Add    - Adds a CommonBuffer's pair (key, val) to the map
 *  2.) Remove - Removes a CommonBuffer's pair from the map
 *  3.) Clear  - Clears the CommonBufferMap: synDestroy any CommandBuffer on it, and clears the map.
 * *************************************************************************************************
 */

class CommandBufferMap
{
public:
    static CommandBufferMap* GetInstance() { return m_pInstance; }

    ~CommandBufferMap() = default;

    synStatus AddCommandBuffer(unsigned        commandBufferSize,
                               CommandBuffer** ppCommandBuffer,
                               bool            isForceMmuMapped = false);

    synStatus AddCommandBufferUpdateOccupancyAndMap(unsigned        commandBufferSize,
                                                    CommandBuffer** ppCommandBuffer,
                                                    char*&          pCommandBufferData,
                                                    bool            isForceMmuMapped = false);

    synStatus RemoveCommandBuffer(CommandBuffer* pCommandBuffer);

    // REMARK - The Clear() method is expected to be called only by the synDestroy on the MAIN (single) thread.
    //          Hence protection is not a performance issue
    // This Clear methood is called upon release device to prevent memory leaks - thus requires protection
    synStatus Clear(uint32_t& numOfDestroyedElem);
    uint32_t  MapSize();

protected:
    CommandBufferMap() = default;

private:
    typedef std::map<void*, CommandBuffer*> CommandBufferMapping;
    typedef CommandBufferMapping::iterator  CommandBufferMappingIter;

    CommandBufferMap(const CommandBufferMap& orig) = delete;
    CommandBufferMap& operator=(const CommandBufferMap& other) = delete;

    CommandBufferMappingIter _remove(CommandBufferMappingIter& commandBufferItr);
    synStatus                _destroyCommandBuffer(CommandBuffer* pCommandBuffer);

    // STATIC DATA
    static CommandBufferMap* m_pInstance;

    // DATA
    CommandBufferMapping m_commandBufferDB;
    std::mutex           m_commandBufferDBmutex;
};

#endif /* COMMAND_BUFFER_H */
