#include "scal_gaudi2.h"
#include "gaudi2/gaudi2_packets.h"
#include "scal_qman_program_gaudi2.h"

bool Scal_Gaudi2::submitQmanWkld(const Qman::Workload &wkld)
{
    unsigned size = 0;
    Qman::Workload::Qid2ProgramInfoMap pdmaMap;
    if (!wkld.getPDmaTransfers().empty())
    {
        pdmaMap[GAUDI2_QUEUE_ID_PDMA_1_0] = {{}, true};
        Qman::Program &pdmaProgram = pdmaMap[GAUDI2_QUEUE_ID_PDMA_1_0].program;

        for (const auto & tran : wkld.getPDmaTransfers())
        {
            pdmaProgram.addCommand(LinDma(tran.dst, tran.src, tran.size));
        }

        size += (pdmaProgram.getSize() + c_cl_size - 1) & ~(c_cl_size - 1);
    }

    for (const auto & sub : wkld.getPrograms())
    {
        for (const auto & prog : sub)
        {
            unsigned progSize = prog.second.program.getSize();
            progSize = (progSize + c_cl_size - 1) & ~(c_cl_size - 1);
            progSize += prog.second.isUpperCp ? 0 : c_cl_size; // add cache line for CPDMA if needed.
            size += progSize;
        }
    }


#ifdef LOG_EACH_COMMAND
    LOG_DEBUG(SCAL,"Workload::{}: size={}", __FUNCTION__, size);
#endif
    void * hostAddr = aligned_alloc(c_cl_size, size);
    if (!hostAddr)
    {
        LOG_ERR(SCAL, "aligned_alloc failed errno={}:д {}. fd={}, size={}",
                errno, std::strerror(errno), m_fd, size);
        assert(0);
        return false;
    }

    const uint64_t deviceAddr = hlthunk_host_memory_map(m_fd, hostAddr, 0, size);
    if (!deviceAddr)
    {
        LOG_ERR(SCAL, "hlthunk_host_memory_map failed errno={}: {}. fd={}, hostAddr={} , size={}",
                errno, std::strerror(errno), m_fd, hostAddr, size);
        assert(0);
        free(hostAddr);
        return false;
    }

    bool ret = true;
    unsigned offset = 0;
    std::vector<uint64_t> seqs;

    for (unsigned mapIdx = 0; mapIdx <= wkld.getPrograms().size(); mapIdx++)
    {
        const auto &sub = mapIdx ? wkld.getPrograms()[mapIdx-1] : pdmaMap;
        if (sub.empty())
        {
            continue;
        }

        struct hl_cs_chunk chunks[GAUDI2_QUEUE_ID_SIZE];
        struct hlthunk_cs_in csIn;
        csIn.chunks_restore = nullptr;
        csIn.chunks_execute = chunks;
        csIn.num_chunks_restore = 0;
        csIn.num_chunks_execute = sub.size();
        csIn.flags = 0;

        // fill cs
        unsigned idx = 0;
        for (const auto &prog : sub)
        {

            memset(&chunks[idx], 0, sizeof(chunks[idx]));
            chunks[idx].cb_handle = deviceAddr + offset;
            chunks[idx].queue_index = prog.first;
            chunks[idx].cs_chunk_flags = HL_CS_CHUNK_FLAGS_USER_ALLOC_CB;

            const unsigned progSize = prog.second.program.getSize();

            if (prog.second.isUpperCp)
            {
                chunks[idx].cb_size = progSize;
                prog.second.program.serialize(((uint8_t*)hostAddr) + offset);
            }
            else
            {
                void *cpDmaPtr = ((uint8_t*)hostAddr) + offset;
                CpDma cpDma(deviceAddr + offset + c_cl_size, progSize);
                chunks[idx].cb_size = cpDma.getSize();
                cpDma.serialize(&cpDmaPtr);
                offset += c_cl_size;
                prog.second.program.serialize(((uint8_t*)hostAddr) + offset);
            }

            offset += (progSize + c_cl_size - 1) & ~(c_cl_size - 1);
            idx++;
        }

        struct hlthunk_cs_out csOut;
        memset(&csOut, 0, sizeof(csOut));
        int rc = hlthunk_command_submission(m_fd, &csIn, &csOut);
        if (rc || (csOut.status != HL_CS_STATUS_SUCCESS))
        {
            LOG_ERR(SCAL,"Workload::{}: hlthunk_command_submission failed. rc={} errno = {} {}", __FUNCTION__, rc, errno, std::strerror(errno));
            ret = false;
            assert(0);
            break;
        }

        if (mapIdx)
        {
            seqs.push_back(csOut.seq);
        }
        else
        {
            // wait fot the PDMA requests before submitting the other programs
            uint32_t status = HL_WAIT_CS_STATUS_COMPLETED;
            rc = hlthunk_wait_for_cs(m_fd, csOut.seq, -1, &status);
            if (rc || (status != HL_WAIT_CS_STATUS_COMPLETED))
            {
                LOG_ERR(SCAL, "Workload::{}: hlthunk_wait_for_cs error. rc = {} status = {} errno = {} {}",
                                __FUNCTION__, rc, status, errno, std::strerror(errno));
                ret = false;
                assert(0);
                break;
            }
        }
    }

    if (ret)
    {
        for (auto seq :seqs)
        {
            uint32_t status = HL_WAIT_CS_STATUS_COMPLETED;
            int rc = hlthunk_wait_for_cs(m_fd, seq, -1, &status);
            if (rc || (status != HL_WAIT_CS_STATUS_COMPLETED))
            {
                LOG_ERR(SCAL, "Workload::{}: hlthunk_wait_for_cs error. rc = {} status = {} errno = {} {}",
                                __FUNCTION__, rc, status, errno, std::strerror(errno));
                ret = false;
                assert(0);
                break;
            }
        }
    }

    int rc = hlthunk_memory_unmap(m_fd, deviceAddr);
    if (rc != 0)
    {
        LOG_ERR(SCAL,"Workload::{}: hlthunk_memory_unmap failed. rc={}", __FUNCTION__, rc);
        ret = false;
        assert(0);
    }

    free(hostAddr);

    return ret;
}