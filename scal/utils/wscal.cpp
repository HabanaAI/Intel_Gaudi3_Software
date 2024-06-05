#include <iostream>
#include "scal.h"
#include "hlthunk.h"
#include "logger.h"
#include "scal_test_utils.h"
#include "wscal.h"

#define Nassert(expr)\
     if(expr) {}\
     else {\
      LOG_ERR(SCAL, "assert failed on file {} line {} expr < {} >", __FILE__, __LINE__, #expr );\
      assert(0);\
      printf("critical error in WScal - see scal_log.txt\n");\
      _exit(-1);}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static const unsigned cCommandBufferMinSize = 64*1024;
static const unsigned hostCyclicBufferSize = cCommandBufferMinSize;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
WScal::WScal(const char* config, strVec streams, strVec cgs, strVec clusters, bool skipIfDirectMode,
            strVec directModeCgs, int fd)
{
    //
    //   get fd from hlthunk and init_scal
    //
    if (fd == -1)// e.g. not supplied
    {
        m_scalFd = hlthunk_open(HLTHUNK_DEVICE_DONT_CARE, NULL);
        m_releaseFd = true;
    }
    else
    {
        m_scalFd = fd;
        m_releaseFd = false;
    }
    m_status = (m_scalFd>=0);
    if (!m_status)
    {
        return;
    }
    int rc;
    rc = hlthunk_get_hw_ip_info(m_scalFd, &m_hw_ip);
    printf("Loading scal with config=%s\n",config);
    rc = scal_init(m_scalFd, config, &m_scalHandle, nullptr);
    if (!rc) rc = InitMemoryPools();
    m_skipInDirectMode = skipIfDirectMode;
    if (!rc) rc = InitStreams(streams, cgs, directModeCgs, clusters);
    if (rc)
    {
        printf("Error in WScal initialization\n");
    }
    m_status = rc;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int WScal::InitMemoryPools()
{
    //  Get Shared Host Memory Pool handle for scheduler command buffers
    int rc = scal_get_pool_handle_by_name(m_scalHandle, "host_shared", &m_hostMemPoolHandle);
    Nassert(rc==0);
    //  Get Device Shared HBM Memory Pool handle
    rc = scal_get_pool_handle_by_name(m_scalHandle, "hbm_shared", &m_deviceSharedMemPoolHandle);
    Nassert(rc==0);
    //  Get Device Global HBM Memory Pool handle
    rc = scal_get_pool_handle_by_name(m_scalHandle, "global_hbm", &m_deviceMemPoolHandle);
    Nassert(rc==0);
    if (rc != 0)// needed for release build
    {
        LOG_ERR(SCAL,"{}: error getting scal handles", __FUNCTION__);
    }
    return rc;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int WScal::InitStreams(strVec streams, strVec cgs, strVec directModeCg, strVec clusters)
{
    Nassert (streams.size() == cgs.size() && streams.size() == clusters.size());
    // for each given stream, allocate completion group, and command buffer
    int rc = 0;
    unsigned numStreams = streams.size();
    if (numStreams > MAX_STREAMS) return -1;
    for(unsigned i = 0; (i < numStreams)  && (rc == 0); ++i)
    {
        std::string streamName = streams[i];
        streamBundle* p =  &m_streamsX[i];
        rc = scal_get_stream_handle_by_name(m_scalHandle, streamName.c_str(),  &p->h_stream);
        Nassert(rc==0);
        // set stream priority
        unsigned priority = 1;
        if (streamName == "pdma1")
            priority = 2; // TBD, the whole priority and dup addresses is a mess
        rc = scal_stream_set_priority(p->h_stream, priority );//  TBD

        Nassert(rc==0);

        // Allocate Buffer on the Host shared pool for the commands cyclic buffer for our stream
        rc = scal_allocate_buffer(m_hostMemPoolHandle, hostCyclicBufferSize, &p->h_cmdBuffer);
        Nassert(rc==0);
        // scheduler ctrl command buffer info.  used to get the host/device address
        rc = scal_buffer_get_info(p->h_cmdBuffer, &p->h_cmdBufferInfo);
        Nassert(rc==0);
        // assign ctrlCmdBuffHandle to be the scheduler command buffer of our stream
        rc = scal_stream_set_commands_buffer(p->h_stream, p->h_cmdBuffer);
        Nassert(rc==0);

        // stream info that we need for submission
        //  NOTE! scal_stream_get_info MUST come AFTER scal_stream_set_commands_buffer
        rc = scal_stream_get_info(p->h_stream, &p->h_streamInfo);
        Nassert(rc==0);
        if (p->h_streamInfo.isDirectMode && m_skipInDirectMode)
        {
            return SCAL_UNSUPPORTED_TEST_CONFIG;
        }

        std::string cgName = (p->h_streamInfo.isDirectMode && directModeCg[i].size() > 0) ? directModeCg[i] : cgs[i];
        // since we have 4 instances of "compute_completion_queue", they are named compute_completion_queue0..3
        rc = scal_get_completion_group_handle_by_name(m_scalHandle, cgName.c_str(), &p->h_cg);
        Nassert(rc==0);
        rc = scal_completion_group_get_infoV2(p->h_cg, &p->h_cgInfo);
        Nassert(rc==0);

        std::string clusterName = clusters[i];
        p->h_clusterInfo.numCompletions = -1;
        if (clusterName != "" && !(p->h_streamInfo.isDirectMode))
        {
            rc = scal_get_cluster_handle_by_name(m_scalHandle, clusterName.c_str(), &p->h_cluster);
            Nassert(rc==0);
            rc = scal_cluster_get_info(p->h_cluster, &p->h_clusterInfo);
            Nassert(rc==0);
            //assert(p->h_clusterInfo.numEngines > 0);// on Gaudi3 dummy clusters (for pdma) numEngines is 0)
            Nassert(p->h_clusterInfo.numCompletions > 0);
        }

        p->h_cmd.Init((char *)p->h_cmdBufferInfo.host_address, hostCyclicBufferSize, p->h_streamInfo.command_alignment, getScalDeviceType());
        p->h_cmd.AllowBufferReset(); // since we do wait for completion before adding new commands, assume you can start from the beginning of the buffer

        m_numStreams++;
    }
    return rc;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// get both buffer handle and info
int WScal::getBufferX(PoolType pType, unsigned size,bufferBundle_t* buf)
{
    Nassert(buf && (size > 0));
    scal_pool_handle_t pool;
    switch (pType)
    {
    case HostSharedPool: pool = m_hostMemPoolHandle;break;
    case GlobalHBMPool: pool = m_deviceMemPoolHandle;break;
    case devSharedPool: pool = m_deviceSharedMemPoolHandle;break;
    default:
        Nassert(0);
        return -1;
    }
    int rc = scal_allocate_buffer(pool, size, &buf->h_Buffer);
    Nassert(rc==0);
    rc = scal_buffer_get_info(buf->h_Buffer, &buf->h_BufferInfo);
    Nassert(rc==0);
    return rc;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
streamBundle* WScal::getStreamX(unsigned index)
{
    if (index < m_numStreams)
        return &m_streamsX[index];
    else
        return nullptr;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
WScal::~WScal()
{
    int rc = 0;
    printf("Clean up\n");
    for(unsigned i=0; i < m_numStreams; i++)
    {
        streamBundle* p =  &m_streamsX[i];
        rc = scal_free_buffer(p->h_cmdBuffer);
        Nassert(rc==0);

    }
    if (rc != 0)// needed for release build
    {
        LOG_ERR(SCAL,"{}: error releasing stream buffers", __FUNCTION__);
    }
    scal_destroy(m_scalHandle);
    if (m_releaseFd)
        hlthunk_close(m_scalFd);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
