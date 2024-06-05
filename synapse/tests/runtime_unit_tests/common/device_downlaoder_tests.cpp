#include <gtest/gtest.h>
#include "runtime/qman/common/device_downloader.hpp"
#include "stream_copy_mock.hpp"
#include <array>
#include "runtime/qman/common/queue_info.hpp"
#include "syn_singleton.hpp"
#include "global_statistics.hpp"

class UTDeviceDownloaderTest : public ::testing::Test
{
public:
    UTDeviceDownloaderTest()
    {
        GlobalConfManager::instance().init(synSingleton::getConfigFilename());
        g_globalStat.configurePostGcfgAndLoggerInit();
    }

    ~UTDeviceDownloaderTest() { g_globalStat.flush(); }

    void validate(synStatus                    status,
                  QueueMock&                   streamCopy,
                  const internalMemcopyParams& rMemcpyParams,
                  QueueInterface*              pPreviousStream);
};

void UTDeviceDownloaderTest::validate(synStatus                    status,
                                      QueueMock&                   streamCopy,
                                      const internalMemcopyParams& rMemcpyParams,
                                      QueueInterface*              pPreviousStream)
{
    EXPECT_EQ(status, synSuccess);
    EXPECT_EQ(streamCopy.m_copyCounter, 1);
    EXPECT_EQ(streamCopy.m_lastMemcpyParams.empty(), false);
    EXPECT_EQ(streamCopy.m_lastMemcpyParams[0].src, rMemcpyParams.front().src);
    EXPECT_EQ(streamCopy.m_lastMemcpyParams[0].dst, rMemcpyParams.front().dst);
    EXPECT_EQ(streamCopy.m_lastMemcpyParams[0].size, rMemcpyParams.front().size);
    EXPECT_EQ(streamCopy.m_lastDirection, MEMCOPY_HOST_TO_DRAM);
    EXPECT_EQ(streamCopy.m_lastIsUserRequest, false);
    EXPECT_EQ(streamCopy.m_pPreviousStream, pPreviousStream);
}

TEST_F(UTDeviceDownloaderTest, downloadProgramCodeBuffer)
{
    QueueMock             streamCopy;
    DeviceDownloader    downloader(streamCopy);
    QueueInterface*       pPreviousStream {nullptr};
    internalMemcopyParams memcpyParams {{.src = 0x1000, .dst = 0x2000, .size = 0x100}};

    synStatus status = downloader.downloadProgramCodeBuffer(0, pPreviousStream, memcpyParams, memcpyParams[0].size);

    validate(status, streamCopy, memcpyParams, pPreviousStream);
}

TEST_F(UTDeviceDownloaderTest, downloadProgramDataBuffer)
{
    QueueMock             streamCopy;
    DeviceDownloader      downloader(streamCopy);
    QueueInterface*       pPreviousStream {nullptr};
    internalMemcopyParams memcpyParams {{.src = 0x1000, .dst = 0x2000, .size = 0x100}};

    synStatus status = downloader.downloadProgramDataBuffer(pPreviousStream, memcpyParams, nullptr);

    validate(status, streamCopy, memcpyParams, pPreviousStream);
}
