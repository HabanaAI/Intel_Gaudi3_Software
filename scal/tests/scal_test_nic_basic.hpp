#include "scal_basic_test.h"
#include "rdma_core/rdma_core_lib_loader.hpp"
#include "scal.h"
#include "logger.h"
#include "scal_test_utils.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <vector>

struct TestNicConfig
{
    const char *scheduler_name;
    const char *stream_name;
    const unsigned schedType;
};

class SCALNicTest : public SCALTestDevice,
                    public testing::WithParamInterface<TestNicConfig>
{
public:
    void nicBasicTest(uint32_t wrapCount = 0);
    void sendNopToNicSched(const TestNicConfig *testConfig, const scal_handle_t &scalHandle, uint32_t expectedSignals, std::string cgName, uint32_t wrapCount = 0, bool isMaster = true);
    void nicMultiSchedTest(const TestNicConfig testConfigs[]);

    virtual void    initNics() {};
    virtual void    releaseNics() {};
    virtual uint8_t getSchedNopOpcode(uint32_t sched_type) = 0;
    virtual uint8_t getSchedAllocNicBarrierOpcode(uint32_t sched_type) = 0;
    virtual uint8_t getSchedLbwWriteOpcode(uint32_t sched_type) = 0;
protected:
    virtual void SetUp();
    virtual void TearDown();
    scal_handle_t m_scalHandle = 0;
};

class SCALGaudi2NicTest : public SCALNicTest
{
public:
    uint8_t getSchedNopOpcode(uint32_t sched_type) override;
    uint8_t getSchedAllocNicBarrierOpcode(uint32_t sched_type) override;
    uint8_t getSchedLbwWriteOpcode(uint32_t sched_type) override;
};

struct ibv_context;
class SCALGaudi3NicTest : public SCALNicTest
{
public:
    void    initNics() override;
    void    releaseNics() override;
    uint8_t getSchedNopOpcode(uint32_t sched_type) override;
    uint8_t getSchedAllocNicBarrierOpcode(uint32_t sched_type) override;
    uint8_t getSchedLbwWriteOpcode(uint32_t sched_type) override;
private:
    IbvLibFunctions m_ibvLibFuncs;
    struct ibv_context* m_ibctx = nullptr;
    std::vector<struct hlibdv_usr_fifo*> m_hlibdvUsrFifos;
    unsigned m_nicUserDbFifoParamsCount = 0;
};