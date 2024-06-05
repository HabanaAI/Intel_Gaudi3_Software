#pragma once

#include "syn_base_test.hpp"
#include "habana_global_conf_runtime.h"
typedef enum _memTestErrorCode
{
    memTestErrorCodeGeneral          = -1,
    memTestErrorCodeHostMallocFail   = -2,
    memTestErrorCodeDeviceMallocFail = -3,
    memTestErrorCodeLaunchFail       = -4
} memTestErrorCode;

class SynFlowMemTests : public SynBaseTest
{
public:
    SynFlowMemTests() : SynBaseTest() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2}); }
    void mem_test_internal(int passOnErr);

protected:
    void allocateWorkspace(unsigned& err, uint64_t freemem, uint64_t topologyWorkspaceSize, synDeviceId deviceId);

private:
    static void _checkStatus(unsigned& err, synStatus status);

    std::vector<uint64_t> m_workspaceAddeVec;
};