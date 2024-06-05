#include "scal_basic_test.h"
#include "scal.h"
#include "hlthunk.h"
#include "logger.h"
#include "scal_test_utils.h"
#include "scal_test_pqm_pkt_utils.h"

#include "scal_gaudi3_sync_monitor.h"

struct ClusterToNumEngines
{
    const char * cluster;
    unsigned numEngines;
};

struct SCALInitOverrideTestParams
{
    std::string override;
    std::vector<ClusterToNumEngines> clusterToNumEngines;
};

class SCALInitOverrideTest : public SCALTestDevice
{
public:
    scal_handle_t scalHandle = nullptr;
    std::vector<const char *> m_compGrpNames;

    void RunTest(SCALInitOverrideTestParams& params);
};

void SCALInitOverrideTest::RunTest(SCALInitOverrideTestParams& params)
{
    int rc = 0;

    const char* envVarName  = "SCAL_CFG_OVERRIDE_PATH";
    const char* envVarValue = "";
    std::string testConfigOverride;
    if (!params.override.empty())
    {
        const char* scalRoot = getenv("SCAL_ROOT");
        ASSERT_NE(scalRoot, nullptr) << "SCAL_ROOT undefined";
        testConfigOverride = scalRoot + params.override;
        envVarValue = getenv(envVarName);
    }

    setenv(envVarName, testConfigOverride.c_str(), 1); // does overwrite

    rc = scal_init(m_fd, ":/default.json", &scalHandle, nullptr);
    ASSERT_EQ(rc, 0);

    for (ClusterToNumEngines clusterToNumEngines : params.clusterToNumEngines)
    {
        scal_cluster_handle_t cluster;
        scal_cluster_info_t   info;

        rc = scal_get_cluster_handle_by_name(scalHandle, clusterToNumEngines.cluster, &cluster);
        ASSERT_EQ(rc, 0);
        rc = scal_cluster_get_info(cluster, &info);
        ASSERT_EQ(rc, 0);
        ASSERT_EQ(info.numEngines, clusterToNumEngines.numEngines);
    }

    for (const char* comp_grp_name : m_compGrpNames)
    {
        scal_comp_group_handle_t comp_grp;
        rc = scal_get_completion_group_handle_by_name(scalHandle, comp_grp_name, &comp_grp);
        ASSERT_EQ(rc, 0);
        scal_completion_group_infoV2_t info;
        rc = scal_completion_group_get_infoV2(comp_grp, &info);
        ASSERT_EQ(rc, 0);
        scal_completion_group_info_t infoV1;
        rc = scal_completion_group_get_info(comp_grp, &infoV1);
        ASSERT_EQ(infoV1.long_so_index, info.long_so_index);
        LOG_INFO(SCAL, "CG name {} LongSo {}", comp_grp_name, info.long_so_index);
    }

    if (envVarValue != nullptr)
    {
        setenv(envVarName, envVarValue, 1); // does overwrite
    }
    else
    {
        unsetenv(envVarName);
    }

    if (scalHandle != nullptr)
    {
        scal_destroy(scalHandle);
    }

    printf("Clean up Finished\n");
}

class SCALInitOverrideTestG3 : public SCALInitOverrideTest,
                               public testing::WithParamInterface<SCALInitOverrideTestParams>
{
public:
    void SetUp() override
    {
        SCALInitOverrideTest::SetUp();
        m_compGrpNames =
        {
            "network_completion_queue_internal_00",
            "network_completion_queue_internal_10",
            "network_completion_queue_internal_20",
            "network_scaleup_init_completion_queue0",
            "network_completion_queue_external_00",
            "network_completion_queue_external_10",
            "network_completion_queue_external_20",
            "compute_completion_queue0", "compute_completion_queue1", "compute_completion_queue2"
        };
    }
};

TEST_P_CHKDEV(SCALInitOverrideTestG3, scal_init_override_test, {GAUDI3})
{
    SCALInitOverrideTestParams params = GetParam();
    RunTest(params);
}

INSTANTIATE_TEST_SUITE_P(, SCALInitOverrideTestG3, testing::Values(
    SCALInitOverrideTestParams({
        "",
        {{"compute_tpc", 64}, {"mme", 8}, {"cme", 1}, {"rotator", 6}, {"network_edma_0", 4}, {"network_edma_slaves", 4}, {"nic_scaleup", 11}, {"nic_scaleout", 2}}
    }),
    SCALInitOverrideTestParams({
        "/configs_override/gaudi3/hd_0_2_4_6_with_network_override.json",
        {{"compute_tpc", 32}, {"mme", 4}, {"cme", 1}, {"nic_scaleup", 11}, {"nic_scaleout", 2}}
    }),
    SCALInitOverrideTestParams({
        "/configs_override/gaudi3/network_only_override.json",
        {{"rotator", 3}, {"network_edma_0", 4}, {"network_edma_slaves", 4}, {"nic_scaleup", 11}, {"nic_scaleout", 2}}
    }),
    SCALInitOverrideTestParams({
        "/configs_override/gaudi3/no_network_hd_0_1_6_7_override.json",
        {{"compute_tpc", 32}, {"mme", 4}, {"cme", 1}}
    }),
    SCALInitOverrideTestParams({
        "/configs_override/gaudi3/no_network_hd_0_2_4_6_full_tpc_override.json",
        {{"compute_tpc", 64}, {"mme", 4}, {"cme", 1}}
    }),
    SCALInitOverrideTestParams({
        "//configs_override/gaudi3/no_network_hd_0_2_4_6_override.json",
        {{"compute_tpc", 32}, {"mme", 4}, {"cme", 1}}
    }),
    SCALInitOverrideTestParams({
        "/configs_override/gaudi3/no_network_full_compute_override.json",
        {{"compute_tpc", 64}, {"mme", 8}, {"cme", 1}}
    })
)
);

class SCALInitOverrideTestG2 : public SCALInitOverrideTest,
                               public testing::WithParamInterface<SCALInitOverrideTestParams>
{
public:
    void SetUp() override
    {
        SCALInitOverrideTest::SetUp();
        m_compGrpNames =
        {
            "network_completion_queue_internal_00",
            "network_completion_queue_internal_10",
            "network_completion_queue_internal_20",
            "network_scaleup_init_completion_queue0",
            "network_completion_queue_external_00",
            "network_completion_queue_external_10",
            "network_completion_queue_external_20",
            "compute_completion_queue0",            "compute_completion_queue1",            "compute_completion_queue2",
            "pdma_rx_completion_queue0",            "pdma_rx_completion_queue1",            "pdma_rx_completion_queue2",
            "pdma_tx_completion_queue0",            "pdma_tx_completion_queue1",            "pdma_tx_completion_queue2",
            "pdma_tx_commands_completion_queue0",   "pdma_tx_commands_completion_queue1",   "pdma_tx_commands_completion_queue2",
            "pdma_device2device_completion_queue0", "pdma_device2device_completion_queue1", "pdma_device2device_completion_queue2",
        };
    }
};

TEST_P_CHKDEV(SCALInitOverrideTestG2, scal_init_override_test, {GAUDI2})
{
    SCALInitOverrideTestParams params = GetParam();
    RunTest(params);
};

INSTANTIATE_TEST_SUITE_P(, SCALInitOverrideTestG2, testing::Values(
    SCALInitOverrideTestParams({
        "",
        {{"compute_tpc", 24}, {"mme", 2}, {"nic_scaleup", 21}, {"nic_scaleout", 3}}
    })
)
);
