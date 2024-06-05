#include "graph_optimizer_test.h"

#include "graph_compiler/habana_global_conf.h"
#include "graph_compiler/smf/shape_func_registry.h"
#include "infra/global_conf_manager.h"
#include "kernel_db.h"

GraphOptimizerTest::GraphOptimizerTest()
{
    GlobalConfManager::instance().init("");
}

GraphOptimizerTest::~GraphOptimizerTest()
{
}

void GraphOptimizerTest::SetUp()
{
    GlobalConfManager::instance().setGlobalConf("ENABLE_EXPERIMENTAL_FLAGS", "true");
    setGlobalConfForTest(GCFG_ALLOW_DUPLICATE_KERNELS, "1");
    setGlobalConfForTest(GCFG_INTERNAL_TEST, "true");
    CREATE_LOGGER(GO_TEST, "graph_optimizer_tests.txt", 1024 * 1024, 1);
    CREATE_LOGGER(SYN_TEST, "graph_optimizer_tests.txt", 1024 * 1024, 1);
    ShapeFuncRegistry::instance().init(synDeviceType::synDeviceTypeInvalid);
    KernelDB::instance().init(tpc_lib_api::DEVICE_ID_MAX);
    KernelDB::instance().registerSif();
}

void GraphOptimizerTest::TearDown()
{
    KernelDB::instance().clear();
    DROP_LOGGER(GO_TEST);
    DROP_LOGGER(SYN_TEST);
    std::remove(".used");

    for (auto gConfValue : globalConfs)
    {
        gConfValue.first->setFromString(gConfValue.second);
    }
    globalConfs.clear();
    ShapeFuncRegistry::instance().destroy();
}

void GraphOptimizerTest::setGlobalConfForTest(hl_gcfg::GcfgItem& gConf, const std::string& stringValue)
{
    globalConfs.push_front(std::make_pair(&gConf, gConf.getValueStr()));
    gConf.setFromString(stringValue);
}
