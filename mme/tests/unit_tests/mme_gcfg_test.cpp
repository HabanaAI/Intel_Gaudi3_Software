#include "mme_unit_test.h"
#include "mme_common/mme_global_conf.h"
#include "utils/mme_global_conf_manager.h"
#include <gtest/gtest.h>

class MmeUT_GCFGTest : public MMEUnitTest
{
};

TEST_F(MmeUT_GCFGTest, basic_test)
{
    setenv("envStr", "string_test", 1);
    GlobalConfString envStr("envStr", "empty description", std::string(), MakePublic);
    MmeGlobalConfManager::instance().init("");
    ASSERT_EQ(std::strcmp(envStr.value().c_str(), "string_test"), 0);
}

TEST_F(MmeUT_GCFGTest, basic_bool_test)
{
    setenv("envBoolF", "0", 1);
    GlobalConfBool envBoolF("envBoolF", "empty description", true, MakePublic);

    setenv("envBoolF2", "false", 1);
    GlobalConfBool envBoolF2("envBoolF2", "empty description", true, MakePublic);

    setenv("envBoolT", "1", 1);
    GlobalConfBool envBoolT("envBoolT", "empty description", false, MakePublic);

    setenv("envBoolT2", "true", 1);
    GlobalConfBool envBoolT2("envBoolT2", "empty description", false, MakePublic);

    setenv("envBoolNotValid", "bla", 1);
    GlobalConfBool envBoolNotValid("envBoolNotValid", "empty description", false, MakePublic);

    MmeGlobalConfManager::instance().init("");
    ASSERT_EQ(envBoolF.value(), false);
    ASSERT_EQ(envBoolF2.value(), false);
    ASSERT_EQ(envBoolT.value(), true);
    ASSERT_EQ(envBoolT2.value(), true);
    ASSERT_EQ(envBoolNotValid.value(), false);  // default
}

#ifdef SWTOOLS_DEP
#include "hl_gcfg/hlgcfg.hpp"
#include "hl_gcfg/hlgcfg_item.hpp"

TEST_F(MmeUT_GCFGTest, global_conf_from_string)
{
    // bool check
    GlobalConfBool g_bool("g_bool", "empty description", MakePublic);
    g_bool.setFromString("true");
    ASSERT_EQ(g_bool.value(), true);
    g_bool.setFromString("0");
    ASSERT_EQ(g_bool.value(), false);
    g_bool.setFromString("1");
    ASSERT_EQ(g_bool.value(), true);
    g_bool.setFromString("false");
    ASSERT_EQ(g_bool.value(), false);
}

#endif
