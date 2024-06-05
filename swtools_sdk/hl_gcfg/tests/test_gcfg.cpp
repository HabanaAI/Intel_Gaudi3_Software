#include <gtest/gtest.h>
#include <fstream>
#include "global_conf.hpp"
#include <hl_gcfg/hlgcfg.hpp>
TEST(hl_config_test, basic)
{
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 999);
    ASSERT_TRUE(GCFG_TEST_VALUE_U64.setFromString("500"));
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 500);
    ASSERT_TRUE(GCFG_TEST_VALUE_U64.setFromString("0x500"));
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 0x500);
    ASSERT_TRUE(GCFG_TEST_VALUE_U64.setValue(600));
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 600);

    ASSERT_EQ(GCFG_TEST_VALUE_BOOL.value(), false);
    ASSERT_TRUE(GCFG_TEST_VALUE_BOOL.setFromString("true"));
    ASSERT_EQ(GCFG_TEST_VALUE_BOOL.value(), true);
    ASSERT_TRUE(GCFG_TEST_VALUE_BOOL.setFromString("false"));
    ASSERT_EQ(GCFG_TEST_VALUE_BOOL.value(), false);
    ASSERT_TRUE(GCFG_TEST_VALUE_BOOL.setFromString("1"));
    ASSERT_EQ(GCFG_TEST_VALUE_BOOL.value(), true);
    ASSERT_TRUE(GCFG_TEST_VALUE_BOOL.setFromString("0"));
    ASSERT_EQ(GCFG_TEST_VALUE_BOOL.value(), false);
    ASSERT_TRUE(GCFG_TEST_VALUE_BOOL.setValue(true));
    ASSERT_EQ(GCFG_TEST_VALUE_BOOL.value(), true);
}

TEST(hl_config_test, basic_error_handling)
{
    ASSERT_FALSE(GCFG_TEST_VALUE_U64.setFromString("500.0"));
    ASSERT_FALSE(GCFG_TEST_VALUE_U64.setFromString("xvy500.0"));
    ASSERT_FALSE(GCFG_TEST_VALUE_BOOL.setFromString("tru"));
    ASSERT_FALSE(GCFG_TEST_VALUE_BOOL.setFromString("1.0"));
}

TEST(hl_config_test, string_access)
{
    ASSERT_TRUE(GCFG_TEST_VALUE_BOOL.setValue(true));
    ASSERT_TRUE(hl_gcfg::getGcfgItemValue("TEST_VALUE_BOOL"));
    ASSERT_EQ(hl_gcfg::getGcfgItemValue("TEST_VALUE_BOOL").value(), "1");
    ASSERT_TRUE(GCFG_TEST_VALUE_BOOL.setValue(false));
    ASSERT_EQ(hl_gcfg::getGcfgItemValue("TEST_VALUE_BOOL").value(), "0");

    ASSERT_FALSE(hl_gcfg::setGcfgItemValue("TEST_VALUE_BOOL", "true"));
    ASSERT_TRUE(hl_gcfg::setGcfgItemValue("TEST_VALUE_BOOL", "true", true));
    ASSERT_EQ(hl_gcfg::getGcfgItemValue("TEST_VALUE_BOOL").value(), "1");

    ASSERT_FALSE(hl_gcfg::setGcfgItemValue("TEST_VALUE_BOOL", "false"));
    ASSERT_TRUE(hl_gcfg::setGcfgItemValue("TEST_VALUE_BOOL", "false", true));
    ASSERT_EQ(hl_gcfg::getGcfgItemValue("TEST_VALUE_BOOL").value(), "0");

    ASSERT_TRUE(hl_gcfg::setGcfgItemValue("TEST_VALUE_BOOL", "1", true));
    ASSERT_EQ(hl_gcfg::getGcfgItemValue("TEST_VALUE_BOOL").value(), "1");

    ASSERT_TRUE(hl_gcfg::setGcfgItemValue("TEST_VALUE_BOOL", "0", true));
    ASSERT_EQ(hl_gcfg::getGcfgItemValue("TEST_VALUE_BOOL").value(), "0");

}

TEST(hl_config_test, env_var)
{
    hl_gcfg::reset();
    GCFG_TEST_VALUE_U64.setValue(100);
    setenv("TEST_VALUE_U64", "900", 1);

    // private access fails
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 100);
    ASSERT_FALSE(GCFG_TEST_VALUE_U64.updateFromEnv(false));
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 100);

    // force update private value
    ASSERT_TRUE(GCFG_TEST_VALUE_U64.updateFromEnv(true));
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 900);

    setenv("TEST_VALUE_U64", "999", 1);
    hl_gcfg::reset();
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 999);

    setenv("TEST_PUBLIC_VALUE_U64", "1", 1);
    hl_gcfg::reset();
    ASSERT_EQ(GCFG_TEST_PUBLIC_UINT64.value(), 1);
    // the same value is allowed to set - no error
    ASSERT_EQ(GCFG_TEST_PUBLIC_UINT64.setFromString("1").has_error(), false);
    // a different value is forbidden to set
    ASSERT_EQ(GCFG_TEST_PUBLIC_UINT64.setFromString("12").errorCode(), hl_gcfg::ErrorCode::valueWasAlreadySetFromEnv);

}

TEST(hl_config_test, private_vars)
{
    setenv(hl_gcfg::getEnableExperimentalFlagsPrimaryName().c_str(), "false", 1);
    setenv(GCFG_TEST_VALUE_U64.primaryName().c_str(), "0", 1);

    hl_gcfg::reset();
    // private access fails
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), GCFG_TEST_VALUE_U64.getDefaultValue(hl_gcfg::InvalidDeviceType));

    setenv(hl_gcfg::getEnableExperimentalFlagsPrimaryName().c_str(), "true", 1);
    setenv(GCFG_TEST_VALUE_U64.primaryName().c_str(), "0", 1);

    hl_gcfg::reset();
    // private access fails
    ASSERT_NE(GCFG_TEST_VALUE_U64.value(), GCFG_TEST_VALUE_U64.getDefaultValue(hl_gcfg::InvalidDeviceType));
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 0);

    unsetenv(hl_gcfg::getEnableExperimentalFlagsPrimaryName().c_str());
    unsetenv(GCFG_TEST_VALUE_U64.primaryName().c_str());
}

TEST(hl_config_test, env_var_2_values)
{
    hl_gcfg::reset();
    GCFG_TEST_VALUE_U64.setValue(100);
    setenv(GCFG_TEST_VALUE_U64.primaryName().c_str(), "900", 1);
    setenv("TEST_VALUE_U64_1", "900", 1);

    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 100);
    ASSERT_FALSE(GCFG_TEST_VALUE_U64.updateFromEnv(false));
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 100);

    ASSERT_TRUE(GCFG_TEST_VALUE_U64.updateFromEnv(true));
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 900);

    setenv("TEST_VALUE_U64_1", "90", 1);
    ASSERT_FALSE(GCFG_TEST_VALUE_U64.updateFromEnv(true));
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 900);

    setenv(GCFG_TEST_VALUE_U64.primaryName().c_str(), "999", 1);
    hl_gcfg::reset();
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 999);

    setenv(GCFG_TEST_VALUE_BOOL.primaryName().c_str(), "true", 1);
    setenv("TEST_VALUE_BOOL_1", "true", 1);

    GCFG_TEST_VALUE_BOOL.setValue(false);
    ASSERT_EQ(GCFG_TEST_VALUE_BOOL.value(), false);
    ASSERT_FALSE(GCFG_TEST_VALUE_BOOL.updateFromEnv(false));
    ASSERT_EQ(GCFG_TEST_VALUE_BOOL.value(), false);

    ASSERT_TRUE(GCFG_TEST_VALUE_BOOL.updateFromEnv(true));
    ASSERT_EQ(GCFG_TEST_VALUE_BOOL.value(), true);

    setenv("TEST_VALUE_BOOL_1", "True", 1);
    ASSERT_TRUE(GCFG_TEST_VALUE_BOOL.updateFromEnv(true));
    ASSERT_EQ(GCFG_TEST_VALUE_BOOL.value(), true);

    setenv("TEST_VALUE_BOOL_1", "1", 1);
    ASSERT_TRUE(GCFG_TEST_VALUE_BOOL.updateFromEnv(true));
    ASSERT_EQ(GCFG_TEST_VALUE_BOOL.value(), true);

    setenv("TEST_VALUE_BOOL_1", "0", 1);
    ASSERT_FALSE(GCFG_TEST_VALUE_BOOL.updateFromEnv(true));
    ASSERT_EQ(GCFG_TEST_VALUE_BOOL.value(), true);

    setenv("TEST_VALUE_BOOL_1", "false", 1);
    ASSERT_FALSE(GCFG_TEST_VALUE_BOOL.updateFromEnv(true));
    ASSERT_EQ(GCFG_TEST_VALUE_BOOL.value(), true);

    unsetenv(GCFG_TEST_VALUE_U64.primaryName().c_str());
    unsetenv("TEST_VALUE_BOOL_1");
}

TEST(hl_config_test, bool_two_names)
{
    hl_gcfg::reset();
    GCFG_TWO_NAMES.setValue(false);
    setenv(GCFG_TWO_NAMES.primaryName().c_str(), "True", 1);
    setenv("TWO_NAMES_1", "true", 1);

    ASSERT_EQ(GCFG_TWO_NAMES.value(), false);
    ASSERT_TRUE(GCFG_TWO_NAMES.updateFromEnv(false));
    ASSERT_EQ(GCFG_TWO_NAMES.value(), true);

    setenv(GCFG_TWO_NAMES.primaryName().c_str(), "False", 1);
    setenv("TWO_NAMES_1", "0", 1);
    hl_gcfg::reset();

    ASSERT_EQ(GCFG_TWO_NAMES.value(), false);

    setenv(GCFG_TWO_NAMES.primaryName().c_str(), "True", 1);
    setenv("TWO_NAMES_1", "1", 1);
    hl_gcfg::reset();
    ASSERT_EQ(GCFG_TWO_NAMES.value(), true);

    setenv(GCFG_TWO_NAMES.primaryName().c_str(), "1", 1);
    setenv("TWO_NAMES_1", "true", 1);
    GCFG_TWO_NAMES.setValue(false);
    hl_gcfg::reset();
    ASSERT_EQ(GCFG_TWO_NAMES.value(), true);

    unsetenv(GCFG_TWO_NAMES.primaryName().c_str());
    unsetenv("TWO_NAMES_1");
}

TEST(hl_config_test, two_vars)
{
    hl_gcfg::reset();
    GCFG_TEST_VALUE_U64.setValue(10);
    unsetenv("ENABLE_EXPERIMENTAL_FLAGS");
    setenv(GCFG_TEST_VALUE_U64.primaryName().c_str(), "25", 1);
    setenv(GCFG_TEST_PUBLIC_UINT64.primaryName().c_str(), "25", 1);

    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 10);
    ASSERT_EQ(GCFG_TEST_VALUE_U64_2.value(), 999);

    hl_gcfg::reset();
    // private
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 999);
    ASSERT_EQ(GCFG_TEST_VALUE_U64_2.value(), 999);

    ASSERT_EQ(GCFG_TEST_PUBLIC_UINT64.value(), 25);
    ASSERT_EQ(GCFG_TEST_PUBLIC_UINT64_2.value(), 25);

    unsetenv(GCFG_TEST_VALUE_U64.primaryName().c_str());
    unsetenv(GCFG_TEST_PUBLIC_UINT64.primaryName().c_str());
}

using hl_gcfg::MakePrivate;
using hl_gcfg::MakePublic;

TEST(hl_config_test, two_vars_aliases)
{
    unsetenv("ENABLE_EXPERIMENTAL_FLAGS");
    setenv("ALIAS_LOCAL", "25", 1);
    {
        GlobalConfUint64 GCFG_TEST_PUBLIC_UINT64_LOCAL(
                "TEST_PUBLIC_VALUE_U64_LOCAL",
                {"ALIAS_LOCAL"},
                "comment (0 == default)",
                999,
                MakePublic);
        // create and destroy the second one
        {
            GlobalConfUint64 GCFG_TEST_PUBLIC_UINT64_LOCAL_2(
                    "TEST_PUBLIC_VALUE_U64_LOCAL",
                    {"ALIAS_LOCAL"},
                    "comment (0 == default)",
                    999,
                    MakePublic);
            hl_gcfg::reset();
            ASSERT_EQ(GCFG_TEST_PUBLIC_UINT64_LOCAL.value(), 25);
            ASSERT_EQ(GCFG_TEST_PUBLIC_UINT64_LOCAL_2.value(), 25);
            auto val = hl_gcfg::getGcfgItemValue("ALIAS_LOCAL");
            ASSERT_TRUE(val.has_value());
            ASSERT_EQ(val.value(), "25");
        }
        auto val = hl_gcfg::getGcfgItemValue("ALIAS_LOCAL");
        ASSERT_TRUE(val.has_value());
        ASSERT_EQ(val.value(), "25");
    }

    auto val = hl_gcfg::getGcfgItemValue("ALIAS_LOCAL");
    ASSERT_TRUE(val.has_error());

    unsetenv("ALIAS_LOCAL");
}

TEST(hl_config_test, device_type)
{
    hl_gcfg::reset();
    ASSERT_EQ(GCFG_DEVICE_DEPENDENT_BOOL.value(), true);
    ASSERT_EQ(GCFG_DEVICE_DEPENDENT_UINT64.value(), 100);
    hl_gcfg::setDeviceType(1);
    ASSERT_EQ(GCFG_DEVICE_DEPENDENT_BOOL.value(), false);
    ASSERT_EQ(GCFG_DEVICE_DEPENDENT_UINT64.value(), 10);
    hl_gcfg::setDeviceType(2);
    ASSERT_EQ(GCFG_DEVICE_DEPENDENT_BOOL.value(), true);
    ASSERT_EQ(GCFG_DEVICE_DEPENDENT_UINT64.value(), 20);
    hl_gcfg::setDeviceType(0);
    ASSERT_EQ(GCFG_DEVICE_DEPENDENT_BOOL.value(), true);
    ASSERT_EQ(GCFG_DEVICE_DEPENDENT_UINT64.value(), 100);
    hl_gcfg::setDeviceType(10);
    ASSERT_EQ(GCFG_DEVICE_DEPENDENT_BOOL.value(), true);
    ASSERT_EQ(GCFG_DEVICE_DEPENDENT_UINT64.value(), 100);
    hl_gcfg::setDeviceType(1);
    ASSERT_EQ(GCFG_DEVICE_DEPENDENT_BOOL.value(), false);
    ASSERT_EQ(GCFG_DEVICE_DEPENDENT_UINT64.value(), 10);

}

TEST(hl_config_test, observers)
{
    ASSERT_EQ(GCFG_CHECK_SECTION_OVERLAP.value(), true);
    ASSERT_EQ(GCFG_MAKE_CTRL_DEP_SOFT.value(), true);

    ASSERT_TRUE(GCFG_CHECK_SECTION_OVERLAP.setValue(false));
    ASSERT_EQ(GCFG_CHECK_SECTION_OVERLAP.value(), false);
    ASSERT_EQ(GCFG_MAKE_CTRL_DEP_SOFT.value(), false);

    ASSERT_TRUE(GCFG_CHECK_SECTION_OVERLAP.setFromString("True"));
    ASSERT_EQ(GCFG_CHECK_SECTION_OVERLAP.value(), true);
    ASSERT_EQ(GCFG_MAKE_CTRL_DEP_SOFT.value(), true);

    ASSERT_TRUE(GCFG_CHECK_SECTION_OVERLAP.setFromString("0"));
    ASSERT_EQ(GCFG_CHECK_SECTION_OVERLAP.value(), false);
    ASSERT_EQ(GCFG_MAKE_CTRL_DEP_SOFT.value(), false);

    ASSERT_TRUE(GCFG_CHECK_SECTION_OVERLAP.setFromString("1"));
    ASSERT_EQ(GCFG_CHECK_SECTION_OVERLAP.value(), true);
    ASSERT_EQ(GCFG_MAKE_CTRL_DEP_SOFT.value(), true);

}

TEST(hl_config_test, load_from_file_save_to_file) {
    const std::string_view fileContent = R"(
CHECK_SECTION_OVERLAP_CHECK=false
MAKE_CTRL_DEP_SOFT=false
TEST_VALUE_U64=117
)";
    std::string fileName = hl_logger::getLogsFolderPath() + "/test.ini";
    std::ofstream ofs(fileName);
    ofs.write(fileContent.data(), fileContent.size());
    ofs.close();
    hl_gcfg::reset();

    ASSERT_EQ(GCFG_CHECK_SECTION_OVERLAP.value(), true);
    ASSERT_EQ(GCFG_MAKE_CTRL_DEP_SOFT.value(), true);
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 999);

    hl_gcfg::loadFromFile(fileName);

    ASSERT_EQ(GCFG_CHECK_SECTION_OVERLAP.value(), false);
    ASSERT_EQ(GCFG_MAKE_CTRL_DEP_SOFT.value(), false);
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 117);

    GCFG_TEST_VALUE_U64.setValue(789);
    GCFG_MAKE_CTRL_DEP_SOFT.setValue(true);

    hl_gcfg::saveToFile(fileName);
    hl_gcfg::reset();

    ASSERT_EQ(GCFG_CHECK_SECTION_OVERLAP.value(), true);
    ASSERT_EQ(GCFG_MAKE_CTRL_DEP_SOFT.value(), true);
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 999);

    hl_gcfg::loadFromFile(fileName);

    ASSERT_EQ(GCFG_CHECK_SECTION_OVERLAP.value(), false);
    ASSERT_EQ(GCFG_MAKE_CTRL_DEP_SOFT.value(), true);
    ASSERT_EQ(GCFG_TEST_VALUE_U64.value(), 789);

}