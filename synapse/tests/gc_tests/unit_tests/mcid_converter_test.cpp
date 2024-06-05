#include <gtest/gtest.h>
#include "test_utils.h"
#include "infra/global_conf_manager.h"
#include "gc_tests/unit_tests/graph_optimizer_test.h"
#include "mcid_converter.h"
#include "scal.h"
#include "scoped_configuration_change.h"

class McidConverterTest : public GraphOptimizerTest
{
};

TEST_F(McidConverterTest, basic_no_rollover)
{
    McidConverter      converter;
    uint16_t           res;
    unsigned           rolloverId;
    bool               changeToDegrade;
    unsigned           maxDegradeFromScal = SCAL_MAX_DEGRADE_MCID_COUNT_GAUDI3;
    unsigned           maxDiscardFromScal = SCAL_MAX_DISCARD_MCID_COUNT_GAUDI3;
    unsigned           converterSafetyFactor = 2;

    ASSERT_EQ(converter.getMaxUsedPhysicalDegrade(), 0);
    ASSERT_EQ(converter.getMaxUsedPhysicalDiscard(), 0);

    converter.convertDegrade(0, res);
    ASSERT_EQ(res, 0);
    converter.convertDegrade(1, res);
    ASSERT_EQ(res, 1);
    converter.convertDegrade(99, res);
    ASSERT_EQ(res, 99);
    ASSERT_EQ(converter.getMaxUsedLogicalDegrade(), 99);
    ASSERT_EQ(converter.getMaxUsedPhysicalDegrade(), 99 + converterSafetyFactor);
    converter.convertDegrade(maxDegradeFromScal - 2, res);
    ASSERT_EQ(res, maxDegradeFromScal - 2);
    converter.convertDegrade(maxDegradeFromScal - 1, res);
    ASSERT_EQ(res, 1); // due to the safety factor
    converter.convertDegrade(maxDegradeFromScal, res);
    ASSERT_EQ(res, 2); // due to the safety factor
    ASSERT_EQ(converter.getMaxUsedLogicalDegrade(), maxDegradeFromScal);
    ASSERT_EQ(converter.getMaxUsedPhysicalDegrade(), maxDegradeFromScal);
    converter.convertDegrade(maxDegradeFromScal + 1, res);
    ASSERT_EQ(res, 3); // due to the safety factor
    ASSERT_EQ(converter.getMaxUsedLogicalDegrade(), maxDegradeFromScal + 1);
    ASSERT_EQ(converter.getMaxUsedPhysicalDegrade(), maxDegradeFromScal);
    converter.convertDegrade(maxDegradeFromScal + 12, res);
    ASSERT_EQ(res, 14); // due to the safety factor
    ASSERT_EQ(converter.getMaxUsedLogicalDegrade(), maxDegradeFromScal + 12);
    ASSERT_EQ(converter.getMaxUsedPhysicalDegrade(), maxDegradeFromScal);

    converter.convertDiscard(0, res, rolloverId);
    ASSERT_EQ(res, 0);
    converter.convertDiscard(1, res, rolloverId);
    ASSERT_EQ(res, 1);
    converter.convertDiscard(99, res, rolloverId);
    ASSERT_EQ(res, 99);
    ASSERT_EQ(converter.getMaxUsedLogicalDiscard(), 99);
    ASSERT_EQ(converter.getMaxUsedPhysicalDiscard(), 99 + converterSafetyFactor);
    converter.convertDiscard(9 * 1024, res, rolloverId);
    ASSERT_EQ(res, 9 * 1024);

    converter.convertDiscard(maxDiscardFromScal - converterSafetyFactor, res, rolloverId);
    ASSERT_EQ(res, maxDiscardFromScal - converterSafetyFactor);
    ASSERT_EQ(rolloverId, 0);
    ASSERT_EQ(converter.getMaxUsedLogicalDiscard(), maxDiscardFromScal - converterSafetyFactor);
    ASSERT_EQ(converter.getMaxUsedPhysicalDiscard(), maxDiscardFromScal);

    converter.convertReleaseDiscard(19 * 1024, res, changeToDegrade);
    ASSERT_EQ(res, 19 * 1024);
    converter.convertReleaseDiscard(maxDiscardFromScal - 7, res, changeToDegrade);
    ASSERT_EQ(res, maxDiscardFromScal - 7);
    ASSERT_EQ(changeToDegrade, false);

    // The max hasn't change
    ASSERT_EQ(converter.getMaxUsedLogicalDiscard(), maxDiscardFromScal - converterSafetyFactor);
    ASSERT_EQ(converter.getMaxUsedPhysicalDiscard(), maxDiscardFromScal);
}

TEST_F(McidConverterTest, rollover)
{
    constexpr unsigned MAX_MCID = 10;
    uint16_t           res;
    unsigned           rolloverId;
    bool               changeToDegrade;
    unsigned           converterSafetyFactor = 2;

    ScopedConfigurationChange mcidLimit("CACHE_MAINT_MCID_DISCARD_LIMIT_FOR_TESTING", std::to_string(MAX_MCID));

    // test-case
    {
    McidConverter converter;
    converter.convertDiscard(1, res, rolloverId);
    ASSERT_EQ(res, 1);
    ASSERT_EQ(rolloverId, 0);
    converter.convertDiscard(10, res, rolloverId);
    ASSERT_EQ(res, 10);
    ASSERT_EQ(rolloverId, 0);
    converter.convertDiscard(14, res, rolloverId);
    ASSERT_EQ(res, 4);
    ASSERT_EQ(rolloverId, 1);
    converter.convertDiscard(26, res, rolloverId);
    ASSERT_EQ(res, 6);
    ASSERT_EQ(rolloverId, 2);
    converter.convertDiscard(57, res, rolloverId);
    ASSERT_EQ(res, 7);
    ASSERT_EQ(rolloverId, 5);
    }

    // test-case
    {
    McidConverter converter;
    converter.convertDiscard(1, res, rolloverId);
    ASSERT_EQ(res, 1);
    ASSERT_EQ(rolloverId, 0);
    converter.convertReleaseDiscard(1, res, changeToDegrade);
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, false);
    ASSERT_EQ(converter.getMaxUsedLogicalDiscard(), 1);
    ASSERT_EQ(converter.getMaxUsedPhysicalDiscard(), 1 + converterSafetyFactor);
    }

    // test-case
    {
    McidConverter converter;
    converter.convertDiscard(10, res, rolloverId);
    ASSERT_EQ(res, 10);
    ASSERT_EQ(rolloverId, 0);
    converter.convertReleaseDiscard(10, res, changeToDegrade);
    ASSERT_EQ(res, 10);
    ASSERT_EQ(changeToDegrade, false);
    ASSERT_EQ(converter.getMaxUsedLogicalDiscard(), 10);
    ASSERT_EQ(converter.getMaxUsedPhysicalDiscard(), 10 + converterSafetyFactor);
    }

    // test-case
    {
    McidConverter converter;
    converter.convertDiscard(11, res, rolloverId);
    ASSERT_EQ(res, 1);
    ASSERT_EQ(rolloverId, 1);
    converter.slideRolloverWindow(); // move to window 1
    converter.convertReleaseDiscard(11, res, changeToDegrade); // release from window 1
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true); // corner case #2, window 0 was not released yet
    ASSERT_EQ(converter.getMaxUsedLogicalDiscard(), 11);
    ASSERT_EQ(converter.getMaxUsedPhysicalDiscard(), 10 + converterSafetyFactor);
    }

    // test-case
    {
    McidConverter converter;
    converter.convertDiscard(11, res, rolloverId);
    ASSERT_EQ(res, 1);
    ASSERT_EQ(rolloverId, 1);
    converter.convertReleaseDiscard(1, res, changeToDegrade); // release from window 0
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, false);
    converter.slideRolloverWindow(); // move to window 1
    converter.convertReleaseDiscard(11, res, changeToDegrade); // release from window 1
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, false);
    ASSERT_EQ(converter.getMaxUsedLogicalDiscard(), 11);
    ASSERT_EQ(converter.getMaxUsedPhysicalDiscard(), 10 + converterSafetyFactor);
    }

    // test-case
    {
    McidConverter converter;
    converter.convertDiscard(11, res, rolloverId);
    ASSERT_EQ(res, 1);
    ASSERT_EQ(rolloverId, 1);
    converter.slideRolloverWindow(); // move to window 1
    converter.convertReleaseDiscard(1, res, changeToDegrade); // release from window 0
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true); // corner case #1
    converter.convertReleaseDiscard(11, res, changeToDegrade);  // release from window 1
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, false);
    ASSERT_EQ(converter.getMaxUsedLogicalDiscard(), 11);
    ASSERT_EQ(converter.getMaxUsedPhysicalDiscard(), 10 + converterSafetyFactor);
    }

    // test-case
    {
    McidConverter converter;
    converter.convertDiscard(155, res, rolloverId);
    ASSERT_EQ(res, 5);
    ASSERT_EQ(rolloverId, 15);
    converter.convertReleaseDiscard(1, res, changeToDegrade); // release from window 0
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, false);
    converter.slideRolloverWindow(); // move to window 1
    converter.convertReleaseDiscard(11, res, changeToDegrade); // release from window 1
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, false);
    converter.slideRolloverWindow(); // move to window 2
    converter.convertReleaseDiscard(21, res, changeToDegrade); // release from window 2
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, false);
    converter.slideRolloverWindow(); // move to window 3
    converter.convertReleaseDiscard(31, res, changeToDegrade); // release from window 3
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, false);
    converter.slideRolloverWindow(); // move to window 4
    converter.convertReleaseDiscard(49, res, changeToDegrade); // release 9 from window 4
    ASSERT_EQ(res, 9);
    ASSERT_EQ(changeToDegrade, true); // 9 was not released by previous windows
    ASSERT_EQ(converter.getMaxUsedLogicalDiscard(), 155);
    ASSERT_EQ(converter.getMaxUsedPhysicalDiscard(), 10 + converterSafetyFactor);
    }

    // test-case
    {
    McidConverter converter;
    converter.convertDiscard(155, res, rolloverId);
    ASSERT_EQ(res, 5);
    ASSERT_EQ(rolloverId, 15);
    for (unsigned i = 0; i < 15 ; i++) converter.slideRolloverWindow(); // move to window 15
    converter.convertReleaseDiscard(41, res, changeToDegrade); // release from window 4
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    ASSERT_EQ(converter.getMaxUsedLogicalDiscard(), 155);
    ASSERT_EQ(converter.getMaxUsedPhysicalDiscard(), 10 + converterSafetyFactor);
    }

    // test-case
    {
    McidConverter converter;
    converter.convertDiscard(155, res, rolloverId);
    ASSERT_EQ(res, 5);
    ASSERT_EQ(rolloverId, 15);
    for (unsigned i = 0; i < 15 ; i++) converter.slideRolloverWindow(); // move to window 15
    converter.convertReleaseDiscard(151, res, changeToDegrade); // release from window 15
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    }

    // test-case
    {
    McidConverter converter;
    converter.convertDiscard(155, res, rolloverId);
    ASSERT_EQ(res, 5);
    ASSERT_EQ(rolloverId, 15);
    for (unsigned i = 0; i < 15 ; i++) converter.slideRolloverWindow(); // move to window 15
    converter.convertReleaseDiscard(1, res, changeToDegrade); // release from window 0
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    }

    // test-case
    {
    McidConverter converter;
    converter.convertDiscard(155, res, rolloverId);
    ASSERT_EQ(res, 5);
    ASSERT_EQ(rolloverId, 15);
    for (unsigned i = 0; i < 15 ; i++) converter.slideRolloverWindow(); // move to window 15
    converter.convertReleaseDiscard(1, res, changeToDegrade); // release from window 0
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    converter.convertReleaseDiscard(151, res, changeToDegrade); // release from window 15
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, false);
    }

    // test-case
    {
    McidConverter converter;
    converter.convertDiscard(155, res, rolloverId);
    ASSERT_EQ(res, 5);
    ASSERT_EQ(rolloverId, 15);
    for (unsigned i = 0; i < 15 ; i++) converter.slideRolloverWindow(); // move to window 15
    converter.convertReleaseDiscard(1, res, changeToDegrade); // release from window 0
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    converter.convertReleaseDiscard(151, res, changeToDegrade); // release from window 15
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, false);
    converter.convertReleaseDiscard(81, res, changeToDegrade); // release from window 8
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    }

    // test-case
    {
    McidConverter converter;
    converter.convertDiscard(155, res, rolloverId);
    ASSERT_EQ(res, 5);
    ASSERT_EQ(rolloverId, 15);
    for (unsigned i = 0; i < 15 ; i++) converter.slideRolloverWindow(); // move to window 15
    converter.convertReleaseDiscard(151, res, changeToDegrade); // release from window 15
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    converter.convertReleaseDiscard(1, res, changeToDegrade); // release from window 0
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    converter.convertReleaseDiscard(81, res, changeToDegrade); // release from window 8
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    }

    // test-case
    {
    McidConverter converter;
    converter.convertDiscard(155, res, rolloverId);
    ASSERT_EQ(res, 5);
    ASSERT_EQ(rolloverId, 15);
    for (unsigned i = 0; i < 15 ; i++) converter.slideRolloverWindow(); // move to window 15
    converter.convertReleaseDiscard(41, res, changeToDegrade); // release from window 4
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    converter.convertReleaseDiscard(1, res, changeToDegrade); // release from window 0
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    converter.convertReleaseDiscard(61, res, changeToDegrade); // release from window 6
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    converter.convertReleaseDiscard(151, res, changeToDegrade); // release from window 15
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, false);
    converter.convertReleaseDiscard(71, res, changeToDegrade); // release from window 7
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    }

    // test-case
    {
    McidConverter converter;
    converter.convertDiscard(155, res, rolloverId);
    ASSERT_EQ(res, 5);
    ASSERT_EQ(rolloverId, 15);
    for (unsigned i = 0; i < 15 ; i++) converter.slideRolloverWindow(); // move to window 15
    converter.convertReleaseDiscard(1, res, changeToDegrade); // release from window 0
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    converter.convertReleaseDiscard(151, res, changeToDegrade); // release from window 15
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, false);
    converter.convertReleaseDiscard(92, res, changeToDegrade); // release 2 from window 9
    ASSERT_EQ(res, 2);
    ASSERT_EQ(changeToDegrade, true); // 2 was not released from previous window
    }

    // test-case
    {
    McidConverter converter;
    converter.convertDiscard(257, res, rolloverId);
    ASSERT_EQ(res, 7);
    ASSERT_EQ(rolloverId, 25);
    for (unsigned i = 0; i < 2 ; i++) converter.slideRolloverWindow(); // move to window 2
    converter.convertReleaseDiscard(21, res, changeToDegrade); // release from window 2
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    converter.slideRolloverWindow(); // move to window 3
    converter.convertReleaseDiscard(31, res, changeToDegrade); // release from window 3
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    converter.convertReleaseDiscard(1, res, changeToDegrade); // release from window 0
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    converter.slideRolloverWindow(); // move to window 4
    converter.convertReleaseDiscard(41, res, changeToDegrade); // release from window 4
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, false);
    converter.convertReleaseDiscard(11, res, changeToDegrade); // release from window 1
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, true);
    }

    // test-case, release twice the same mcid without sliding the window - not a valid situation
    {
    McidConverter converter;
    converter.convertDiscard(1, res, rolloverId);
    ASSERT_EQ(res, 1);
    ASSERT_EQ(rolloverId, 0);
    converter.convertReleaseDiscard(1, res, changeToDegrade);
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, false);
    converter.convertReleaseDiscard(1, res, changeToDegrade);
    ASSERT_EQ(res, 1);
    ASSERT_EQ(changeToDegrade, false);
    }
}
