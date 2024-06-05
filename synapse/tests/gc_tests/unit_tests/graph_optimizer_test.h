#ifndef _GRAPH_OPTIMIZER_TEST_H_
#define _GRAPH_OPTIMIZER_TEST_H_

#include <gtest/gtest.h>
#include <list>
#include <hl_gcfg/hlgcfg_item.hpp>
#ifdef NDEBUG
#define EXPECT_HB_ASSERT(x_) EXPECT_ANY_THROW((x_));
#else
#define EXPECT_HB_ASSERT(x_) EXPECT_DEATH((x_), "");
#endif

const char NOP_KERNEL_NAME[] = "nop";

class GraphOptimizerTest : public ::testing::Test
{
public:
    GraphOptimizerTest();
    virtual ~GraphOptimizerTest();

protected:
    virtual void SetUp();
    virtual void TearDown();

    virtual void setGlobalConfForTest(hl_gcfg::GcfgItem& gConf, const std::string& stringValue);

private:
    using GConfToOldValue = std::pair<hl_gcfg::GcfgItem*, std::string>;
    std::list<GConfToOldValue> globalConfs;
};

class GoNodeTest : public GraphOptimizerTest
{
};

class GoTpcTest : public GraphOptimizerTest
{
};

class CONV : public GraphOptimizerTest
{
};

class GraphTests : public GraphOptimizerTest
{
};

#endif // _GRAPH_OPTIMIZER_TEST_H_
