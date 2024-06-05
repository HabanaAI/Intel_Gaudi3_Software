#pragma once

#include "synapse_test.hpp"
#include <memory>
#include <gtest/gtest.h>
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "sim_graph.h"
#include "habana_graph.h"
#include "test_utils.h"
#include "pass_manager.h"
#include "graph_optimizer_test.h"
#include <iostream>

#define NEW_DALI_TEST_PASS(PassName, Id, Priority, PredicateSet, DependencySet)                                        \
    class PassName : public Pass                                                                                       \
    {                                                                                                                  \
    public:                                                                                                            \
        std::function<void()> glambda; /*enables adding functionality to apply */                                      \
        PassName() : Pass(#PassName, Id, Priority, PredicateSet, DependencySet)                                        \
        {                                                                                                              \
            glambda = []() { /*DO NOTHING*/ };                                                                         \
        }                                                                                                              \
        void setApply(std::function<void()>&& f) { glambda = f; }                                                      \
        PassName(PassIDSet dependencySet) : Pass(#PassName, Id, Priority, {}, dependencySet) {}                        \
        bool Apply(HabanaGraph& g) const override                                                                      \
        {                                                                                                              \
            LOG_INFO(GC, "Applying test pass: {}", #PassName);                                                         \
            glambda();                                                                                                 \
            return true;                                                                                               \
        }                                                                                                              \
        virtual pPass create() const override { return pPass(new PassName()); }                                        \
    };

#define PASS_ID(num) static_cast<PassId>(num)
#define PRED_ID(num) static_cast<PredicateId>(num)

class PassManagerTest : public GraphOptimizerTest
{
public:
    static void assertExecutionOrder(const PassManager& passManager, const std::vector<PassId>& expExecOrder);
};

/*************************************************Predicate Id*********************************************************/
#define PREDICATE_1_ID                             PRED_ID(1)
#define PREDICATE_2_ID                             PRED_ID(2)
#define PREDICATE_3_ID                             PRED_ID(3)
#define PREDICATE_4_ID                             PRED_ID(4)
#define PREDICATE_5_ID                             PRED_ID(5)
#define PREDICATE_6_ID                             PRED_ID(6)
#define PREDICATE_7_ID                             PRED_ID(7)
/**************************************************Passes Id***********************************************************/
#define PASS_INVALID_PASS                          PASS_ID(0)
#define PASS_A_ID                                  PASS_ID(1)
#define PASS_B_ID                                  PASS_ID(2)
#define PASS_C_ID                                  PASS_ID(3)
#define PASS_D_ID                                  PASS_ID(4)
#define PASS_E_ID                                  PASS_ID(5)
#define PASS_F_ID                                  PASS_ID(6)
#define PASS_G_ID                                  PASS_ID(7)
#define PASS_H_ID                                  PASS_ID(8)
#define PASS_I_ID                                  PASS_ID(9)
#define PASS_BG_ID                                 PASS_ID(2)
#define PASS_GB_ID                                 PASS_ID(7)
/************************************************Dependency Sets*******************************************************/
#define PASS_A_DEP_SET
#define PASS_B_DEP_SET                             PASS_A_ID
#define PASS_C_DEP_SET                             PASS_A_ID, PASS_B_ID
#define PASS_D_DEP_SET                             PASS_A_ID, PASS_B_ID, PASS_C_ID
#define PASS_E_DEP_SET                             PASS_A_ID, PASS_B_ID, PASS_C_ID, PASS_D_ID
#define PASS_F_DEP_SET                             PASS_A_ID, PASS_B_ID, PASS_C_ID, PASS_D_ID, PASS_E_ID
#define PASS_G_DEP_SET                             PASS_F_ID
#define PASS_H_DEP_SET
#define PASS_I_DEP_SET
#define PASS_BG_DEP_SET                            PASS_A_ID, PASS_GB_ID // cyclic dependency
#define PASS_GB_DEP_SET                            PASS_F_ID             // cyclic dependency
/**************************************************Test Passes*********************************************************/
NEW_DALI_TEST_PASS(PassA,  PASS_A_ID,  PASS_DEF_PRIO, PredicateIDSet{PREDICATE_1_ID}, PassIDSet{PASS_A_DEP_SET});
NEW_DALI_TEST_PASS(PassB,  PASS_B_ID,  PASS_DEF_PRIO, {},                             PassIDSet{PASS_B_DEP_SET});
NEW_DALI_TEST_PASS(PassC,  PASS_C_ID,  PASS_DEF_PRIO, {},                             PassIDSet{PASS_C_DEP_SET});
NEW_DALI_TEST_PASS(PassD,  PASS_D_ID,  PASS_DEF_PRIO, PredicateIDSet{PREDICATE_1_ID}, PassIDSet{PASS_D_DEP_SET});
NEW_DALI_TEST_PASS(PassE,  PASS_E_ID,  PASS_DEF_PRIO, {},                             PassIDSet{PASS_E_DEP_SET});
NEW_DALI_TEST_PASS(PassF,  PASS_F_ID,  PASS_DEF_PRIO, {},                             PassIDSet{PASS_F_DEP_SET});
NEW_DALI_TEST_PASS(PassG,  PASS_G_ID,  PASS_DEF_PRIO, {},                             PassIDSet{PASS_G_DEP_SET});
NEW_DALI_TEST_PASS(PassH,  PASS_H_ID,  PASS_DEF_PRIO, {},                             PassIDSet{PASS_H_DEP_SET});
NEW_DALI_TEST_PASS(PassI,  PASS_I_ID,  PASS_DEF_PRIO, {},                             PassIDSet{PASS_I_DEP_SET});
NEW_DALI_TEST_PASS(PassBG, PASS_BG_ID, PASS_DEF_PRIO, {},                             PassIDSet{PASS_BG_DEP_SET});
NEW_DALI_TEST_PASS(PassGB, PASS_GB_ID, PASS_DEF_PRIO, {},                             PassIDSet{PASS_GB_DEP_SET});

/**********************************************************************************************************************/
/****************************************GROUP Dependency testing******************************************************/
/**********************************************************************************************************************/

// GROUP TESTING : passes
#define PASS_A_ID_GROUP_TEST                       PASS_ID(1)
#define PASS_B_ID_GROUP_TEST                       PASS_ID(2)
#define PASS_C_ID_GROUP_TEST                       PASS_ID(3)
#define PASS_GROUP_A_ID                            PASS_ID(4)// group is an 'empty pass' pass
#define PASS_D_ID_GROUP_TEST                       PASS_ID(5)
#define PASS_E_ID_GROUP_TEST                       PASS_ID(6)
#define PASS_F_ID_GROUP_TEST                       PASS_ID(7)
#define PASS_G_ID_GROUP_TEST                       PASS_ID(8)
#define PASS_GROUP_B_ID                            PASS_ID(9)// group is an 'empty pass' pass
#define PASS_H_ID_GROUP_TEST                       PASS_ID(10)
#define PASS_I_ID_GROUP_TEST                       PASS_ID(11)

// GROUP TESTING:dep set
// group a
#define PASS_A_ID_GROUP_TEST_DEP_SET
#define PASS_B_ID_GROUP_TEST_DEP_SET               PASS_A_ID_GROUP_TEST
#define PASS_C_ID_GROUP_TEST_DEP_SET               PASS_A_ID_GROUP_TEST, PASS_B_ID_GROUP_TEST
#define PASS_GROUP_A_ID_DEP_SET                    PASS_A_ID_GROUP_TEST, PASS_B_ID_GROUP_TEST, PASS_C_ID_GROUP_TEST
// group b
#define PASS_D_ID_GROUP_TEST_DEP_SET               PASS_GROUP_A_ID
#define PASS_E_ID_GROUP_TEST_DEP_SET               PASS_GROUP_A_ID, PASS_D_ID_GROUP_TEST
#define PASS_F_ID_GROUP_TEST_DEP_SET               PASS_GROUP_A_ID, PASS_D_ID_GROUP_TEST
#define PASS_G_ID_GROUP_TEST_DEP_SET               PASS_GROUP_A_ID, PASS_F_ID_GROUP_TEST
#define PASS_GROUP_B_ID_DEP_SET                    PASS_GROUP_A_ID, PASS_D_ID_GROUP_TEST, PASS_E_ID_GROUP_TEST, PASS_F_ID_GROUP_TEST, PASS_G_ID_GROUP_TEST
// group c
#define PASS_H_ID_GROUP_TEST_DEP_SET               PASS_GROUP_B_ID
#define PASS_I_ID_GROUP_TEST_DEP_SET               PASS_GROUP_B_ID

// passes objects
NEW_DALI_TEST_PASS(PassAGroupTest,  PASS_A_ID_GROUP_TEST, PASS_DEF_PRIO, {}, PassIDSet{PASS_A_ID_GROUP_TEST_DEP_SET});
NEW_DALI_TEST_PASS(PassBGroupTest,  PASS_B_ID_GROUP_TEST, PASS_DEF_PRIO, {}, PassIDSet{PASS_B_ID_GROUP_TEST_DEP_SET});
NEW_DALI_TEST_PASS(PassCGroupTest,  PASS_C_ID_GROUP_TEST, PASS_DEF_PRIO, {}, PassIDSet{PASS_C_ID_GROUP_TEST_DEP_SET});
NEW_DALI_TEST_PASS(PassGroupA,      PASS_GROUP_A_ID,      PASS_DEF_PRIO, {}, PassIDSet{PASS_GROUP_A_ID_DEP_SET});
NEW_DALI_TEST_PASS(PassDGroupTest,  PASS_D_ID_GROUP_TEST, PASS_DEF_PRIO, {}, PassIDSet{PASS_D_ID_GROUP_TEST_DEP_SET});
NEW_DALI_TEST_PASS(PassEGroupTest,  PASS_E_ID_GROUP_TEST, PASS_DEF_PRIO, {}, PassIDSet{PASS_E_ID_GROUP_TEST_DEP_SET});
NEW_DALI_TEST_PASS(PassFGroupTest,  PASS_F_ID_GROUP_TEST, PASS_DEF_PRIO, {}, PassIDSet{PASS_F_ID_GROUP_TEST_DEP_SET});
NEW_DALI_TEST_PASS(PassGGroupTest,  PASS_G_ID_GROUP_TEST, PASS_DEF_PRIO, {}, PassIDSet{PASS_G_ID_GROUP_TEST_DEP_SET});
NEW_DALI_TEST_PASS(PassGroupB,      PASS_GROUP_B_ID,      PASS_DEF_PRIO, {}, PassIDSet{PASS_GROUP_B_ID_DEP_SET});
NEW_DALI_TEST_PASS(PassHGroupTest,  PASS_H_ID_GROUP_TEST, PASS_DEF_PRIO, {}, PassIDSet{PASS_H_ID_GROUP_TEST_DEP_SET});
NEW_DALI_TEST_PASS(PassIGroupTest,  PASS_I_ID_GROUP_TEST, PASS_DEF_PRIO, {}, PassIDSet{PASS_I_ID_GROUP_TEST_DEP_SET});
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/************************************** Priority testing **************************************************************/
/**********************************************************************************************************************/
class PassWithPriority : public Pass
{
public:
    PassWithPriority(PassId id, PassPriority priority, PassIDSet dependencies)
    : Pass("", id, priority, {}, dependencies)
    {}
    bool Apply(HabanaGraph& g) const override { return true; }
    pPass create() const override { return nullptr; }
};

NEW_DALI_TEST_PASS(PrioPredPassA,  PASS_A_ID,  1, PredicateIDSet{PREDICATE_1_ID}, {});
NEW_DALI_TEST_PASS(PrioPredPassB,  PASS_B_ID,  0, {},                             {PASS_A_ID});
NEW_DALI_TEST_PASS(PrioPredPassC,  PASS_C_ID,  1, {},                             {PASS_B_ID});
NEW_DALI_TEST_PASS(PrioPredPassD,  PASS_D_ID,  2, PredicateIDSet{PREDICATE_2_ID}, {PASS_C_ID});

/**********************************************************************************************************************/

/**********************************************************************************************************************/
/************************************** Print passes     **************************************************************/
/**********************************************************************************************************************/
template <class BaseGraph>
class GraphForPassPrinting : public BaseGraph
{
public:
    void printAllPasses()
    {
        using std::setw;

        const int priorityWidth = 9;
        const int idWidth = 4;

        this->compile();
        std::cout << setw(priorityWidth) << "Priority"
                  << setw(idWidth) << "ID" << "  Name" << std::endl;
        for (const auto& pass : this->m_passManager->getExecutionOrder())
        {
            std::cout << setw(priorityWidth) << pass->getPriority()
                      << setw(idWidth) << pass->getId()
                      << "  " << pass->getName() << std::endl;
        }
    }
};
/**********************************************************************************************************************/
