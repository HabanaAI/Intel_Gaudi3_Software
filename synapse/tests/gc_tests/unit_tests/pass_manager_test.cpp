//
// Created by daharon on 7/15/19.
//

#include "pass_manager_test.h"
#include "gaudi_graph.h"

void PassManagerTest::assertExecutionOrder(const PassManager& passManager, const std::vector<PassId>& expExecOrder)
{
    int i = 0;

    auto expExecIt = expExecOrder.begin();
    for (pPass pass : passManager.getExecutionOrder())
    {
        ASSERT_EQ(*expExecIt++, pass->getId()) << "Error in iteration " << i << " of execution order assertion.";
        i++;
    }
    ASSERT_EQ(expExecOrder.end(), expExecIt) << "Not all expected passes were run.";
}

TEST_F(PassManagerTest, simplePassesDependency)
{
    Pass* passA = new PassA();
    Pass* passB = new PassB();
    Pass* passC = new PassC();
    Pass* passD = new PassD();
    Pass* passE = new PassE();
    Pass* passF = new PassF();
    Pass* passG = new PassG();
    PassManager passManager;
    GaudiGraph  g;

    passManager.registerPass(pPass(passB));
    passManager.registerPass(pPass(passC));
    passManager.registerPass(pPass(passE));
    passManager.registerPass(pPass(passF));
    passManager.registerPass(pPass(passA));
    passManager.registerPass(pPass(passG));
    passManager.registerPass(pPass(passD));

    passManager.run(g);

    ASSERT_EQ(passManager.getExecutionOrder()[0]->getId(), PASS_A_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[1]->getId(), PASS_B_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[2]->getId(), PASS_C_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[3]->getId(), PASS_D_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[4]->getId(), PASS_E_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[5]->getId(), PASS_F_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[6]->getId(), PASS_G_ID);
}

TEST_F(PassManagerTest, passesCyclicDependency)
{
    GaudiGraph  g;
    PassManager passManager;
    Pass* passA  = new PassA();
    Pass* passBG = new PassBG();//depends on GB
    Pass* passC  = new PassC();
    Pass* passD  = new PassD();
    Pass* passE  = new PassE();
    Pass* passF  = new PassF();
    Pass* passGB = new PassGB();//depends on BG

    passManager.registerPass(pPass(passA));
    passManager.registerPass(pPass(passBG));
    passManager.registerPass(pPass(passC));
    passManager.registerPass(pPass(passD));
    passManager.registerPass(pPass(passE));
    passManager.registerPass(pPass(passF));
    passManager.registerPass(pPass(passGB));

    ASSERT_EQ(passManager.run(g), false) << "Pass manager should not operate with cyclic graph!";
    ASSERT_EQ(passManager.getExecutionList().size(), 0) <<
    "Pass manager should not create execution list in case of cyclic graph";
}

TEST_F(PassManagerTest, samePassTwice)
{
    GaudiGraph  g;
    PassManager passManager;
    Pass* passA  = new PassA();
    Pass* passB  = new PassB();
    Pass* passC  = new PassC();

    pPass pPassB(passB);

    passManager.registerPass(pPass(passA));
    passManager.registerPass(pPassB);
    // register passB again, the registration should fail
    ASSERT_EQ(passManager.registerPass(pPassB), false);
    passManager.registerPass(pPass(passC));

    ASSERT_EQ(passManager.run(g), true);
}

TEST_F(PassManagerTest, multipleExecution)
{
    PassA*      passA            = new PassA();
    PassB*      passB            = new PassB();
    PassC*      passC            = new PassC();
    Pass* passD            = new PassD();
    Pass* passE            = new PassE();
    Pass* passF            = new PassF();
    PassG*      passG            = new PassG();
    Pass* unregisteredPass = new PassH();
    PassManager passManager;
    GaudiGraph  g;

    // 1. passB will trigger passA rerun
    passB->setApply([&passA, &passManager]() mutable -> void
                    {
                        // this lambda expresion capture passManager and passA
                        passManager.reRunPass(passA->getId());
                    });

    // 2. passG will trigger passA rerun
    passG->setApply([&passA, &passManager]() mutable -> void
                    {
                        // this lambda expresion capture passManager and passA
                        passManager.reRunPass(passA->getId());
                    });

    // 3. passC will try to trigger passG rerun
    //    since passG runs after passC, it should not affect the order of execution
    passC->setApply([&passG, &passManager]() mutable -> void
                    {
                        // this lambda expresion capture passManager and passG
                        passManager.reRunPass(passG->getId());
                    });

    // 4. passH never registered to pass manager, pass manager should ignore the rerun request
    passC->setApply([&unregisteredPass, &passManager]() mutable -> void
                    {
                        // this lambda expresion capture passManager and passH
                        passManager.reRunPass(unregisteredPass->getId());
                    });


    passManager.registerPass(pPass(passD));
    passManager.registerPass(pPass(passA));
    passManager.registerPass(pPass(passE));
    passManager.registerPass(pPass(passC));
    passManager.registerPass(pPass(passF));
    passManager.registerPass(pPass(passG));
    passManager.registerPass(pPass(passB));

    passManager.run(g);

    ASSERT_EQ(passManager.getExecutionOrder()[0]->getId(), PASS_A_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[1]->getId(), PASS_B_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[2]->getId(), PASS_A_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[3]->getId(), PASS_C_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[4]->getId(), PASS_D_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[5]->getId(), PASS_E_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[6]->getId(), PASS_F_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[7]->getId(), PASS_G_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[8]->getId(), PASS_A_ID);
    ASSERT_EQ(passManager.getExecutionOrder().size(), 9);

    delete unregisteredPass;
}

TEST_F(PassManagerTest, groupDependency)
{
    GaudiGraph      g;
    PassAGroupTest* passA     = new PassAGroupTest();
    PassBGroupTest* passB     = new PassBGroupTest();
    PassCGroupTest* passC     = new PassCGroupTest();
    PassGroupA*     groupA    = new PassGroupA();
    PassDGroupTest* passD     = new PassDGroupTest();
    PassEGroupTest* passE     = new PassEGroupTest();
    PassFGroupTest* passF     = new PassFGroupTest();
    PassGGroupTest* passG     = new PassGGroupTest();
    PassGroupB*     groupB    = new PassGroupB();
    PassHGroupTest* passH     = new PassHGroupTest();
    PassIGroupTest* passI     = new PassIGroupTest();
    PassManager     passManager;

    passManager.registerPass(pPass(groupB));
    passManager.registerPass(pPass(groupA));
    passManager.registerPass(pPass(passB));
    passManager.registerPass(pPass(passC));
    passManager.registerPass(pPass(passD));
    passManager.registerPass(pPass(passE));
    passManager.registerPass(pPass(passF));
    passManager.registerPass(pPass(passG));
    passManager.registerPass(pPass(passA));
    passManager.registerPass(pPass(passH));
    passManager.registerPass(pPass(passI));

    passManager.run(g);

    ASSERT_EQ(passManager.getExecutionOrder()[0]->getId(),  PASS_A_ID_GROUP_TEST);
    ASSERT_EQ(passManager.getExecutionOrder()[1]->getId(),  PASS_B_ID_GROUP_TEST);
    ASSERT_EQ(passManager.getExecutionOrder()[2]->getId(),  PASS_C_ID_GROUP_TEST);
    ASSERT_EQ(passManager.getExecutionOrder()[3]->getId(),  PASS_GROUP_A_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[4]->getId(),  PASS_D_ID_GROUP_TEST);
    ASSERT_EQ(passManager.getExecutionOrder()[5]->getId(),  PASS_F_ID_GROUP_TEST);
    ASSERT_EQ(passManager.getExecutionOrder()[6]->getId(),  PASS_E_ID_GROUP_TEST);
    ASSERT_EQ(passManager.getExecutionOrder()[7]->getId(),  PASS_G_ID_GROUP_TEST);
    ASSERT_EQ(passManager.getExecutionOrder()[8]->getId(),  PASS_GROUP_B_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[9]->getId(),  PASS_I_ID_GROUP_TEST);
    ASSERT_EQ(passManager.getExecutionOrder()[10]->getId(), PASS_H_ID_GROUP_TEST);
    ASSERT_EQ(passManager.getExecutionOrder().size(), 11);
}

TEST_F(PassManagerTest, basicPredicateTest)
{
    PassA*      passA            = new PassA();
    PassB*      passB            = new PassB();
    PassC*      passC            = new PassC();
    PassManager passManager;
    GaudiGraph  g;

    LOG_DEBUG(GC, "Basic predicate test");
    passC->setApply([&passManager]() mutable -> void
                    {
                        passManager.turnOnPredicate(PREDICATE_1_ID);
                    });

    passManager.registerPass(pPass(passA));
    passManager.registerPass(pPass(passC));
    passManager.registerPass(pPass(passB));

    passManager.run(g);

    ASSERT_EQ(passManager.getExecutionOrder()[0]->getId(), PASS_A_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[1]->getId(), PASS_B_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[2]->getId(), PASS_C_ID);
    ASSERT_EQ(passManager.getExecutionOrder()[3]->getId(), PASS_A_ID);
    ASSERT_EQ(passManager.getExecutionOrder().size(), 4);
}

TEST_F(PassManagerTest, twoPassesRegisteredtoSamePredTest)
{
    /* Passes A and D registered to pred1,
     * The static execution order is A->B->C->D
     * Pass B turning on pred1.
     * Only passA should be dynamically executed after B since passD dependencies are not fulfilled*/
    PassA*      passA            = new PassA();// Registered to pred1
    PassB*      passB            = new PassB();
    PassC*      passC            = new PassC();
    PassD*      passD            = new PassD();// Registered to pred1
    PassManager passManager;
    GaudiGraph  g;

    LOG_DEBUG(GC, "Basic predicate test");
    passB->setApply([&passManager]() mutable -> void
                    {
                        passManager.turnOnPredicate(PREDICATE_1_ID);
                    });

    passManager.registerPass(pPass(passA));
    passManager.registerPass(pPass(passC));
    passManager.registerPass(pPass(passB));
    passManager.registerPass(pPass(passD));

    passManager.run(g);

    // pass D registered to pred1, should not run
    ASSERT_EQ(passManager.getExecutionOrder()[0]->getId(), PASS_A_ID);// Static order
    ASSERT_EQ(passManager.getExecutionOrder()[1]->getId(), PASS_B_ID);// Static order
    ASSERT_EQ(passManager.getExecutionOrder()[2]->getId(), PASS_A_ID);// Dynamic execution
    ASSERT_EQ(passManager.getExecutionOrder()[3]->getId(), PASS_C_ID);// Static order
    ASSERT_EQ(passManager.getExecutionOrder()[4]->getId(), PASS_D_ID);// Static order
    ASSERT_EQ(passManager.getExecutionOrder().size(), 5);
}

TEST_F(PassManagerTest, twoPassesRegisteredtoSamePredBothRun)
{
    PassA*      passA            = new PassA();// Registered to pred1
    PassB*      passB            = new PassB();
    PassC*      passC            = new PassC();
    PassD*      passD            = new PassD();// Registered to pred1
    PassE*      passE            = new PassE();
    PassManager passManager;
    GaudiGraph  g;

    LOG_DEBUG(GC, "Basic predicate test");
    passE->setApply([&passManager]() mutable -> void
                    {
                        passManager.turnOnPredicate(PREDICATE_1_ID);
                    });

    passManager.registerPass(pPass(passA));
    passManager.registerPass(pPass(passC));
    passManager.registerPass(pPass(passB));
    passManager.registerPass(pPass(passD));
    passManager.registerPass(pPass(passE));

    passManager.run(g);

    // pass D registered to pred1, should not run
    ASSERT_EQ(passManager.getExecutionOrder()[0]->getId(), PASS_A_ID);// Static order
    ASSERT_EQ(passManager.getExecutionOrder()[1]->getId(), PASS_B_ID);// Static order
    ASSERT_EQ(passManager.getExecutionOrder()[2]->getId(), PASS_C_ID);// Static order
    ASSERT_EQ(passManager.getExecutionOrder()[3]->getId(), PASS_D_ID);// Static order
    ASSERT_EQ(passManager.getExecutionOrder()[4]->getId(), PASS_E_ID);// Static order
    ASSERT_EQ(passManager.getExecutionOrder()[5]->getId(), PASS_A_ID);// Dynamic execution
    ASSERT_EQ(passManager.getExecutionOrder()[6]->getId(), PASS_D_ID);// Dynamic execution
    ASSERT_EQ(passManager.getExecutionOrder().size(), 7);
}

TEST_F(PassManagerTest, priority_base)
{
    PassManager passManager;
    passManager.setLegacyMode();  // static priority exists only in legacy mode
    passManager.registerPass(pPass(new PassWithPriority(PASS_A_ID, 0, {})));
    passManager.registerPass(pPass(new PassWithPriority(PASS_B_ID, 1, {})));
    passManager.registerPass(pPass(new PassWithPriority(PASS_C_ID, 2, {})));

    GaudiGraph g;
    ASSERT_TRUE(passManager.run(g));

    assertExecutionOrder(passManager, {PASS_C_ID, PASS_B_ID, PASS_A_ID});
}

TEST_F(PassManagerTest, priority_with_dep)
{
    PassManager passManager;
    passManager.setLegacyMode();  // static priority exists only in legacy mode
    passManager.registerPass(pPass(new PassWithPriority(PASS_A_ID, 0, {})));
    passManager.registerPass(pPass(new PassWithPriority(PASS_B_ID, 1, {})));
    passManager.registerPass(pPass(new PassWithPriority(PASS_C_ID, 3, {PASS_A_ID})));
    passManager.registerPass(pPass(new PassWithPriority(PASS_D_ID, 2, {})));

    GaudiGraph g;
    ASSERT_TRUE(passManager.run(g));

    assertExecutionOrder(passManager, {PASS_D_ID, PASS_B_ID, PASS_A_ID, PASS_C_ID});
}

TEST_F(PassManagerTest, multiple_passes_with_same_priority)
{
    PassManager passManager;
    passManager.setLegacyMode();  // static priority exists only in legacy mode
    passManager.registerPass(pPass(new PassWithPriority(PASS_A_ID, 7, {})));
    passManager.registerPass(pPass(new PassWithPriority(PASS_B_ID, 7, {})));
    passManager.registerPass(pPass(new PassWithPriority(PASS_C_ID, 7, {})));
    passManager.registerPass(pPass(new PassWithPriority(PASS_D_ID, 13, {PASS_A_ID})));
    passManager.registerPass(pPass(new PassWithPriority(PASS_E_ID, 13, {PASS_A_ID})));
    passManager.registerPass(pPass(new PassWithPriority(PASS_F_ID, 13, {PASS_A_ID})));
    passManager.registerPass(pPass(new PassWithPriority(PASS_G_ID, 10, {PASS_B_ID})));
    passManager.registerPass(pPass(new PassWithPriority(PASS_H_ID, 10, {PASS_B_ID})));
    passManager.registerPass(pPass(new PassWithPriority(PASS_I_ID, 10, {PASS_B_ID})));

    GaudiGraph g;
    ASSERT_TRUE(passManager.run(g));

    assertExecutionOrder(passManager, {
        PASS_C_ID,

        PASS_B_ID, // 'B' immediately followed by all the high priority dependents
        PASS_I_ID,
        PASS_H_ID,
        PASS_G_ID,

        PASS_A_ID, // 'A' immediately followed by all the high priority dependents
        PASS_F_ID,
        PASS_E_ID,
        PASS_D_ID,
    });
}

/*
 * Test the following:
 * pass A, priority 1, dep: [ ], triggers pass D
 * pass B, priority 0, dep: [A], triggers pass A
 * pass C, priority 1, dep: [B], triggers pass A
 * pass D, priority 2, dep: [C]
 *
 * Static execution list is expected to be:
 * A, B, C, D
 *
 * Final execution Order is expected to be:
 * A, B, A, C, D, A, D
 * explanation:
 * First A triggers D but ignored (D is still scheduled in static order)
 * B triggers A, which is executed before C since their priorities are equal and A comes from the re-run queue.
 * second A triggers D, Ignored again.
 * C triggers A, but D has higher priority so it runs first, then A.
 * A triggers another D run.
 */
TEST_F(PassManagerTest, priority_with_predicates)
{
    PassManager passManager;
    passManager.setLegacyMode();  // static priority exists only in legacy mode
    PrioPredPassA* passA = new PrioPredPassA();
    passA->setApply([&passManager]() mutable -> void { passManager.turnOnPredicate(PREDICATE_2_ID); });
    PrioPredPassB* passB = new PrioPredPassB();
    passB->setApply([&passManager]() mutable -> void { passManager.turnOnPredicate(PREDICATE_1_ID); });
    PrioPredPassC* passC = new PrioPredPassC();
    passC->setApply([&passManager]() mutable -> void { passManager.turnOnPredicate(PREDICATE_1_ID); });
    PrioPredPassD* passD = new PrioPredPassD();

    passManager.registerPass(pPass(passA));
    passManager.registerPass(pPass(passB));
    passManager.registerPass(pPass(passC));
    passManager.registerPass(pPass(passD));

    GaudiGraph g;
    ASSERT_TRUE(passManager.run(g));

    assertExecutionOrder(passManager, {PASS_A_ID, PASS_B_ID, PASS_A_ID, PASS_C_ID, PASS_D_ID, PASS_A_ID, PASS_D_ID});
}
