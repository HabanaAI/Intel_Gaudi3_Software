#include <cstdint>
#include <gtest/gtest.h>
#include "wcm_cs_querier_mock.hpp"
#include "wcm_observer_mock.hpp"
#include "runtime/qman/common/wcm/work_completion_manager.hpp"
#include "defs.h"
#include "hpp/syn_context.hpp"
#include "runtime/common/recipe/recipe_tensor_processor.hpp"

class UTGaudiWcmTest : public ::testing::Test
{
    syn::Context context;

public:
    void addCs(WorkCompletionManagerInterface& rWcm,
               WcmPhysicalQueuesId             physicalQueuesId,
               WcmObserverRecorderMock*        pObserver,
               const std::vector<uint64_t>&    rCsHandles);

    void basic_test_3_observers(const std::array<WcmPhysicalQueuesId, 3>&             physicalQueuesId,
                                const std::deque<std::array<std::set<uint64_t>, 32>>& rQueryPermutations,
                                const std::array<std::set<uint64_t>, 3>&              rCsHandles);

    void query_test_3_observers(const std::array<WcmPhysicalQueuesId, 3>&             physicalQueuesId,
                                const std::deque<std::array<std::set<uint64_t>, 32>>& rQueryPermutations,
                                const std::array<std::set<uint64_t>, 3>&              rCsHandles);
};

void UTGaudiWcmTest::addCs(WorkCompletionManagerInterface& rWcm,
                           WcmPhysicalQueuesId             physicalQueuesId,
                           WcmObserverRecorderMock*        pObserver,
                           const std::vector<uint64_t>&    rCsHandles)
{
    WcmCsHandleQueue csHandles;
    for (auto csHandle : rCsHandles)
    {
        csHandles.push_back(csHandle);
    }

    pObserver->addCs(csHandles);

    for (auto csHandle : rCsHandles)
    {
        rWcm.addCs(physicalQueuesId, pObserver, csHandle);
    }
}

void UTGaudiWcmTest::basic_test_3_observers(const std::array<WcmPhysicalQueuesId, 3>&             physicalQueuesId,
                                            const std::deque<std::array<std::set<uint64_t>, 32>>& rQueryPermutations,
                                            const std::array<std::set<uint64_t>, 3>&              rCsHandles)
{
    WcmCsQuerierCheckerMock        querier {rQueryPermutations};
    std::deque<std::set<uint64_t>> observerInput0 {rCsHandles[0]};
    std::deque<std::set<uint64_t>> observerInput1 {rCsHandles[1]};
    std::deque<std::set<uint64_t>> observerInput2 {rCsHandles[2]};

    std::array<WcmObserverAdvanceCheckerMock, 3> observers {observerInput0, observerInput1, observerInput2};

    {
        WorkCompletionManager      wcm(&querier);

        for (unsigned iter = 0; iter < rCsHandles.size(); iter++)
        {
            for (auto csHandle : rCsHandles[iter])
            {
                wcm.addCs(physicalQueuesId[iter], &observers[iter], csHandle);
            }
        }

        wcm.start();
    }
}

void UTGaudiWcmTest::query_test_3_observers(const std::array<WcmPhysicalQueuesId, 3>&             physicalQueuesId,
                                            const std::deque<std::array<std::set<uint64_t>, 32>>& rQueryPermutations,
                                            const std::array<std::set<uint64_t>, 3>&              rCsHandles)
{
    WcmCsQuerierCheckerMock querier {rQueryPermutations};

    std::array<WcmObserverCheckerMock, 3> observers {rCsHandles[0], rCsHandles[1], rCsHandles[2]};

    {
        WorkCompletionManager      wcm(&querier);

        for (unsigned iter = 0; iter < rCsHandles.size(); iter++)
        {
            for (auto csHandle : rCsHandles[iter])
            {
                wcm.addCs(physicalQueuesId[iter], &observers[iter], csHandle);
            }
        }

        wcm.start();
    }
}

TEST_F(UTGaudiWcmTest, basic_test_trivial)
{
    const uint64_t         queryAmount = 0;
    WcmCsQuerierStatusMock querier(queryAmount, synSuccess);
    {
        WorkCompletionManager wcm(&querier);
        wcm.start();
        wcm.dump();
    }
}

TEST_F(UTGaudiWcmTest, basic_test_1_observer)
{
    WcmCsQuerierRecorderMock querier;
    const unsigned           notifyNum = 1;
    WcmObserverRecorderMock  observer(notifyNum);

    {
        WorkCompletionManager      wcm(&querier);
        const WcmPhysicalQueuesId  physicalQueuesId = 0x83;
        const uint64_t             csHandle         = 0x5678;
        addCs(wcm, physicalQueuesId, &observer, {csHandle});
        wcm.start();
        const hlthunk_wait_multi_cs_in inParams = querier.m_inParams.get_future().get();
        HB_ASSERT(1 == inParams.seq_len,
                  "{}: Failure inParams.seq_len expected 0x{:x} actual 0x{:x}",
                  __FUNCTION__,
                  1,
                  inParams.seq_len);
        const uint8_t notifyNumActual = observer.m_completedCsdcAmount.get_future().get();
        HB_ASSERT(notifyNum == notifyNumActual,
                  "{}: Failure completedCsdcAmount expected 0x{:x} actual 0x{:x}",
                  __FUNCTION__,
                  notifyNum,
                  notifyNumActual);
    }
}

TEST_F(UTGaudiWcmTest, basic_test_1_observer_several_queries)
{
    const uint8_t          batchSize = 32;
    const uint64_t         queryAmount = 0;
    WcmCsQuerierStatusMock querier(queryAmount, synSuccess);
    const uint8_t          completedCsdcAmountMax = 70;
    WcmObserverRecorderMock observer(completedCsdcAmountMax);

    {
        WorkCompletionManager      wcm(&querier);
        const WcmPhysicalQueuesId  physicalQueuesId = 0x83;
        const uint64_t             csHandle         = 0x5678;
        std::array<std::vector<uint64_t>, (completedCsdcAmountMax + batchSize - 1) / batchSize> csHandlesBatches;

        for (unsigned csHandleIter = 0; csHandleIter < completedCsdcAmountMax; csHandleIter++)
        {
            csHandlesBatches[csHandleIter / batchSize].push_back(csHandle + csHandleIter);
        }
        for (const auto& csHandles : csHandlesBatches)
        {
            addCs(wcm, physicalQueuesId, &observer, csHandles);
        }
        wcm.start();
        const uint8_t completedCsdcAmountActual = observer.m_completedCsdcAmount.get_future().get();
        HB_ASSERT(completedCsdcAmountMax == completedCsdcAmountActual,
                  "{}: Failure completedCsdcAmount expected 0x{:x} actual 0x{:x}",
                  __FUNCTION__,
                  completedCsdcAmountMax,
                  completedCsdcAmountActual);
    }
}

TEST_F(UTGaudiWcmTest, basic_test_3_observers_1_physical_queues_id)
{
    const std::array<WcmPhysicalQueuesId, 3>             physicalQueuesId {0x83, 0x83, 0x83};
    const std::deque<std::array<std::set<uint64_t>, 32>> queryPermutations {
        {{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {}, {}, {},
          {},  {},  {},  {},  {},  {},  {},  {},  {},  {},   {},   {},   {},   {}, {}, {}}}};

    const std::array<std::set<uint64_t>, 3> csHandles {{{1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}, {11, 12, 13}}};
    basic_test_3_observers(physicalQueuesId, queryPermutations, csHandles);
}

TEST_F(UTGaudiWcmTest, basic_test_3_observers_3_physical_queues_id)
{
    const std::array<WcmPhysicalQueuesId, 3>             physicalQueuesId {0x83, 0x84, 0x85};
    const std::deque<std::array<std::set<uint64_t>, 32>> queryPermutations {
        {{{1, 5, 11}, {1, 5, 11}, {1, 5, 11}, {2, 6, 12}, {2, 6, 12}, {2, 6, 12}, {3, 7, 13}, {3, 7, 13},
          {3, 7, 13}, {4, 8},     {4, 8},     {9},        {10},       {},         {},         {},
          {},         {},         {},         {},         {},         {},         {},         {},
          {},         {},         {},         {},         {},         {},         {},         {}}}};

    const std::array<std::set<uint64_t>, 3> csHandles {{{1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}, {11, 12, 13}}};
    basic_test_3_observers(physicalQueuesId, queryPermutations, csHandles);
}

TEST_F(UTGaudiWcmTest, query_test_3_observers_1_physical_queues_id)
{
    const std::array<WcmPhysicalQueuesId, 3> physicalQueuesId {0x83, 0x83, 0x83};

    const std::deque<std::array<std::set<uint64_t>, 32>> queryPermutations {
        {{{1},  {2},  {3},  {4},  {5},  {6},  {7},  {8},  {9},  {10}, {11}, {12}, {13}, {14}, {15}, {16},
          {17}, {18}, {19}, {20}, {21}, {22}, {23}, {24}, {25}, {26}, {27}, {28}, {29}, {30}, {31}, {32}}},
        {{{33}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
          {},   {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}};

    const std::array<std::set<uint64_t>, 3> csHandles {{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                                                        {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
                                                        {23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}}};

    query_test_3_observers(physicalQueuesId, queryPermutations, csHandles);
}

TEST_F(UTGaudiWcmTest, query_test_3_observers_3_physical_queues_id)
{
    const std::array<WcmPhysicalQueuesId, 3>             physicalQueuesId {0x83, 0x84, 0x85};
    const std::deque<std::array<std::set<uint64_t>, 32>> queryPermutations {
        {{{1, 12, 23}, {1, 12, 23}, {1, 12, 23}, {2, 13, 24},  {2, 13, 24},  {2, 13, 24},  {3, 14, 25},  {3, 14, 25},
          {3, 14, 25}, {4, 15, 26}, {4, 15, 26}, {4, 15, 26},  {5, 16, 27},  {5, 16, 27},  {5, 16, 27},  {6, 17, 28},
          {6, 17, 28}, {6, 17, 28}, {7, 18, 29}, {7, 18, 29},  {7, 18, 29},  {8, 19, 30},  {8, 19, 30},  {8, 19, 30},
          {9, 20, 31}, {9, 20, 31}, {9, 20, 31}, {10, 21, 32}, {10, 21, 32}, {10, 21, 32}, {11, 22, 33}, {11, 22, 33}}},
        {{{11, 22, 33}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
          {},           {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}};

    const std::array<std::set<uint64_t>, 3> csHandles {{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                                                        {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
                                                        {23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}}};
    query_test_3_observers(physicalQueuesId, queryPermutations, csHandles);
}

TEST_F(UTGaudiWcmTest, timeout_test)
{
    const uint64_t             csHandle    = 39;
    const uint64_t             queryAmount = 2;
    WcmCsQuerierIncompleteMock querier(csHandle, queryAmount);

    const std::deque<std::set<uint64_t>> completedCsHandles {{csHandle}};
    WcmObserverAdvanceCheckerMock        observer(completedCsHandles);

    {
        WorkCompletionManager      wcm(&querier);
        const WcmPhysicalQueuesId  physicalQueuesId = 0x83;
        wcm.addCs(physicalQueuesId, &observer, csHandle);
        wcm.start();
    }
}

TEST_F(UTGaudiWcmTest, incomplete_test)
{
    const uint64_t                       csHandle = 39;
    const uint64_t                       queryAmount = 2;
    WcmCsQuerierIncompleteMock           querier(csHandle, queryAmount);
    const std::deque<std::set<uint64_t>> completedCsHandles {
        {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 31},
        {32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48,
         49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63},
        {64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
         80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94},
        {39,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
         110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125},
        {126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
         142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157}};
    WcmObserverAdvanceCheckerMock observer(completedCsHandles);

    {
        WorkCompletionManager      wcm(&querier);
        const WcmPhysicalQueuesId  physicalQueuesId = 0x83;
        for (uint64_t iter = 0; iter <= 157; iter++)
        {
            wcm.addCs(physicalQueuesId, &observer, iter);
        }
        wcm.start();
    }
}

TEST_F(UTGaudiWcmTest, query_reset_test)
{
    const uint64_t                       csHandle = 0x5678;
    const uint64_t                       queryAmount = 1;
    WcmCsQuerierStatusMock               querier(queryAmount, synDeviceReset);
    const std::deque<std::set<uint64_t>> completedCsHandles {
        {csHandle}};  // Even though a reset is expected, the observer is ecpected to be notified
    WcmObserverAdvanceCheckerMock observer(completedCsHandles, true);

    {
        WorkCompletionManager      wcm(&querier);
        const WcmPhysicalQueuesId  physicalQueuesId = 0x83;
        wcm.addCs(physicalQueuesId, &observer, csHandle);
        wcm.start();
    }
}

TEST_F(UTGaudiWcmTest, query_fail_test)
{
    const uint64_t                       queryAmount = 1;
    WcmCsQuerierStatusMock               querier(queryAmount, synFail);
    const std::deque<std::set<uint64_t>> completedCsHandles {{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                              11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                              22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
                                                             {32}};
    WcmObserverAdvanceCheckerMock        observer(completedCsHandles);

    {
        WorkCompletionManager      wcm(&querier);
        const WcmPhysicalQueuesId  physicalQueuesId = 0x83;

        for (uint64_t iter = 0; iter <= 32; iter++)
        {
            wcm.addCs(physicalQueuesId, &observer, iter);
        }

        wcm.start();
    }
}

TEST_F(UTGaudiWcmTest, refresh_incomplete_during_work_test)
{
    std::array<std::promise<hlthunk_wait_multi_cs_out>, 2> queryPausersPromise;
    std::array<std::future<hlthunk_wait_multi_cs_out>, 2>  queryPausersFuture {queryPausersPromise[0].get_future(),
                                                                              queryPausersPromise[1].get_future()};
    WcmCsQuerierPause2Mock                                 querier(queryPausersFuture);
    // 33 + 1 out of them 32 (0-31) in the first query and 1+1=2 (32-33) in the second query where
    // 33 is added in between the two consecutive queries.
    const std::deque<std::set<uint64_t>> completedCsHandles {{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                              11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                              22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 31},
                                                             {32, 33}};
    WcmObserverAdvanceCheckerMock        observer(completedCsHandles);

    {
        WorkCompletionManager      wcm(&querier);
        const WcmPhysicalQueuesId  physicalQueuesId = 0x83;

        for (uint64_t iter = 0; iter <= 32; iter++)
        {
            wcm.addCs(physicalQueuesId, &observer, iter);
        }

        wcm.start();

        // Wait for 32 query
        const hlthunk_wait_multi_cs_in inParams1 = querier.m_inParams[0].get_future().get();

        HB_ASSERT(inParams1.seq_len == 32,
                  "{}: Failure seq_len expected {} actual {}",
                  __FUNCTION__,
                  32,
                  inParams1.seq_len);

        // Add 33
        wcm.addCs(physicalQueuesId, &observer, 33);

        // Release for 1 query
        queryPausersPromise[0].set_value({0, 0xFFFFFFFF, 32});

        // Wait for 2 query
        const hlthunk_wait_multi_cs_in inParams2 = querier.m_inParams[1].get_future().get();

        HB_ASSERT(inParams2.seq_len == 2,
                  "{}: Failure seq_len expected {} actual {}",
                  __FUNCTION__,
                  2,
                  inParams2.seq_len);

        // Release for 2 query
        queryPausersPromise[1].set_value({0, 0x00000003, 2});
    }
}