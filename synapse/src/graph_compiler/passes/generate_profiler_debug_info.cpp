#include "generate_profiler_debug_info.h"

#include "habana_graph.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>

// H9-5339 WA - TPC will not see the 2 MSB bits, so recipe-id indicator moved from bit 15 to bit 13,
// and for backward compatibility the recipe-id will set the 3 MSBs as on
constexpr uint16_t USED_BITS_MASK = 0x1FFF;
// 0xFFFF is a saved context-id for FW null descriptors.
// 0xE000 is a saved context-id for FW barrier.
constexpr uint16_t RECIPE_ID_MIN_VAL = 0xE001;
constexpr uint16_t RECIPE_ID_MAX_VAL = 0xFFFE;

namespace
{
// Note that this approach is used instad of a thread_local random to avoid collisions
// even in extreme stress test situations like the one in
// GaudiRecipeTest.debug_profilier_context_id_compile_parallel_test
class HabanaRecipeDebugIDGenerator final
{
public:
    ~HabanaRecipeDebugIDGenerator() = default;

    static HabanaRecipeDebugIDGenerator& instance()
    {
        static auto singleton = std::unique_ptr<HabanaRecipeDebugIDGenerator>(new HabanaRecipeDebugIDGenerator());
        return *singleton;
    }

    [[nodiscard]] uint16_t getID() { return m_ids[m_index++ % m_ids.size()]; }

    HabanaRecipeDebugIDGenerator(const HabanaRecipeDebugIDGenerator&) = delete;
    HabanaRecipeDebugIDGenerator(HabanaRecipeDebugIDGenerator&&)      = delete;
    HabanaRecipeDebugIDGenerator& operator=(const HabanaRecipeDebugIDGenerator&) = delete;
    HabanaRecipeDebugIDGenerator& operator=(HabanaRecipeDebugIDGenerator&&) = delete;

private:
    HabanaRecipeDebugIDGenerator();

    std::array<uint16_t, (RECIPE_ID_MAX_VAL - RECIPE_ID_MIN_VAL + 1)> m_ids {};
    std::atomic<unsigned>                                             m_index {};
};
}  // anonymous namespace

HabanaRecipeDebugIDGenerator::HabanaRecipeDebugIDGenerator()
{
    // fill with [RECIPE_ID_MIN_VAL..RECIPE_ID_MAX_VAL] and shuffle them
    std::iota(m_ids.begin(), m_ids.end(), RECIPE_ID_MIN_VAL);
    std::shuffle(m_ids.begin(), m_ids.end(), std::mt19937(std::random_device()()));
}

void initializeProfilerDebugInfo()
{
    (void)HabanaRecipeDebugIDGenerator::instance();
}

/*
 * Each execution engine type maintains its own execution order based on the final execution order.
 * The context id field in the first node in each execution engine type contains the graph id.
 */
bool generateProfilerDebugInfo(HabanaGraph& g)
{
    if (!GCFG_ENABLE_PROFILER.value()) return true;

    std::array<uint32_t, LAST_HABANA_DEVICE> deviceFullContextId = {};
    std::array<bool, LAST_HABANA_DEVICE>     deviceSeenAlready   = {};

    // 16bits random number with MSB = 1
    uint16_t recipeDebugID = HabanaRecipeDebugIDGenerator::instance().getID();
    g.setRecipeDebugId(recipeDebugID);

    for (const NodePtr& n : g.getExeSortedNodes())
    {
        if (n->isLogicalOperation()) continue;

        HabanaDeviceType deviceType = g.getNodeDebugDeviceType(n);
        HB_ASSERT(deviceType < LAST_HABANA_DEVICE, "deviceType out of range - {}", deviceType);

        uint32_t fullContextId = deviceFullContextId[deviceType];
        n->setFullContextId(fullContextId);
        uint16_t contextId = fullContextId & USED_BITS_MASK;  // context id is 13bit with MSB = 0
        // The first node executed in each of the engines contains context-id with MSB=1 and value of recipe-id.
        n->setContextId(deviceSeenAlready[deviceType] ? contextId : recipeDebugID);
        deviceSeenAlready[deviceType] = true;
        deviceFullContextId[deviceType]++;
    }
    return true;
}
