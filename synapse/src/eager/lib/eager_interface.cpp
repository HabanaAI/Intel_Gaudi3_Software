#include "include/eager/eager_interface.h"

// eager includes (relative to src/eager/lib/)
#include "eager_graph.h"
#include "recipe_gen/recipe_templates.h"

namespace eager_mode
{
// Generate eager templates. Normally called once at synSingleton::initSingleton().
// If note done, would generate the tempaltes on first use which might be on the hot path.
void createEagerTemplates()
{
    eager_mode::RecipeTemplates::getInstance().createAllTemplates();
}

// Check whether an EagerGraph can be created for a given device type
bool isValidForEager(synDeviceType deviceType)
{
    return EagerGraph::isValidForEager(deviceType);
}

// Create an EagerGraph for the given device type
HabanaGraph* createEagerGraph(synDeviceType deviceType)
{
    return new EagerGraph(deviceType);
}

const EagerMmeBrainBase& getEagerMmeBrain(const HabanaGraph& eagerGraph)
{
    return static_cast<const eager_mode::EagerGraph&>(eagerGraph).getEagerMmeBrain();
}

}  // namespace eager_mode
