#include "recipe_generator.h"
#include "command_queue.h"
#include "habana_global_conf.h"
#include "gaudi2_arc_host_packets.h"  // in order to get ARC_FW_INIT_CONFIG_VER for version consistency check
#include "gaudi2_eng_arc_hooks.h"
#include "queue_command.h"

namespace gaudi2
{
Gaudi2RecipeGenerator::Gaudi2RecipeGenerator(const HabanaGraph* g) : RecipeGenerator(g)
{
    m_ecbContainer.setEngArcHooks(std::make_shared<EngArcHooks>(gaudi2::EngArcHooks::instance()));
}

std::string Gaudi2RecipeGenerator::getEngineStr(unsigned id) const
{
    return getEngineName(static_cast<gaudi2_queue_id>(id));
}

void Gaudi2RecipeGenerator::validateQueue(ConstCommandQueuePtr queue, bool isSetup) const
{
    if (isSetup)
    {
        HB_ASSERT(queue->getCommands(true).size() == 0, "We expect the Activate part to be empty in Gaudi2");
    }
}

bool Gaudi2RecipeGenerator::isSFGInitCommand(QueueCommand* cmd) const
{
    return (dynamic_cast<SFGInitCmd*>(cmd) != nullptr);
}

unsigned Gaudi2RecipeGenerator::getInitSfgValue(QueueCommand* cmd) const
{
    SFGInitCmd* sfgInitCmd = dynamic_cast<SFGInitCmd*>(cmd);

    if (sfgInitCmd)
    {
        return sfgInitCmd->getSfgSyncObjValue();
    }

    return 0;
}

uint32_t Gaudi2RecipeGenerator::getGaudi2VersionMinor()
{
    return ARC_FW_INIT_CONFIG_VER;  // for GC-RT-SCAL-FW version consistency check
}

void Gaudi2RecipeGenerator::setBlobFlags(RecipeBlob* blob, QueueCommand* cmd) const
{
    RecipeGenerator::setBlobFlags(blob, cmd);
    if (cmd->isSwitchCQ()) blob->setContainsSwtc();

    if (!cmd->isDynamic()) return;  // currently only dynamic commands require meta-data updates

    SFGCmd*      sfgCmd       = nullptr;
    ResetSobs*   resetSobsCmd = nullptr;
    bool         update       = false;
    BlobMetaData md;

    // get existing meta-data flags if any
    if (blob->getBlobMetaData().is_set())
    {
        md = blob->getBlobMetaData().value();
    }

    // see if we need to update the meta-data flags
    if ((sfgCmd = dynamic_cast<SFGCmd*>(cmd)) != nullptr)
    {
        md.sfgSobInc = sfgCmd->getSfgSyncObjValue();
        update       = true;
    }
    else if ((resetSobsCmd = dynamic_cast<ResetSobs*>(cmd)) != nullptr)
    {
        HB_ASSERT(resetSobsCmd->getTarget() && !resetSobsCmd->getTargetXps(), "target, and only target, must be set");
        md.resetSob = {resetSobsCmd->getTarget(), 0, resetSobsCmd->getTotalNumEngs()};
        update      = true;
    }

    if (update) blob->getBlobMetaData().set(md);
}

uint32_t Gaudi2RecipeGenerator::getVersionMinor() const
{
    return getGaudi2VersionMinor();
}

}  // namespace gaudi2