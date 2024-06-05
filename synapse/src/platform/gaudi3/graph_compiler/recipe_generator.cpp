#include "recipe_generator.h"
#include "command_queue.h"
#include "gaudi3/gaudi3_arc_host_packets.h"
#include "gaudi3_eng_arc_hooks.h"
#include "queue_command.h"

namespace gaudi3
{
Gaudi3RecipeGenerator::Gaudi3RecipeGenerator(const HabanaGraph* g) : RecipeGenerator(g)
{
    m_ecbContainer.setEngArcHooks(std::make_shared<EngArcHooks>(gaudi3::EngArcHooks::instance()));
    m_ecbContainer.setEngArcHooksForCme(std::make_shared<EngArcHooksForCme>(gaudi3::EngArcHooksForCme::instance()));
}

std::string Gaudi3RecipeGenerator::getEngineStr(unsigned id) const
{
    return getEngineName(static_cast<gaudi3_engine_id>(id));
}

void Gaudi3RecipeGenerator::validateQueue(ConstCommandQueuePtr queue, bool isSetup) const
{
    if (isSetup)
    {
        HB_ASSERT(queue->getCommands(true).size() == 0, "We expect the Activate part to be empty in Gaudi3");
    }
}

bool Gaudi3RecipeGenerator::shouldCreateECBs() const
{
    return true;
}

void Gaudi3RecipeGenerator::setBlobFlags(RecipeBlob* blob, QueueCommand* cmd) const
{
    RecipeGenerator::setBlobFlags(blob, cmd);
    if (cmd->isSwitchCQ()) blob->setContainsSwtc();

    if (!cmd->isDynamic()) return;  // currently only dynamic commands require meta-data updates

    ResetSobs*    resetSobsCmd    = nullptr;
    McidRollover* mcidRolloverCmd = nullptr;
    ArcExeWdTpc*  exeWdTpcCmd     = nullptr;
    bool          update          = false;
    BlobMetaData  md;

    // get existing meta-data flags if any
    if (blob->getBlobMetaData().is_set())
    {
        md = blob->getBlobMetaData().value();
    }

    // see if we need to update the meta-data flags
    if ((exeWdTpcCmd = dynamic_cast<ArcExeWdTpc*>(cmd)) != nullptr)
    {
        md.numDcoresSplit = exeWdTpcCmd->getNumCtxs();
        update = true;
    }
    else if ((resetSobsCmd = dynamic_cast<ResetSobs*>(cmd)) != nullptr)
    {
        if (resetSobsCmd->getTarget() || resetSobsCmd->getTargetXps())
        {
            md.resetSob = {resetSobsCmd->getTarget(), resetSobsCmd->getTargetXps(), resetSobsCmd->getTotalNumEngs()};
        }
        else
        {
            md.resetSob.isCanceled = true; // we joined the gemm and xpose resets and canceled this command
        }
        update = true; // in any case we want to update, even if we have a canceled resetSob
    }
    else if ((mcidRolloverCmd = dynamic_cast<McidRollover*>(cmd)) != nullptr)
    {
        if (mcidRolloverCmd->getTarget() || mcidRolloverCmd->getTargetXps())
        {
            md.rollovers.push_back(
                BlobMetaData::Rollover {mcidRolloverCmd->getTarget(), mcidRolloverCmd->getTargetXps(), false});
        }
        else
        {
            // isCanceled = true --> we joined the gemm and xpose rollover and canceled this command
            md.rollovers.push_back(BlobMetaData::Rollover {0, 0, true});
        }
        update = true;  // in any case we want to update, even if we have a canceled mcidRollover
    }

    if (update) blob->getBlobMetaData().set(md);
}

uint32_t Gaudi3RecipeGenerator::getGaudi3VersionMinor()
{
    return ARC_FW_INIT_CONFIG_VER;  // for GC-RT-SCAL-FW version consistency check
}

uint32_t Gaudi3RecipeGenerator::getVersionMinor() const
{
    return getGaudi3VersionMinor();
}

bool Gaudi3RecipeGenerator::isMMEDmaNode(const NodePtr& n) const
{
    return MmeNode::isDmaOperation(n);
}



}  // namespace gaudi3