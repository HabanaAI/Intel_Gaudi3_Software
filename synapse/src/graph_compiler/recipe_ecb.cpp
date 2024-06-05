#include "recipe_ecb.h"
#include "recipe_allocator.h"
#include "recipe.h"
#include "eng_arc_hooks.h"

thread_local std::shared_ptr<NopEngArcCommand> RecipeEcb::s_pFullChunkNop = nullptr;

static Recipe::EngineType type2logical(HabanaDeviceType type)
{
    switch (type)
    {
        case DEVICE_TPC:
            return Recipe::EngineType::TPC;
        case DEVICE_EDMA:
            return Recipe::EngineType::DMA;
        case DEVICE_MME:
            return Recipe::EngineType::MME;
        case DEVICE_ROTATOR:
            return Recipe::EngineType::ROT;
        case DEVICE_CME:
            return Recipe::EngineType::CME;
        default:
            break;
    }
    HB_ASSERT(0, "Don't know to convert HabanaDeviceType to recipe logical engine id");
    return Recipe::EngineType::INVALID;
}

RecipeEcb::RecipeEcb(HabanaDeviceType devType, std::shared_ptr<EngArcHooks> engArcHooks)
: m_deviceType(devType),
  m_dccmMngr(0,  // zero base address
             engArcHooks->getWdCtxtCount(),
             devType == DEVICE_TPC    ? engArcHooks->getTpcWdCtxtSize()
             : devType == DEVICE_MME  ? engArcHooks->getMmeWdCtxtSize()
             : devType == DEVICE_EDMA ? engArcHooks->getEdmaWdCtxtSize()
                                      : engArcHooks->getRotWdCtxtSize()),
  m_suspensionLength(engArcHooks->getWdCtxtCount()),
  m_engArcHooks(engArcHooks),
  m_staticEcbChunkSize(engArcHooks->getStaticEcbChunkSize()),
  m_dynamicEcbChunkSize(engArcHooks->getDynamicEcbChunkSize())
{
    // First, push a ListSize command to the static and dynamic ECBs
    m_pStaticSizeCmd  = m_engArcHooks->getListSizeEngArcCommand();
    m_pDynamicSizeCmd = m_engArcHooks->getListSizeEngArcCommand();

    m_pStaticSizeCmd->setTopologyStart();
    m_pDynamicSizeCmd->setTopologyStart();

    m_staticCmdsSize  = m_pStaticSizeCmd->sizeInBytes();
    m_dynamicCmdsSize = m_pDynamicSizeCmd->sizeInBytes();

    m_staticCmds.emplace_back(m_pStaticSizeCmd);
    m_dynamicCmds.emplace_back(m_pDynamicSizeCmd);

    // Prepare full chunk padding nop command for static ECB round-up
    if (s_pFullChunkNop == nullptr)
    {
        s_pFullChunkNop = m_engArcHooks->getNopEngArcCommand();
        s_pFullChunkNop->setPadding((m_staticEcbChunkSize - s_pFullChunkNop->sizeInBytes()) / DWORD_SIZE);
    }
}

void RecipeEcb::initDynamicQueue(const RecipeProgram& program)
{
    if (program.isInitSfg())
    {
        // create signal out command - we should not switch here
        EngArcCmdPtr pSfgCmd = m_engArcHooks->getSignalOutEngArcCommand(program.getInitSfgValue(), false, false);
        addCmdToQueue(m_dynamicCmds, m_dynamicCmdsSize, pSfgCmd, m_dynamicEcbChunkSize);
    }

    // We may have multiple rollover Ids
    for (auto rolloverId : program.getPreNodesRolloverIds())
    {
        // create rolloverId command - we should not switch here
        EngArcCmdPtr pRolloverCmd = m_engArcHooks->getMcidRolloverArcCommand(0, 0, false, false);
        addCmdToQueue(m_dynamicCmds, m_dynamicCmdsSize, pRolloverCmd, m_dynamicEcbChunkSize);
    }

    // Arcs always start a workload from the dynamic CQ. GC always has some static configuration before the first
    // activation; thus, we add a Nop with SwitchCQ right at the beginning of the workload (switch! not yield)
    addCmdToQueue(m_dynamicCmds, m_dynamicCmdsSize, m_engArcHooks->getNopEngArcCommand(true), m_dynamicEcbChunkSize);
}

void RecipeEcb::generateCmds(unsigned                   engineId,
                             const RecipeProgram&       program,
                             const RecipeBlobContainer& blobContainer,
                             bool                       needToCreateDynamicCmds)
{
    if (needToCreateDynamicCmds) initDynamicQueue(program);

    const std::vector<uint64_t>& blobIndicies = program.blobIndicies();
    const std::vector<Settable<BlobMetaData>>& blobsMetaData = program.getBlobsMetaData();

    // Generate the commands
    for (auto i = 0; i < blobIndicies.size(); i++)
    {
        uint64_t          blobIndex       = blobIndicies[i];
        const RecipeBlob* pBlob           = blobContainer.getBlobByIndex(blobIndex);
        bool              currentBlobIsWd = pBlob->isWorkDistBlob();
        bool              nextBlobIsWd    = false;

        // Check if the next blob is work-distribution blob (i.e. dynamic blob)
        if ((i + 1) < blobIndicies.size())
        {
            nextBlobIsWd = blobContainer.getBlobByIndex(blobIndicies[i + 1])->isWorkDistBlob();
        }

        if (currentBlobIsWd)
        {
            if (needToCreateDynamicCmds)
            {
                addCmdsForWorkDistBlob(pBlob, blobsMetaData[i], true /* always yield naively from dynamic to static */);
            }
        }
        else  // current blob is static
        {
            addCmdsForStaticBlob(engineId, pBlob, nextBlobIsWd);
        }
    }
}

EngArcCmdPtr RecipeEcb::popSuspendedExe()
{
    EngArcCmdPtr ret = m_suspendedExeCmds.front();
    m_suspendedExeCmds.pop();
    return ret;
}

void RecipeEcb::finalize()
{
    while (!m_suspendedExeCmds.empty())  // spill the suspended commands into the queue
    {
        m_suspendedExeCmds.front()->setYield(true);  // always yield naively from dynamic to static
        addCmdToQueue(m_dynamicCmds, m_dynamicCmdsSize, popSuspendedExe(), m_dynamicEcbChunkSize);
    }
    m_suspendedWdCmdCount = 0;

    // We need to ensure the workload ends while the Arc is on the dynamic CQ fetcher. GC always finishes the
    // topology while the static CQ is active; thus, we add a Nop with SwitchCQ at the end of the static ECB.
    addCmdToQueue(m_staticCmds, m_staticCmdsSize, m_engArcHooks->getNopEngArcCommand(true), m_staticEcbChunkSize);

    // Make sure each ECB ends on chunk boundary by adding padding NOP as needed
    addPaddingNop(m_staticCmds, m_staticCmdsSize, m_staticEcbChunkSize, m_staticEcbChunkSize);
    addPaddingNop(m_dynamicCmds, m_dynamicCmdsSize, m_dynamicEcbChunkSize, m_dynamicEcbChunkSize);

    // Update the ListSize commands with the final sizes
    m_pStaticSizeCmd->setListSize(m_staticCmdsSize);
    m_pDynamicSizeCmd->setListSize(m_dynamicCmdsSize);
}

void RecipeEcb::roundUpStaticSize(uint64_t maxStaticSizeBytes)
{
    HB_ASSERT((maxStaticSizeBytes % m_staticEcbChunkSize == 0) &&
              (m_staticCmdsSize   % m_staticEcbChunkSize == 0) &&
              (maxStaticSizeBytes >= m_staticCmdsSize),
              "unexpected maxStaticSizeBytes");

    // Round-up the static ECB size with full chunks of nop until we reach maxStaticSizeBytes
    while (maxStaticSizeBytes - m_staticCmdsSize > 0)
    {
        m_staticCmds.emplace_back(s_pFullChunkNop);
        m_staticCmdsSize += s_pFullChunkNop->sizeInBytes();  // nop command size includes its padding
    }
    m_pStaticSizeCmd->setListSize(m_staticCmdsSize);  // update the static ListSize command with the final round-up size
}

void RecipeEcb::addCmdsForWorkDistBlob(const RecipeBlob* pBlob, const Settable<BlobMetaData>& blobMD, bool yield)
{
    unsigned dccmOffset = m_dccmMngr.getSlotAllocation();
    unsigned numDcoreSplit = blobMD.is_set() ? (blobMD.value().numDcoresSplit) : 1;
    // create schedule dma command
    EngArcCmdPtr pCmdDma =
        m_engArcHooks->getScheduleDmaEngArcCommand(pBlob->getSerializedOffset(),
                                                   EngArcBufferAddrBase::DYNAMIC_ADDR_BASE,
                                                   dccmOffset,
                                                   pBlob->sizeInBytes() / numDcoreSplit,
                                                   (numDcoreSplit > 1),
                                                   yield);

    addCmdToQueue(m_dynamicCmds, m_dynamicCmdsSize, pCmdDma, m_dynamicEcbChunkSize);

    // create work distribute execution command and suspend it
    m_suspendedExeCmds.push(m_engArcHooks->getDynamicWorkDistEngArcCommand(m_dccmMngr.addrToSlot(dccmOffset), false));
    m_suspendedWdCmdCount++;

    // add additional ECB commands per blob meta-data flags
    if (blobMD.is_set())
    {
        bool rolloverCanceled     = false;
        bool immediateCmdWasAdded = false;

        // Check if need to insert SFG immediate-command
        if (blobMD.value().sfgSobInc > 0)
        {
            // create SFG command and suspend it
            m_suspendedExeCmds.push(m_engArcHooks->getSignalOutEngArcCommand(blobMD.value().sfgSobInc, false, false));
            immediateCmdWasAdded = true;
        }

        // Check if need to insert rollover immediate-command
        for (auto rolloverId : blobMD.value().rollovers)
        {
            if (rolloverId.target || rolloverId.targetXps)
            {
                // create reset rollover command and suspend it
                unsigned target    = rolloverId.target;
                unsigned targetXps = rolloverId.targetXps;
                m_suspendedExeCmds.push(m_engArcHooks->getMcidRolloverArcCommand(target, targetXps, false, false));
                immediateCmdWasAdded = true;
            }
            else if (rolloverId.isCanceled)
            {
                rolloverCanceled = true;
            }
        }

        // Check if need to insert ResetSob immediate-command
        if (blobMD.value().resetSob.target || blobMD.value().resetSob.targetXps)
        {
            // create reset SOBs command and suspend it
            unsigned target    = blobMD.value().resetSob.target;
            unsigned targetXps = blobMD.value().resetSob.targetXps;
            unsigned numEngs   = blobMD.value().resetSob.totalNumEngs;
            m_suspendedExeCmds.push(m_engArcHooks->getResetSobsArcCommand(target, targetXps, numEngs, false, false));
            immediateCmdWasAdded = true;
        }

        if (!immediateCmdWasAdded && (blobMD.value().resetSob.isCanceled || rolloverCanceled))
        {
            // Add ECB NOP if a reset sob or rollover command was canceled so we will have someone to carry the switch
            // bit
            m_suspendedExeCmds.push(m_engArcHooks->getNopEngArcCommand());
            immediateCmdWasAdded = true;
        }

        // Set the switch_cq bit on the last command. This is required only for ECB commands that generated from the
        // meta-data flags since they are immediate commands. The work-dist specifies the switch_cq via the context.
        if (immediateCmdWasAdded)
        {
            m_suspendedExeCmds.back()->setSwitchCQ(pBlob->isContainingSwtc());
        }
    }

    if (m_suspendedWdCmdCount == m_suspensionLength)
    {
        addCmdToQueue(m_dynamicCmds, m_dynamicCmdsSize, popSuspendedExe(), m_dynamicEcbChunkSize);
        m_suspendedWdCmdCount--;

        // The first command is always WorkDist. We will call addCmdToQueue repeatedly till the next WorkDist command
        while (m_suspendedExeCmds.size() && !m_suspendedExeCmds.front()->isArcExeWd())
        {
            addCmdToQueue(m_dynamicCmds, m_dynamicCmdsSize, popSuspendedExe(), m_dynamicEcbChunkSize);
        }
    }
}

void RecipeEcb::addCmdsForStaticBlob(unsigned engineId, const RecipeBlob* pBlob, bool yield)
{
    unsigned cpuId;
    bool     isStaticBlobCommon = (m_deviceType == DEVICE_TPC); // Only TPC supports work distribution

    if (isStaticBlobCommon)
    {
        cpuId = m_engArcHooks->getCpuIdAll();
    }
    else
    {
        unsigned stream      = 0xFFFFFFFF;
        unsigned engineIndex = 0xFFFFFFFF;
        bool     status      = m_engArcHooks->getQueueIdInfo(m_deviceType, engineIndex, stream, engineId);
        HB_ASSERT(status, "failed to get queue info for queue id {}", engineId);
        cpuId = m_engArcHooks->engIdx2cpuId(engineIndex, m_deviceType);
    }

    // create static CP dma command with potential yield
    EngArcCmdPtr pCmd = m_engArcHooks->getStaticCpDmaEngArcCommand(
        pBlob->getSerializedOffset(),
        pBlob->isPatchingBlob() ? EngArcBufferAddrBase::PATCHING_ADDR_BASE : EngArcBufferAddrBase::EXECUTE_ADDR_BASE,
        pBlob->sizeInBytes(),
        yield,
        cpuId);

    addCmdToQueue(m_staticCmds, m_staticCmdsSize, pCmd, m_staticEcbChunkSize);
}

void RecipeEcb::addCmdToQueue(std::vector<EngArcCmdPtr>& cmdsVec,
                              uint64_t&                  cmdsBinSize,
                              EngArcCmdPtr               cmd,
                              unsigned                   chunkSize) const
{
    // make sure the new cmd is not crossing ECB chunk boundary by adding NOP as needed, push to queue and update size
    addPaddingNop(cmdsVec, cmdsBinSize, cmd->sizeInBytes(), chunkSize);
    cmdsVec.emplace_back(cmd);
    cmdsBinSize += cmd->sizeInBytes();
}

void RecipeEcb::addPaddingNop(std::vector<EngArcCmdPtr>& cmdsVec,
                              uint64_t&                  cmdsBinSize,
                              unsigned                   nextCmdBinSize,
                              unsigned                   chunkSize) const
{
    HB_ASSERT(nextCmdBinSize <= chunkSize, "engine ARC command cannot be larger than ECB chunk size");

    unsigned freeBytesInChunk = chunkSize - (cmdsBinSize % chunkSize);

    if (nextCmdBinSize > freeBytesInChunk)
    {
        // fill the command vector with nop that will consume all the space until the next chunk boundary
        std::shared_ptr<NopEngArcCommand> pNop           = m_engArcHooks->getNopEngArcCommand();
        unsigned                          numDWordsToAdd = freeBytesInChunk / DWORD_SIZE;

        pNop->setPadding(numDWordsToAdd - (pNop->sizeInBytes() / DWORD_SIZE));  // initial padding size is 0
        cmdsVec.emplace_back(pNop);
        cmdsBinSize += pNop->sizeInBytes();  // nop command size includes now its padding
    }
}

void RecipeEcb::serializeCmds(uint8_t* pFiller, bool staticCmds) const
{
    const std::vector<EngArcCmdPtr>& cmds = staticCmds ? m_staticCmds : m_dynamicCmds;
    for (auto cmd : cmds)
    {
        cmd->serialize(pFiller);
        pFiller += cmd->sizeInBytes();
    }
}

void RecipeEcb::printCmds(bool staticCmds) const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(RECIPE_GEN) && !LOG_LEVEL_AT_LEAST_DEBUG(GC_ARC)) return;
    const std::vector<EngArcCmdPtr>& cmds = staticCmds ? m_staticCmds : m_dynamicCmds;
    for (auto cmd : cmds)
    {
        cmd->print();
    }
}

//-----------------------------------------------------------------------------
//         CmeRecipeEcb
//-----------------------------------------------------------------------------

CmeRecipeEcb::CmeRecipeEcb(std::shared_ptr<EngArcHooks> engArcHooks)
{
    m_deviceType          = DEVICE_CME;
    m_engArcHooks         = engArcHooks;
    m_dynamicEcbChunkSize = engArcHooks->getDynamicEcbChunkSize();
}

void CmeRecipeEcb::registerCmds(const std::list<EngArcCmdPtr>& cmeCmds)
{
    for (EngArcCmdPtr cmd : cmeCmds)
    {
        addCmdToQueue(m_dynamicCmds, m_dynamicCmdsSize, cmd, m_dynamicEcbChunkSize);
    }
    // Make sure ECB ends on chunk boundary by adding padding NOP as needed
    addPaddingNop(m_dynamicCmds, m_dynamicCmdsSize, m_dynamicEcbChunkSize, m_dynamicEcbChunkSize);
}

//-----------------------------------------------------------------------------
//         RecipeEcbContainer
//-----------------------------------------------------------------------------

void RecipeEcbContainer::generateECBs(const RecipeProgramContainer& programContainer,
                                      const RecipeBlobContainer&    blobContainer)
{
    for (unsigned i = 0; i < programContainer.getNumPrograms(); i++)
    {
        const RecipeProgram& program = programContainer.getProgramByIndex(i);
        HabanaDeviceType     device  = program.getDeviceType();

        HB_ASSERT(device == DEVICE_TPC || device == DEVICE_EDMA || device == DEVICE_MME || device == DEVICE_ROTATOR,
                  "Unexpected device type");

        bool needToCreateDynamicCmds = m_ecbs[device].empty();  // generate dynamic commands only for the first ECB
        m_ecbs[device].emplace_back(std::make_shared<RecipeEcb>(device, m_engArcHooks));
        m_ecbs[device].back()->generateCmds(program.getEngineId(), program, blobContainer, needToCreateDynamicCmds);
    }

    for (auto& ecbVec : m_ecbs)
    {
        uint64_t maxStaticSize = 0;
        for (auto& ecb : ecbVec.second)
        {
            ecb->finalize();
            maxStaticSize = std::max(maxStaticSize, ecb->getStaticCmdsSize());
        }
        for (auto& ecb : ecbVec.second)
        {
            ecb->roundUpStaticSize(maxStaticSize);
        }
    }
}

void RecipeEcbContainer::registerCmeCommands(const std::list<EngArcCmdPtr>& cmeCmds)
{
    if (cmeCmds.empty() || GCFG_DISABLE_ALL_CME_COMMANDS.value()) return;
    std::shared_ptr<CmeRecipeEcb> cmeEcb = std::make_shared<CmeRecipeEcb>(m_engArcHooksForCme);
    cmeEcb->registerCmds(cmeCmds);
    m_ecbs[DEVICE_CME].push_back(cmeEcb);
}

void RecipeEcbContainer::serialize(uint32_t* pNumArcJobs, arc_job_t** ppArcJobs, RecipeAllocator* pRecipeAlloc) const
{
    *pNumArcJobs       = m_ecbs.size();
    *ppArcJobs         = (arc_job_t*)pRecipeAlloc->allocate(*pNumArcJobs * sizeof(arc_job_t));
    arc_job_t* pFiller = *ppArcJobs;

    for (const auto& ecbVec : m_ecbs)
    {
        pFiller->logical_engine_id = type2logical(ecbVec.first);
        pFiller->engines_filter    = 0;  // currently not used

        // Serialize static ECB
        struct ecb_t& staticEcb  = pFiller->static_ecb;  // shortcut
        unsigned staticCmdsSize0 = ecbVec.second[0]->getStaticCmdsSize();  // all static ECBs have the same size
        staticEcb.cmds_size      = ecbVec.second.size() * staticCmdsSize0;
        staticEcb.cmds           = (uint8_t*)pRecipeAlloc->allocate(staticEcb.cmds_size, true);
        uint8_t* currPtr         = staticEcb.cmds;
        for (const auto& ecb : ecbVec.second)
        {
            ecb->serializeCmds(currPtr, true);
            currPtr += staticCmdsSize0;
        }
        // If the device is operating in work-distribution mode (like TPC) and thus has only 1 ECB, then there is no
        // "private" static ECB for individual engines and so the cmds_eng_offset is 0; otherwise the size is taken
        // from the first ECB since all static ECBs have the same size due to the roundUpStaticSize we did earlier.
        staticEcb.cmds_eng_offset = ecbVec.second.size() == 1 ? 0 : staticCmdsSize0;

        // Serialize dynamic ECB
        struct ecb_t& dynamicEcb   = pFiller->dynamic_ecb;  // shortcut
        dynamicEcb.cmds_eng_offset = 0;
        dynamicEcb.cmds_size       = ecbVec.second[0]->getDynamicCmdsSize();
        dynamicEcb.cmds            = (uint8_t*)pRecipeAlloc->allocate(dynamicEcb.cmds_size, true);
        ecbVec.second[0]->serializeCmds(dynamicEcb.cmds, false);  // only the first ECB has dynamic commands

        // Move to next device's ECB
        pFiller++;
    }
}

void RecipeEcbContainer::print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(RECIPE_GEN) && !LOG_LEVEL_AT_LEAST_DEBUG(GC_ARC)) return;
    LOG_DEBUG(RECIPE_GEN, "    Number of Arc jobs = {}", m_ecbs.size());
    for (const auto& ecbVec : m_ecbs)
    {
        Recipe::EngineType logicalId = type2logical(ecbVec.first);
        std::string        logicalIdStr("???");

        if (logicalId == Recipe::EngineType::TPC) logicalIdStr = std::string("TPC");
        if (logicalId == Recipe::EngineType::MME) logicalIdStr = std::string("MME");
        if (logicalId == Recipe::EngineType::DMA) logicalIdStr = std::string("DMA");
        if (logicalId == Recipe::EngineType::ROT) logicalIdStr = std::string("ROT");
        if (logicalId == Recipe::EngineType::CME) logicalIdStr = std::string("CME");

        uint64_t dynamicCmdsSize = ecbVec.second[0]->getDynamicCmdsSize();  // only the first ECB has dynamic commands
        uint64_t numDynamicCmds  = ecbVec.second[0]->getNumDynamicCmds();   // only the first ECB has dynamic commands
        uint64_t staticCmdsSize  = ecbVec.second.size() * ecbVec.second[0]->getStaticCmdsSize();
        uint64_t numEngEcbs      = ecbVec.second.size() == 1 ? 0 : ecbVec.second.size();
        uint64_t engEcbSize      = numEngEcbs == 0 ? 0 : ecbVec.second[0]->getStaticCmdsSize();  // all have same size
        uint64_t numStaticCmds   = 0;
        std::for_each(ecbVec.second.begin(), ecbVec.second.end(), [&numStaticCmds](const RecipeEcbPtr& ecb) {
            numStaticCmds += ecb->getNumStaticCmds();
        });

        LOG_DEBUG(RECIPE_GEN, "    Arc job for {}:", logicalIdStr);
        LOG_DEBUG(RECIPE_GEN, "      Num commands in static ECB = {}, Size = {}", numStaticCmds, staticCmdsSize);
        LOG_DEBUG(RECIPE_GEN, "      Num engine-specific static ECBs = {}, Size of one = {}", numEngEcbs, engEcbSize);
        LOG_DEBUG(RECIPE_GEN, "      Num commands in dynamic ECB = {}, Size = {}", numDynamicCmds, dynamicCmdsSize);

        if (numStaticCmds > 0)
        {
            LOG_DEBUG(RECIPE_GEN, "      Static commands (open GC_ARC log-type):");
            for (const auto& ecb : ecbVec.second)
            {
                ecb->printCmds(true);
            }
        }
        if (numDynamicCmds > 0)
        {
            LOG_DEBUG(RECIPE_GEN, "      Dynamic commands (open GC_ARC log-type):");
            ecbVec.second[0]->printCmds(false);  // only the first ECB has dynamic commands
        }
    }
}
