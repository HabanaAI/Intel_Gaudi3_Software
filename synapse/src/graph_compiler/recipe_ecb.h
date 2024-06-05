#pragma once

#include <cstdint>
#include <vector>
#include <queue>
#include "recipe_program.h"
#include "recipe_blob.h"
#include "eng_arc_command.h"
#include "eng_arc_dccm_mngr.h"
#include "eng_arc_hooks.h"

class RecipeEcb;
struct arc_job_t;

using RecipeEcbPtr = std::shared_ptr<RecipeEcb>;

class RecipeEcb
{
public:
    RecipeEcb(HabanaDeviceType devType, std::shared_ptr<EngArcHooks> engArcHooks);
    RecipeEcb() {}

    void generateCmds(unsigned                   engineId,
                      const RecipeProgram&       program,
                      const RecipeBlobContainer& blobContainer,
                      bool                       needToCreateDynamicCmds);

    void roundUpStaticSize(uint64_t maxStaticSizeBytes);
    void finalize();
    void serializeCmds(uint8_t* pFiller, bool staticCmds) const;
    void printCmds(bool staticCmds) const;

    unsigned getNumStaticCmds() const { return m_staticCmds.size(); }
    uint64_t getStaticCmdsSize() const { return m_staticCmdsSize; }
    unsigned getNumDynamicCmds() const { return m_dynamicCmds.size(); }
    uint64_t getDynamicCmdsSize() const { return m_dynamicCmdsSize; }

protected:
    void addCmdsForWorkDistBlob(const RecipeBlob* pBlob, const Settable<BlobMetaData>& blobMD, bool yield);
    void addCmdsForStaticBlob(unsigned engineId, const RecipeBlob* pBlob, bool yield);

    void addCmdToQueue(std::vector<EngArcCmdPtr>& cmdsVec,
                       uint64_t&                  cmdsBinSize,
                       EngArcCmdPtr               cmd,
                       unsigned                   chunkSize) const;

    void addPaddingNop(std::vector<EngArcCmdPtr>& cmdsVec,
                       uint64_t&                  cmdsBinSize,
                       unsigned                   nextCmdBinSize,
                       unsigned                   chunkSize) const;

    EngArcCmdPtr popSuspendedExe();

    void initDynamicQueue(const RecipeProgram& program);

    HabanaDeviceType                         m_deviceType = LAST_HABANA_DEVICE;
    std::vector<EngArcCmdPtr>                m_staticCmds;
    std::vector<EngArcCmdPtr>                m_dynamicCmds;
    uint64_t                                 m_staticCmdsSize  = 0;  // in bytes
    uint64_t                                 m_dynamicCmdsSize = 0;  // in bytes
    std::shared_ptr<ListSizeEngArcCommand>   m_pStaticSizeCmd  = nullptr;
    std::shared_ptr<ListSizeEngArcCommand>   m_pDynamicSizeCmd = nullptr;
    EngArcDccmMngr                           m_dccmMngr;
    std::queue<EngArcCmdPtr>                 m_suspendedExeCmds;
    const unsigned                           m_suspensionLength    = 0;
    unsigned                                 m_suspendedWdCmdCount = 0;
    std::shared_ptr<EngArcHooks>             m_engArcHooks         = nullptr;
    unsigned                                 m_staticEcbChunkSize  = 0;
    unsigned                                 m_dynamicEcbChunkSize = 0;
    thread_local static std::shared_ptr<NopEngArcCommand> s_pFullChunkNop;
};

class CmeRecipeEcb : public RecipeEcb
{
public:
    CmeRecipeEcb(std::shared_ptr<EngArcHooks> engArcHooks);
    CmeRecipeEcb() = delete;
    virtual ~CmeRecipeEcb() = default;
    void registerCmds(const std::list<EngArcCmdPtr>& cmeCmds);
};

class RecipeEcbContainer
{
public:
    RecipeEcbContainer() = default;

    void generateECBs(const RecipeProgramContainer& programContainer, const RecipeBlobContainer& blobContainer);
    void serialize(uint32_t* pNumArcJobs, arc_job_t** ppArcJobs, RecipeAllocator* pRecipeAlloc) const;
    void print() const;
    void setEngArcHooks(std::shared_ptr<EngArcHooks> engArcHooks) { m_engArcHooks = engArcHooks; }
    void setEngArcHooksForCme(std::shared_ptr<EngArcHooks> engArcHooks) { m_engArcHooksForCme = engArcHooks; }
    void registerCmeCommands(const std::list<EngArcCmdPtr>& cmeCmds);

private:
    std::map<HabanaDeviceType, std::vector<RecipeEcbPtr>> m_ecbs;
    std::shared_ptr<EngArcHooks>                          m_engArcHooks       = nullptr;
    std::shared_ptr<EngArcHooks>                          m_engArcHooksForCme = nullptr;
};
