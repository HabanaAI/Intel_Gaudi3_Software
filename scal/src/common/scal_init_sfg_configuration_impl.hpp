#include "scal_sfg_configuration_helpers.h"
#include "scal_base.h"
#include "infra/monitor.hpp"
#include "infra/sob.hpp"

using CqEn      = Monitor::CqEn;
using LongSobEn = Monitor::LongSobEn;
using LbwEn     = Monitor::LbwEn;

template <template<class> class TsfgMonitorsHierarchyMetaData, class TSobTypes>
inline void Scal::configSfgMetaData(const CompletionGroup *cg, TsfgMonitorsHierarchyMetaData<TSobTypes> &sfgMD, uint32_t maxNbMessagesPerMonitor)
{
    const CompletionGroupSFGInfo& sfgInfo = cg->sfgInfo;

    // Fill engines type related info
    uint32_t baseSobIdx = sfgInfo.sfgSosPool->nextAvailableIdx;
    for (uint32_t engType = 0; engType < uint32_t(EngineTypes::items_count); engType++)
    {
        auto clusterIt = m_clusters.find(getEngTypeStr(EngineTypes(engType)));
        if (clusterIt == m_clusters.end())
        {
            continue;
        }
        sfgMD.engines[engType].engineType = getEngTypeStr(EngineTypes(engType));
        sfgMD.engines[engType].numOfPhysicalEngines = clusterIt->second.engines.size();
        sfgMD.engines[engType].baseSobIdx = baseSobIdx;
        sfgMD.engines[engType].sfgSignals = engType == uint32_t(EngineTypes::mme) ? getNbMmeCompletionSignals() : 1;
        if (engType == uint32_t(EngineTypes::mme))
        {
            sfgMD.engines[engType].numOfPhysicalEngines *= getMmeEnginesMultiplicationFactor();
        }
        sfgMD.numOfComputeEngines += sfgMD.engines[engType].numOfPhysicalEngines;
        const_cast<CompletionGroup*>(cg)->sfgInfo.baseSfgSob[engType] = baseSobIdx;
        baseSobIdx += sfgMD.engines[engType].numOfPhysicalEngines;
    }

    // Fill SM related info
    sfgMD.smIdx            = sfgInfo.sfgMonitorsPool->smIndex;
    sfgMD.smBaseAddr       = sfgInfo.sfgMonitorsPool->smBaseAddr;
    sfgMD.longSoSmIdx      = cg->longSosPool->smIndex;
    sfgMD.longSoSmBaseAddr = cg->longSosPool->smBaseAddr;
    sfgMD.cqSmIdx          = cg->syncManager->smIndex;
    sfgMD.cqSmBaseAddr     = cg->syncManager->baseAddr;

    // Fill monitors related info
    sfgMD.baseMonIdx     = sfgInfo.sfgMonitorsPool->nextAvailableIdx;
    sfgMD.curMonIdx      = sfgMD.baseMonIdx;
    sfgMD.baseCqMonIdx   = sfgInfo.sfgCqMonitorsPool->nextAvailableIdx;
    sfgMD.curCqMonIdx    = sfgMD.baseCqMonIdx;

    // Fill sync objects related info
    sfgMD.baseSobIdx     = sfgInfo.sfgSosPool->nextAvailableIdx;
    sfgMD.curSobIdx      = sfgMD.baseSobIdx;
    sfgMD.interSobIdx    = sfgMD.baseSobIdx + sfgMD.numOfComputeEngines;
    sfgMD.rearmSobIdx    = sfgMD.interSobIdx + 1;
    sfgMD.longSobIdx     = cg->longSoIndex;
    sfgMD.cqIdx          = cg->cqIdx;

    // Common payload data to inc a sob val by 1
    sfgMD.incSobPayloadData = TSobTypes::SobX::buildVal(1, SobLongSobEn::off, SobOp::inc);

    // Common payload data to dec a sob val by 1
    sfgMD.decSobPayloadData = TSobTypes::SobX::buildVal(-1, SobLongSobEn::off, SobOp::inc);

    sfgMD.maxNbMessagesPerMonitor       = maxNbMessagesPerMonitor;

    LOG_INFO(SCAL,"{}: SFG: SFG conf for completion group {}: smIdx {}, numOfComputeEngines {}, baseMonIdx SM{}_MON_{}, "
                  "baseSobIdx SM{}_SOB_{}, Intermediate-SSOB SM{}_SOB_{}, Rearm-SSOB SM{}_SOB_{}",
                  __FUNCTION__, cg->name, sfgMD.smIdx, sfgMD.numOfComputeEngines, sfgMD.smIdx, sfgMD.baseMonIdx,
                  sfgMD.smIdx, sfgMD.baseSobIdx, sfgMD.smIdx, sfgMD.interSobIdx, sfgMD.smIdx, sfgMD.rearmSobIdx);
}

/**********************************************************************************************************************************************************************************************************/
/* The below function configures the SFG sync hierarchy of monitors and sync objects.
* Each layer in the Hierarchy is configured in a different function and will be described in details within the layer's functions.
*
*
*                                   sobId 0      sobId 1     sobId 2 ......................................................... sobId numEngs-1
*                                 ------------ ------------ -------------     ------------             ------------ ------------ ------------
*                                 | SSOB MME | | SSOB MME | | SSOB EDMA |.....| SSOB TPC |......24.....| SSOB TPC | | SSOB ROT | | SSOB ROT |
*                                 ------------ ------------ -------------     ------------             ------------ ------------ ------------
*    max number of messages in G2: 4, G3: 16
*    ------------------------------  ------------------------------  ------------------------------  ------------------------------                  -----------------------------------------------
*    | SMON group: sobs(0-2) >= 1 |  | SMON group: sobs(3-5) >= 1 |  | SMON group: sobs(6-7) >= 1 |  | SMON group: sobs(8-10) >= 1|                  | SMON group: sobs(numEngs-3 - numEngs-1) >= 1|
*    |                            |  |                            |  |                            |  |                            |                  |                                             |
*    | msg0:SSOB0 Auto Dec        |  | msg0:SSOB3 Auto Dec        |  | msg0:SSOB6 Auto Dec        |  | msg0:SSOB8 Auto Dec        | ................ | msg0:SSOB(numEngs-3) Auto Dec               |
*    | msg1:SSOB1 Auto Dec        |  | msg1:SSOB4 Auto Dec        |  | msg1:SSOB7 Auto Dec        |  | msg1:SSOB9 Auto Dec        |                  | msg1:SSOB(numEngs-2) Auto Dec               |
*    | msg2:SSOB2 Auto Dec        |  | msg2:SSOB5 Auto Dec        |  | msg2:Intermediate-SSOB Inc |  | msg2:SSOB10 Auto Dec       |                  | msg2:SSOB(numEngs-1) Auto Dec               |
*    | msg3:Intermediate-SSOB Inc |  | msg3:Intermediate-SSOB Inc |  |                            |  | msg3:Intermediate-SSOB Inc |                  | msg3:Intermediate-SSOB Inc                  |
*    ------------------------------  ------------------------------  ------------------------------  ------------------------------                  -----------------------------------------------
*
*                                                              ---------------------
*                                                              | Intermediate-SSOB |
*                                                              ---------------------
*
*                                                                             (On CQ's DCORE - Can increase CQ only from within its DCORE)
*                                 --------------------------------------      ------------------------------
*                                 | SMON group: Intermediate-SSOB >= N |      | SMON group: Arm 0x0        |
*                                 |                                    |      |                            |
*                                 | msg0:Intermediate-SSOB Dec by N    |      | msg0:CQ Inc                |
*                                 | msg1:LongSO Inc                    | ---> | msg1:Rearm-SSOB Inc        |
*                                 | msg2:Arm and trigger CQ mon        |      |                            |
*                                 |                                    |      |                            |
*                                 --------------------------------------      ------------------------------
*
*                                                                  --------------
*                                                                  | Rearm-SSOB |
*                                                                  --------------
*
*          == Dec Rearm-SSOB monitor ==          ============================================= ReArm monitor CHAIN ==================================================
*                  HEAD of CHAIN                                                                                                               TAIL of CHAIN
*                                                max number of messages in G2: 4, G3: 16
*          ------------------------------       -------------------------------      -------------------------------                    ------------------------------
*          | SMON group: Rearm-SSOB >= 1 |      | SMON group: Rearm-SSOB >= 0 |      | SMON group: Rearm-SSOB >= 0 |                    | SMON group: Rearm-SSOB >= 0|
*          |                             |      |                             |      |                             |                    |                            |
*          | msg0:Rearm-SSOB Auto Dec    |      | msg0:Rearm SMON             |      | msg0:Rearm SMON             |                    | msg0:Rearm SMON            |
*          | msg1:Rearm self             | ---> | msg1:Rearm SMON             | ---> | msg1:Rearm SMON             | --- > ....... ---> | msg1:Rearm SMON            |
*          | msg2:ARM next mon in chain  |      | msg2:Rearm SMON             |      | msg2:Rearm SMON             |                    | msg2:Rearm SMON            |
*          |                             |      | msg3:ARM next mon in chain  |      | msg3:ARM next mon in chain  |                    |                            |
*          -------------------------------      -------------------------------      ------------------------------                     ------------------------------
*
***********************************************************************************************************************************************************************************************************/
template <class TsfgMonitorsHierarchyMetaData>
void Scal::configSfgSyncHierarchy(Qman::Program & prog, const CompletionGroup *cg, uint32_t maxNbMessagesPerMonitor)
{
    LOG_INFO(SCAL,"{}: SFG: ================================ SFG Configuration - START ===============================================", __FUNCTION__);
    LOG_INFO(SCAL,"{}: SFG: Start SFG monitor and sync objs configuration for completion group {}", __FUNCTION__, cg->name);

/* ===================== Get SFG Meta Data ======================== */

    TsfgMonitorsHierarchyMetaData sfgMD;
    configSfgMetaData(cg, sfgMD, maxNbMessagesPerMonitor);

/* =================== Create SFG Hierarchy ======================= */

    configSfgFirstLayer(prog, sfgMD);
    configSfgLongSoAndCqLayer(prog, sfgMD);
    configSfgRearmLayer(prog, sfgMD);

/* ================= Set Indexes for next stream ================== */

    // Set next available monitors for next completion grou of the next stream
    cg->sfgInfo.sfgMonitorsPool->nextAvailableIdx = sfgMD.curMonIdx;
    cg->sfgInfo.sfgCqMonitorsPool->nextAvailableIdx += 4;  // We use 4 monitors (messages for the CQ layer)

    // Keep offset for fillEngineConfigs
    const_cast<CompletionGroup*>(cg)->sfgInfo.sobsOffsetToNextStream = sfgMD.curSobIdx - cg->sfgInfo.sfgSosPool->nextAvailableIdx;
    cg->sfgInfo.sfgSosPool->nextAvailableIdx = sfgMD.curSobIdx;

    LOG_INFO(SCAL,"{}: SFG: Finish successfully SFG monitor and sync objs configuration for completion group {}", __FUNCTION__, cg->name);
    LOG_INFO(SCAL,"{}: SFG: ================================ SFG Configuration - FINISH ==============================================", __FUNCTION__);;
}

/**********************************************************************************************************************************************************************************************************/
/*
* Engines SFG SOBs (SSOB):
*
*       sobId 0      sobId 1     sobId 2 ......................................................... sobId numEngs-1
*    ------------ ------------ -------------     ------------             ------------ ------------ ------------
*    | SSOB MME | | SSOB MME | | SSOB EDMA |.....| SSOB TPC |......24.....| SSOB TPC | | SSOB ROT | | SSOB ROT |
*    ------------ ------------ -------------     ------------             ------------ ------------ ------------
*
* First layer of SFG monitors (SMON):
* - Each monitor master has up to 4 massages
* - A monitor can monitor only sobs from the same group (8 sobs in a single group)
* - Each monitor will monitor up to 3(G2)8(G3) SSOBs of physical engines to reach val >= 1
* - Each monitor will dec by 1 the SSOBs he is monitoring
* - Each monitor will inc by 1 the Intermediate-SSOB
*
*      ================================= First SOBs group (8 sobs) ===========================           ===== Second SOBs group ==== ..................................===== Last SOBs group ====
*    ------------------------------  ------------------------------  ------------------------------  ------------------------------                  -----------------------------------------------
*    | SMON group: sobs(0-2) >= 1 |  | SMON group: sobs(3-5) >= 1 |  | SMON group: sobs(6-7) >= 1 |  | SMON group: sobs(8-10) >= 1|                  | SMON group: sobs(numEngs-3 - numEngs-1) >= 1|
*    |                            |  |                            |  |                            |  |                            |                  |                                             |
*    | msg0:SSOB0 Auto Dec        |  | msg0:SSOB3 Auto Dec        |  | msg0:SSOB6 Auto Dec        |  | msg0:SSOB8 Auto Dec        | ................ | msg0:SSOB(numEngs-3) Auto Dec               |
*    | msg1:SSOB1 Auto Dec        |  | msg1:SSOB4 Auto Dec        |  | msg1:SSOB7 Auto Dec        |  | msg1:SSOB9 Auto Dec        |                  | msg1:SSOB(numEngs-2) Auto Dec               |
*    | msg2:SSOB2 Auto Dec        |  | msg2:SSOB5 Auto Dec        |  | msg2:Intermediate-SSOB Inc |  | msg2:SSOB10 Auto Dec       |                  | msg2:SSOB(numEngs-1) Auto Dec               |
*    | msg3:Intermediate-SSOB Inc |  | msg3:Intermediate-SSOB Inc |  |                            |  | msg3:Intermediate-SSOB Inc |                  | msg3:Intermediate-SSOB Inc                  |
*    ------------------------------  ------------------------------  ------------------------------  ------------------------------                  -----------------------------------------------
*
* Intermediate SFG SOB:
*
*        sobId numEngs
*    ---------------------
*    | Intermediate-SSOB |
*    ---------------------
*
**********************************************************************************************************************************************************************************************************/
template <template<class> class TsfgMonitorsHierarchyMetaData, class TSobTypes>
void Scal::configSfgFirstLayer(Qman::Program & prog, TsfgMonitorsHierarchyMetaData<TSobTypes> &sfgMD)
{
    LOG_INFO(SCAL,"{}: SFG: ================================ SFG Configuration - First Hierarch LAYER ================================", __FUNCTION__);

    // For each engine compute cluster, loop over its physical engines and allocate SOBs
    for (uint32_t engType = 0; engType < unsigned(EngineTypes::items_count); engType++)
    {
        LOG_INFO(SCAL,"{}: SFG: Config first layer for engine type cluster {}", __FUNCTION__, getEngTypeStr(EngineTypes(engType)));

        EngineTypeInfo const & engInfo       = sfgMD.engines[engType];
        sfgMD.decSobPayloadData = TSobTypes::SobX::setVal(sfgMD.decSobPayloadData, 0x7FFF - engInfo.sfgSignals + 1); // Signed (goes with 7FFF to create decrement)

        for(uint32_t engIdx = 0; engIdx < engInfo.numOfPhysicalEngines;)
        {
            const unsigned maxNbDecMsgPerMon = sfgMD.maxNbMessagesPerMonitor - 1;
            const unsigned maxNbSobPerMonGroup  = c_sync_object_group_size; // G3 => 8 as well

            uint32_t mon[maxNbDecMsgPerMon];
            uint32_t sob[maxNbDecMsgPerMon];
            uint32_t groupId[maxNbDecMsgPerMon];

            uint32_t numOfSobsInMonGroup = 0;
            for (unsigned i = 0 ; i < maxNbDecMsgPerMon; ++i)
            {
                mon[i] = sfgMD.curMonIdx + i; // i == 0 => master monitor
                sob[i] = sfgMD.curSobIdx + i;
                groupId[i] = sob[i] / maxNbSobPerMonGroup;
                // sob must be within the same group as sob[0] and withing numOfPhysicalEngines
                if (groupId[i] != groupId[0] || sob[i] >= (engInfo.baseSobIdx + engInfo.numOfPhysicalEngines))
                {
                    break;
                }
                numOfSobsInMonGroup++;
            }
            LOG_INFO(SCAL,"{}: SFG: Config master mon SM{}_MON_{} with {} messages: dec {} "
                          "seq sobs start at SM{}_SOB_{} and inc Intermediate-SSOB SM{}_SOB_{}",
                          __FUNCTION__, sfgMD.smIdx, mon[0], numOfSobsInMonGroup + 1, numOfSobsInMonGroup,
                          sfgMD.smIdx, sob[0], sfgMD.smIdx, sfgMD.interSobIdx);
            /* ====================================================================================== */
            /*                                  Create a monitor group                                */
            /* ====================================================================================== */

            // Need to write up to maxNbMessagesPerMonitor messages (up to maxNbMessagesPerMonitor-1 sobs decrements + inc Intermediate-SSOB)
            uint32_t groupMonitor = TSobTypes::MonitorX::buildConfVal(sfgMD.rearmSobIdx, numOfSobsInMonGroup, CqEn::off, LongSobEn::off, LbwEn::off);

            /* ====================================================================================== */
            /*                              Add massages to monitor group                             */
            /* ====================================================================================== */

            // msg[0..numOfSobsInMonGroup-1] Auto dec for first sob we are monitoring in this group
            for (unsigned i = 0; i < numOfSobsInMonGroup; ++i)
            {
                LOG_DEBUG(SCAL,"{}: SFG: Configure SM{}_MON_{} to dec SM{}_SOB_{} by {}",
                               __FUNCTION__, sfgMD.smIdx, mon[i], sfgMD.smIdx, sob[i], engInfo.sfgSignals);
                configureMonitor(prog, mon[i], sfgMD.smBaseAddr, groupMonitor, TSobTypes::SobX::getAddr(sfgMD.smBaseAddr, sob[i]), sfgMD.decSobPayloadData);
            }

            // last msg:   Increment of the Intermediate-SSOB
            uint32_t intermediateMon = mon[0] + numOfSobsInMonGroup;
            uint64_t intermediateSobAddr = TSobTypes::SobX::getAddr(sfgMD.smBaseAddr, sfgMD.interSobIdx);

            LOG_DEBUG(SCAL,"{}: SFG: Configure SM{}_MON_{} to inc Intermediate-SSOB SM{}_SOB_{} by 1",
                           __FUNCTION__, sfgMD.smIdx, intermediateMon, sfgMD.smIdx, sfgMD.interSobIdx);
            configureMonitor(prog, intermediateMon, sfgMD.smBaseAddr, groupMonitor, intermediateSobAddr, sfgMD.incSobPayloadData);
            /* ====================================================================================== */
            /*                     ARM the group monitor to monitor up to numOfSobsInMonGroup ssobs   */
            /* ====================================================================================== */
            uint8_t mask = 0xFF;
            for (unsigned i = 0 ; i < numOfSobsInMonGroup; ++i)
            {
                mask &= ~(1 << (sob[i] % c_sync_object_group_size));
            }
            uint32_t target = engInfo.sfgSignals;
            uint32_t armMonPayloadData = TSobTypes::MonitorX::buildArmVal(sob[0], target, mask);

            LOG_DEBUG(SCAL,"{}: SFG: Arm master SM{}_MON_{} with payload data {:#x}, start sobId SM{}_SOB_{}, sobIdGroup {}, mask {:0>8b}, targetVal {}",
                           __FUNCTION__, sfgMD.smIdx, mon[0], armMonPayloadData, sfgMD.smIdx, sob[0], groupId[0], mask, target);

            uint64_t masterMonAddr = typename TSobTypes::MonitorX(sfgMD.smBaseAddr, mon[0]).getRegsAddr().arm;
            prog.addCommand(MsgLong(masterMonAddr, armMonPayloadData));

            /* ====================================================================================== */
            /*                                   Update indexes                                       */
            /* ====================================================================================== */

            // Keep information of the monitors to rearm in the last layer in hierarchy
            RearmMonPayload<TSobTypes> monInfo;
            monInfo.monIdx   = mon[0];
            monInfo.monAddr  = masterMonAddr;
            monInfo.data     = armMonPayloadData;
            sfgMD.rearmMonitors.push_back(monInfo);

            // We used numOfSobsInMonGroup sobs in this mon group
            sfgMD.curSobIdx += numOfSobsInMonGroup;
            engIdx += numOfSobsInMonGroup;

            // We used numOfSobsInMonGroup (ssob auto dec) + 1 (inc intermediate ssob) monitors in this group
            sfgMD.curMonIdx += numOfSobsInMonGroup + 1;
        }
    }
    // Set back decrement value by 1
    sfgMD.decSobPayloadData = TSobTypes::SobX::setVal(sfgMD.decSobPayloadData, 0x7FFF);
}

/******************************************************************************/
/*                                                                            *
* Intermediate SFG SOB:                                                       *
*                                                                             *
*        sobId numEngs                                                        *
*    ---------------------                                                    *
*    | Intermediate-SSOB |                                                    *
*    ---------------------                                                    *
*                                                                             *
* Monitor for longSO and Monitor for CQ:                                      *
* N - Num of group monitors from first hierarchy layer                        *
*                                                                             *
*                                              (On CQ's DCORE - Can increase  *
*                                              CQ only from within its DCORE) *
* --------------------------------------      ------------------------------  *
* | SMON group: Intermediate-SSOB >= N |      | SMON group: Arm 0x0        |  *
* |                                    |      |                            |  *
* | msg0:Intermediate-SSOB Dec         |      | msg0:CQ Inc                |  *
* | msg1:LongSO Inc                    | ---> | msg1:Rearm-SSOB Inc        |  *
* | msg2:Arm and trigger CQ mon        |      |                            |  *
* |                                    |      |                            |  *
* --------------------------------------      ------------------------------  *
*                                                                             *
*    --------------                                                           *
*    | Rearm-SSOB |                                                           *
*    --------------                                                           *
*                                                                             *
******************************************************************************/
template <template<class> class TsfgMonitorsHierarchyMetaData, class TSobTypes>
void Scal::configSfgLongSoAndCqLayer(Qman::Program & prog, TsfgMonitorsHierarchyMetaData<TSobTypes> &sfgMD)
{
    uint32_t mon1 = sfgMD.curMonIdx + 0; // Master monitor for long so inc
    uint32_t mon2 = sfgMD.curMonIdx + 1;
    uint32_t mon3 = sfgMD.curMonIdx + 2;

    uint32_t cqMon1 = sfgMD.curCqMonIdx + 0; // Master monitor for CQ inc
    uint32_t cqMon2 = sfgMD.curCqMonIdx + 1;

    LOG_INFO(SCAL,"{}: SFG: =========================== SFG Configuration - LongSO And CQ Hierarch LAYER =============================", __FUNCTION__);
    LOG_INFO(SCAL,"{}: SFG: Config master mon SM{}_MON_{} with 3 messages: dec Intermediate-SSOB SM{}_SOB_{}, inc longSO SM{}_SOB_{}, Arm CQ monitor on SM{} to trigger immediately",
                  __FUNCTION__, sfgMD.smIdx, mon1, sfgMD.smIdx, sfgMD.interSobIdx, sfgMD.longSoSmIdx, sfgMD.longSobIdx, sfgMD.cqSmIdx);
    LOG_INFO(SCAL,"{}: SFG: Config master mon SM{}_MON_{} with 2 messages: inc Rearm-SSOB SM{}_SOB_{} and inc CQ {} in SM{}",
                  __FUNCTION__, sfgMD.cqSmIdx, cqMon1, sfgMD.smIdx, sfgMD.rearmSobIdx, sfgMD.cqIdx, sfgMD.cqSmIdx);

    /* ====================================================================================== */
    /*                                  Create a monitor group                                */
    /* ====================================================================================== */

    // Need 3 writes (dec Intermediate-SSOB, inc longSO, Arm CQ monitor)
    uint32_t groupLongSoMonitor = TSobTypes::MonitorX::buildConfVal(sfgMD.interSobIdx, 2, CqEn::off, LongSobEn::off, LbwEn::off);

//    typename TSobTypes::reg_mon_config groupCqMonitor = getRegMonConfig<TSobTypes>(1, 0);

    /* ====================================================================================== */
    /*                              Add massages to monitor groups                            */
    /* ====================================================================================== */

    // msg0 (groupLongSoMonitor):     Decrement Intermediate-SSOB
    sfgMD.decSobPayloadData = TSobTypes::SobX::setVal(sfgMD.decSobPayloadData, 0x7FFF - sfgMD.rearmMonitors.size() + 1);

    LOG_DEBUG(SCAL,"{}: SFG: Configure SM{}_MON_{} to dec Intermediate-SSOB SM{}_SOB_{} by {}",
                   __FUNCTION__, sfgMD.smIdx, mon1, sfgMD.smIdx, sfgMD.interSobIdx, sfgMD.rearmMonitors.size());
    configureMonitor(prog, mon1, sfgMD.smBaseAddr, groupLongSoMonitor, TSobTypes::SobX::getAddr(sfgMD.smBaseAddr, sfgMD.interSobIdx), sfgMD.decSobPayloadData);

    // Set back decrement value by 1
    sfgMD.decSobPayloadData = TSobTypes::SobX::setVal(sfgMD.decSobPayloadData, 0x7FFF); // Signed (goes with 7FFF to create decrement)

    // msg1 (groupLongSoMonitor):     Increment longSO
    uint64_t longSoAddr = TSobTypes::SobX::getAddr(sfgMD.longSoSmBaseAddr, sfgMD.longSobIdx);

    sfgMD.incSobPayloadData = TSobTypes::SobX::setLongEn(sfgMD.incSobPayloadData, SobLongSobEn::on);
    LOG_DEBUG(SCAL,"{}: SFG: Configure SM{}_MON_{} to inc cg's SM{}_SOB_{} by 1",
                   __FUNCTION__, sfgMD.smIdx, mon2, sfgMD.longSoSmIdx, sfgMD.longSobIdx);
    configureMonitor(prog, mon2, sfgMD.smBaseAddr, groupLongSoMonitor, longSoAddr, sfgMD.incSobPayloadData);
    sfgMD.incSobPayloadData = TSobTypes::SobX::setLongEn(sfgMD.incSobPayloadData, SobLongSobEn::off);

    // msg0 (groupCqMonitor):         Increment CQ
    // Need 2 writes (inc Rearm-SSOB, inc CQ)
    uint32_t groupCqMonitor = TSobTypes::MonitorX::buildConfVal(0, 1, CqEn::on, LongSobEn::off, LbwEn::on);
    LOG_DEBUG(SCAL,"{}: SFG: Configure SM{}_MON_{} to inc cg's CQ {}",
                   __FUNCTION__, sfgMD.cqSmIdx, cqMon1, sfgMD.cqIdx);
    configureMonitor(prog, cqMon1, sfgMD.cqSmBaseAddr, groupCqMonitor, (uint64_t)sfgMD.cqIdx, 0x1);

    // msg1 (groupCqMonitor):         Increment Rearm-SSOB
    groupCqMonitor = TSobTypes::MonitorX::buildConfVal(0, 1, CqEn::off, LongSobEn::off, LbwEn::off);
    LOG_DEBUG(SCAL,"{}: SFG: Configure SM{}_MON_{} to inc Rearm-SSOB SM{}_SOB_{} by 1",
                   __FUNCTION__, sfgMD.cqSmIdx, cqMon2, sfgMD.smIdx, sfgMD.rearmSobIdx);

    configureMonitor(prog, cqMon2, sfgMD.cqSmBaseAddr, groupCqMonitor, TSobTypes::SobX::getAddr(sfgMD.smBaseAddr, sfgMD.rearmSobIdx), sfgMD.incSobPayloadData);

    // msg2 (groupLongSoMonitor):     Arm CQ monitor to trigger immediately
    LOG_DEBUG(SCAL,"{}: SFG: Configure SM{}_MON_{} to Arm CQ monitor on SM{} to trigger immediately",
                   __FUNCTION__, sfgMD.smIdx, mon3, sfgMD.cqSmIdx);

    uint64_t masterCqMonAddr = typename TSobTypes::MonitorX(sfgMD.cqSmBaseAddr, cqMon1).getRegsAddr().arm;
    configureMonitor(prog, mon3, sfgMD.smBaseAddr, groupLongSoMonitor, masterCqMonAddr, 0x0);

    /* ====================================================================================== */
    /*                     ARM the group monitor to monitor Intermediate-SSOB                 */
    /* ====================================================================================== */

    uint32_t armMonPayloadData = TSobTypes::MonitorX::buildArmVal(sfgMD.interSobIdx, sfgMD.rearmMonitors.size());
    uint64_t masterMonAddr = typename TSobTypes::MonitorX(sfgMD.smBaseAddr, mon1).getRegsAddr().arm;

    LOG_DEBUG(SCAL,"{}: SFG: Arm master SM{}_MON_{} with payload data {:#x}, SM{}_SOB_{}, targetVal {}",
                   __FUNCTION__, sfgMD.smIdx, mon1, armMonPayloadData, sfgMD.smIdx, sfgMD.interSobIdx, sfgMD.rearmMonitors.size());
    prog.addCommand(MsgLong(masterMonAddr, armMonPayloadData));

    // Keep information of the monitors to rearm in the last layer in hierarchy
    RearmMonPayload<TSobTypes> monInfo;
    monInfo.monIdx   = mon1;
    monInfo.monAddr  = masterMonAddr;
    monInfo.data     = armMonPayloadData;
    sfgMD.rearmMonitors.push_back(monInfo);

    /* ====================================================================================== */
    /*                                   Update indexes                                       */
    /* ====================================================================================== */

    // For Intermediate-SSOB and Rearm-SSOB
    sfgMD.curSobIdx += 2;

    // For 3 writes in groupLongSoMonitor
    sfgMD.curMonIdx += 3;

    // For 2 writes in groupCqMonitor
    sfgMD.curCqMonIdx += 2;
}

/********************************************************************************************************************************************************************************/
/*                                                                                                                                                                              *
* Rearm SFG SOB:                                                                                                                                                                *
*                                                                                                                                                                               *
*    sobId numEngs+1                                                                                                                                                            *
*    --------------                                                                                                                                                             *
*    | Rearm-SSOB |                                                                                                                                                             *
*    --------------                                                                                                                                                             *
*                                                                                                                                                                               *
* Rearm layer of SFG monitors - monitors chain:                                                                                                                                 *
* - Head of chain will monitor the Rearm-SSOB to reach val >= 1                                                                                                                 *
* - When fire, it will dec Rearm-SSOB, rearm self and ARM the next monitor in chain                                                                                             *
* - Rest of monitors in chain will rearm up to 3 monitors in the hierarchy and ARM the next monitor in chain                                                                    *
*                                                                                                                                                                               *
*          == Dec Rearm-SSOB monitor ==          ============================================= ReArm monitor CHAIN ==================================================           *
*                  HEAD of CHAIN                                                                                                               TAIL of CHAIN                    *
*          ------------------------------       -------------------------------      -------------------------------                    ------------------------------          *
*          | SMON group: Rearm-SSOB >= 1 |      | SMON group: Rearm-SSOB >= 0 |      | SMON group: Rearm-SSOB >= 0 |                    | SMON group: Rearm-SSOB >= 0|          *
*          |                             |      |                             |      |                             |                    |                            |          *
*          | msg0:Rearm-SSOB Auto Dec    |      | msg0:Rearm SMON             |      | msg0:Rearm SMON             |                    | msg0:Rearm SMON            |          *
*          | msg1:Rearm self             | ---> | msg1:Rearm SMON             | ---> | msg1:Rearm SMON             | --- > ....... ---> | msg1:Rearm SMON            |          *
*          | msg2:ARM next mon in chain  |      | msg2:Rearm SMON             |      | msg2:Rearm SMON             |                    | msg2:Rearm SMON            |          *
*          |                             |      | msg3:ARM next mon in chain  |      | msg3:ARM next mon in chain  |                    |                            |          *
*          -------------------------------      -------------------------------      ------------------------------                     ------------------------------          *
*                                                                                                                                                                               *
********************************************************************************************************************************************************************************/
template <template<class> class TsfgMonitorsHierarchyMetaData, class TSobTypes>
void Scal::configSfgRearmLayer(Qman::Program & prog, TsfgMonitorsHierarchyMetaData<TSobTypes> &sfgMD)
{
    uint32_t mon1 = sfgMD.curMonIdx + 0; // Master monitor
    uint32_t mon2 = sfgMD.curMonIdx + 1;
    uint32_t mon3 = sfgMD.curMonIdx + 2;

    LOG_INFO(SCAL,"{}: SFG: ============================== SFG Configuration - Rearm CHAIN LAYER =====================================", __FUNCTION__);;
    LOG_INFO(SCAL,"{}: SFG: Config master mon SM{}_MON_{} with 3 messages: dec Rearm-SSOB {}, rearm self, arm first mon in chain",
                  __FUNCTION__, sfgMD.smIdx, mon1, sfgMD.rearmSobIdx);

    /* ====================================================================================== */
    /*                        Create a monitor group (Head of chain)                          */
    /* ====================================================================================== */
    // Need 3 writes (dec for Rearm-SSOB + rearm self + arm next mon in chain)
    uint32_t groupMonitor = TSobTypes::MonitorX::buildConfVal(sfgMD.rearmSobIdx, 2, CqEn::off, LongSobEn::off, LbwEn::off);

    /* ====================================================================================== */
    /*                              Add massages to monitor group                             */
    /* ====================================================================================== */

    uint32_t armMonPayloadData = TSobTypes::MonitorX::buildArmVal(sfgMD.rearmSobIdx, 1);

    // msg0:   Decrement Rearm-SSOB
    LOG_DEBUG(SCAL,"{}: SFG: Configure SM{}_MON_{} to dec Rearm-SSOB SM{}_SOB_{} by 1", __FUNCTION__, sfgMD.smIdx, mon1, sfgMD.smIdx, sfgMD.rearmSobIdx);
    configureMonitor(prog, mon1, sfgMD.smBaseAddr, groupMonitor, TSobTypes::SobX::getAddr(sfgMD.smBaseAddr, sfgMD.rearmSobIdx), sfgMD.decSobPayloadData);

    // msg1:   Rearm self
    uint64_t masterMonAddr = typename TSobTypes::MonitorX(sfgMD.smBaseAddr, mon1).getRegsAddr().arm;
    LOG_DEBUG(SCAL,"{}: SFG: Configure SM{}_MON_{} to rearm self", __FUNCTION__, sfgMD.smIdx, mon2);
    configureMonitor(prog, mon2, sfgMD.smBaseAddr, groupMonitor, masterMonAddr, armMonPayloadData);

    // msg3:   Keep info for the next monitor group in chain to ARM current mon
    uint32_t prevMonIdx = mon3;

    /* ====================================================================================== */
    /*                       ARM the group monitor to monitor Rearm-SSOB                      */
    /* ====================================================================================== */

    LOG_DEBUG(SCAL,"{}: SFG: Arm master SM{}_MON_{} with payload data {:#x}, SM{}_SOB_{}, targetVal 1",
                   __FUNCTION__, sfgMD.smIdx, mon1, armMonPayloadData, sfgMD.smIdx, sfgMD.rearmSobIdx);
    prog.addCommand(MsgLong(masterMonAddr, armMonPayloadData));

    /* ====================================================================================== */
    /*                                   Update indexes                                       */
    /* ====================================================================================== */

    // We used 3 monitors for this monitor group (dec of Rearm-SSOB + rearm self + arm next monitor in chain)
    sfgMD.curMonIdx += 3;

    /* ====================================================================================== */
    /*                                      ARM chain                                         */
    /* ====================================================================================== */

    configSfgChainInRearmLayer(prog, sfgMD, prevMonIdx);
}
template <template<class> class TsfgMonitorsHierarchyMetaData, class TSobTypes>
void Scal::configSfgChainInRearmLayer(Qman::Program & prog, TsfgMonitorsHierarchyMetaData<TSobTypes> &sfgMD, uint32_t prevMonIdx)
{
    // Loop over all monitors we need to rearm
    for (uint32_t rearmMonIdxInVector = 0; rearmMonIdxInVector < sfgMD.rearmMonitors.size();)
    {
        uint32_t masterMon = sfgMD.curMonIdx; // Master monitor
        uint32_t firstRearmMonIdxInVector = rearmMonIdxInVector;

        // Can rearm up to (sfgMD.maxNbMessagesPerMonitor - 1) monitors (only tail can rearm sfgMD.maxNbMessagesPerMonitor)
        const uint32_t maxNbRearmMon = sfgMD.maxNbMessagesPerMonitor;

        // Indexes in vector of monitors to rearm
        uint32_t idx[maxNbRearmMon];
        std::iota(&idx[0], &idx[maxNbRearmMon], rearmMonIdxInVector);
        uint32_t numOfMonsToRearm = std::clamp<uint32_t>(rearmMonIdxInVector + maxNbRearmMon, 0, sfgMD.rearmMonitors.size()) - rearmMonIdxInVector;
        const bool isTail = rearmMonIdxInVector + numOfMonsToRearm == sfgMD.rearmMonitors.size();
        // only tail can rearm maxNbRearmMon monitors because it does not need to arm the next monitor group in the chain
        if (numOfMonsToRearm == maxNbRearmMon && isTail == false)
        {
            numOfMonsToRearm--;
        }
        // Last monitor group in chain should not ARM next in chain, therefore need 1 less write
        const uint32_t numOfWrites = isTail ? numOfMonsToRearm - 1 : numOfMonsToRearm;
        /* ====================================================================================== */
        /*                                  Create a monitor group                                */
        /* ====================================================================================== */

        // Need to write numOfMonsToRearm messages (up to 4 monitors to rearm)
        uint32_t groupMonitor = TSobTypes::MonitorX::buildConfVal(sfgMD.rearmSobIdx, numOfWrites, CqEn::off, LongSobEn::off, LbwEn::off);

        LOG_INFO(SCAL,"{}: SFG: Config master mon SM{}_MON_{} with {} messages: ream sequential {} mons start at SM{}_MON_{}, ARM next in chain",
                 __FUNCTION__, sfgMD.smIdx, masterMon, numOfWrites + 1, numOfWrites, sfgMD.smIdx, sfgMD.rearmMonitors[idx[0]].monIdx);

        /* ====================================================================================== */
        /*                              Add massages to monitor group                             */
        /* ====================================================================================== */

        for (; rearmMonIdxInVector < firstRearmMonIdxInVector + numOfMonsToRearm; rearmMonIdxInVector++, sfgMD.curMonIdx++)
        {
            const uint32_t monToConfigure = sfgMD.curMonIdx;
            RearmMonPayload<TSobTypes> monToArm = sfgMD.rearmMonitors[rearmMonIdxInVector];

            LOG_DEBUG(SCAL,"{}: SFG: Configure SM{}_MON_{} to rearm SM{}_MON_{}",
                      __FUNCTION__, sfgMD.smIdx, monToConfigure, sfgMD.smIdx, monToArm.monIdx);
            configureMonitor(prog, monToConfigure, sfgMD.smBaseAddr, groupMonitor, monToArm.monAddr, monToArm.data);
        }

        /* ====================================================================================== */
        /*                        ARM the group monitor to fire immediately                       */
        /* ====================================================================================== */

        // Set prev monitor to rearm current
        uint32_t armMonPayloadData = TSobTypes::MonitorX::buildArmVal(sfgMD.rearmSobIdx, 0);

        uint64_t masterMonAddr = typename TSobTypes::MonitorX(sfgMD.smBaseAddr, masterMon).getRegsAddr().arm;
        LOG_DEBUG(SCAL,"{}: SFG: Configure SM{}_MON_{} to ARM (master DCORE{}_MON_{})", __FUNCTION__, sfgMD.smIdx, prevMonIdx, sfgMD.smIdx, masterMon);
        configureMonitor(prog, prevMonIdx, sfgMD.smBaseAddr, groupMonitor, masterMonAddr, armMonPayloadData);

        /* ====================================================================================== */
        /*                                   Update indexes                                       */
        /* ====================================================================================== */

        // msg4:   Keep info for the next monitor group in chain to ARM current mon
        prevMonIdx = sfgMD.curMonIdx;
        sfgMD.curMonIdx++;
    }
}
