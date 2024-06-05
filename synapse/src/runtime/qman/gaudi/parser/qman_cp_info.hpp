#pragma once

#include "define.hpp"

#include "runtime/qman/common/parser/qman_cp_info.hpp"

#include <array>
#include <unordered_map>

namespace gaudi
{
class QmanCpInfo : virtual public common::QmanCpInfo
{
    using eMonitorSetupPhase = common::QmanCpInfo::eMonitorSetupPhase;

public:
    QmanCpInfo()          = default;
    virtual ~QmanCpInfo() = default;

    virtual bool parseSinglePacket(common::eCpParsingState state) override;

    // Return predicate-value or INVALID_PRED_VALUE in case not relevant
    uint16_t parseControlBlock(bool shouldPrintPred);

    virtual bool parseArbPoint() { return false; };  // Only relevant for the Upper-CP
    virtual bool parseCpDma() { return false; };     // Only relevant for the Upper-CP
    bool         parseLinDma();
    bool         parseMsgLong();
    bool         parseMsgShort();
    bool         parseMsgProt();
    bool         parseWreg32();
    bool         parseWregBulk();
    bool         parseNop();
    bool         parseStop();
    bool         parseFence(bool ignoreFenceConfig);
    bool         parseWait();
    bool         parseRepeat();
    bool         parseLoadAndExecute();

    void parseMsgShortBasic(uint32_t base, uint32_t operation, uint32_t msgAddressOffset, uint32_t value);

    bool parseSyncObjUpdate(void*                       pPacketBuffer,
                            uint32_t                    sobjAddressOffset,
                            uint32_t                    value,
                            gaudi::eSyncManagerInstance syncMgrInstance,
                            bool                        isMsgShort);

    bool parseMonitorPacket(void*                       pPacketBuffer,
                            uint32_t                    monitorAddressOffset,
                            gaudi::eSyncManagerInstance syncMgrInstance,
                            bool                        isMsgShort);

    bool parseMonitorArm(void*                       pPacketBuffer,
                         uint32_t                    monitorId,
                         gaudi::eSyncManagerInstance syncMgrInstance,
                         bool                        isMsgShort);

    bool parseMonitorSetup(void*                       pPacketBuffer,
                           uint32_t                    msgAddressOffset,
                           gaudi::eSyncManagerInstance syncMgrInstance,
                           bool                        isMsgShort);

    bool getSyncObjectAddressOffset(uint64_t                     fullAddress,
                                    uint32_t&                    syncObjectAddressOffset,
                                    gaudi::eSyncManagerInstance& syncMgrInstance);

    bool getMonitorAddressOffset(uint64_t                     fullAddress,
                                 uint32_t&                    monitorAddressOffset,
                                 gaudi::eSyncManagerInstance& syncMgrInstance);

    // virtual
    virtual bool        isValidPacket(uint64_t packetId) = 0;
    virtual std::string getIndentation()                 = 0;
    virtual std::string getCtrlBlockIndentation()        = 0;

    // static
    static std::string getPacketName(uint64_t packetId);

private:
    virtual uint64_t getFenceIdsPerCp() override;

    static bool _getQmanIndex(uint32_t& qmanIndex, common::eQmanType qmanType, uint32_t qmanTypeIndex);

    static bool _getFenceInfo(uint32_t& fenceId, uint32_t& cpIndex, uint32_t fenceIndex);

    static std::string _getSyncManagerDescription(eSyncManagerInstance syncMgrInstance);

    static const std::string m_packetsName[];

    static const std::array<std::string, SYNC_MNGR_LAST> m_syncManagerInstanceDesc;
};
}  // namespace gaudi