#include "queue_command.h"

static void prepareFieldInfosTwoDwordsHeader(BasicFieldInfoSet& basicFieldsInfoSet)
{
    BasicFieldInfoSet updatedBasicFieldSet;
    for (auto& singleBasicFieldsInfoPair : basicFieldsInfoSet)
    {
        auto copy = singleBasicFieldsInfoPair;
        // Adding two bytes for header size
        copy.first += 2;
        copy.second->setFieldIndexOffset(copy.second->getFieldIndexOffset() + 2);
        updatedBasicFieldSet.insert(copy);
    }

    basicFieldsInfoSet.clear();
    basicFieldsInfoSet.insert(updatedBasicFieldSet.begin(), updatedBasicFieldSet.end());
}

static void prepareFieldInfosTwoDwordsHeader(BasicFieldInfoSet&   basicFieldsInfoSet,
                                             AddressFieldInfoSet& addressFieldsInfoSet)
{
    BasicFieldInfoSet tempBasicSet;
    transferAddressToBasic(addressFieldsInfoSet, tempBasicSet);
    prepareFieldInfosTwoDwordsHeader(tempBasicSet);
    transferBasicToAddress(tempBasicSet, addressFieldsInfoSet);

    prepareFieldInfosTwoDwordsHeader(basicFieldsInfoSet);
}

// --------------------------------------------------------
// ------------------ QueueCommandCommon ------------------
// --------------------------------------------------------

QueueCommandCommon::QueueCommandCommon() : QueueCommand() {}

QueueCommandCommon::QueueCommandCommon(uint32_t packetType) : QueueCommand(packetType) {}

QueueCommandCommon::QueueCommandCommon(uint32_t packetType, uint64_t commandId) : QueueCommand(packetType, commandId) {}

QueueCommandCommon::~QueueCommandCommon() {}

void QueueCommandCommon::WritePB(gc_recipe::generic_packets_container* pktCon)
{
    LOG_ERR(QMAN, "{}: should not be invoked for Gaudi3", HLLOG_FUNC);
    HB_ASSERT(0, "Func should not be invoked for Gaudi3");
}

void QueueCommandCommon::WritePB(gc_recipe::generic_packets_container* pktCon, ParamsManager* params)
{
    LOG_ERR(QMAN, "{}: should not be invoked for Gaudi3", HLLOG_FUNC);
    HB_ASSERT(0, "Func should not be invoked for Gaudi3");
}

// --------------------------------------------------------
// ----------------------- NopCommon ----------------------
// --------------------------------------------------------

void NopCommon::Print() const
{
    LOG_DEBUG(QMAN, "      NOP, swtc={}", getSwtc());
}

unsigned NopCommon::GetBinarySize() const
{
    return getNopPacketSize();
}

uint64_t NopCommon::writeInstruction(void* whereTo) const
{
    uint32_t nopPacketSize = getNopPacketSize();
    memcpy(whereTo, (void*)getNopPacketAddr(), nopPacketSize);
    return nopPacketSize;
}

void NopCommon::prepareFieldInfos() {}

void NopCommon::setSwitchCQ()
{
    setSwtc(1);
}

void NopCommon::resetSwitchCQ()
{
    setSwtc(0);
}

void NopCommon::toggleSwitchCQ()
{
    setSwtc(~getSwtc());
}

bool NopCommon::isSwitchCQ() const
{
    return getSwtc();
}

// --------------------------------------------------------
// -------------------- WaitCommon ------------------------
// --------------------------------------------------------

void WaitCommon::Print() const
{
    LOG_DEBUG(GC, "      Wait # {}, increment value {}, cycles {}", getId(), getIncVal(), getNumCyclesToWait());
}

unsigned WaitCommon::GetBinarySize() const
{
    return getWaitPacketSize();
}

uint64_t WaitCommon::writeInstruction(void* whereTo) const
{
    uint32_t waitPacketSize = getWaitPacketSize();
    memcpy(whereTo, (void*)getWaitPacketAddr(), waitPacketSize);
    return waitPacketSize;
}

void WaitCommon::prepareFieldInfos() {}

// --------------------------------------------------------
// -------------------- WriteRegisterCommon ---------------
// --------------------------------------------------------

WriteRegisterCommon::WriteRegisterCommon() : QueueCommandCommon() {}

WriteRegisterCommon::WriteRegisterCommon(uint64_t commandId) : QueueCommandCommon(INVALID_PACKET_TYPE, commandId) {}

void WriteRegisterCommon::fillPackets(unsigned regOffset, unsigned value, uint32_t predicate)
{
    setValue(value);
    setPred(predicate);
    setReg(0x0);
    setRegOffset(regOffset);
    setEngBarrier(0x0);
    setMsgBarrier(0x1);
    setSwtc(0x0);
}

void WriteRegisterCommon::Print() const
{
    LOG_DEBUG(QMAN,
              "      Write 0x{:x} to register 0x{:x} reg={} pred={} swtc={}",
              getValue(),
              getRegOffset(),
              getReg(),
              getPred(),
              getSwtc());
}
void WriteRegisterCommon::prepareFieldInfos() {}

unsigned WriteRegisterCommon::GetBinarySize() const
{
    return getWriteRegisterPacketSize();
}

uint64_t WriteRegisterCommon::writeInstruction(void* whereTo) const
{
    uint32_t writeRegisterPacketSize = getWriteRegisterPacketSize();
    memcpy(whereTo, (void*)getWriteRegisterPacketAddr(), writeRegisterPacketSize);
    return writeRegisterPacketSize;
}
void WriteRegisterCommon::setSwitchCQ()
{
    setSwtc(1);
}

void WriteRegisterCommon::resetSwitchCQ()
{
    setSwtc(0);
}

void WriteRegisterCommon::toggleSwitchCQ()
{
    setSwtc(~getSwtc());
}

bool WriteRegisterCommon::isSwitchCQ() const
{
    return getSwtc();
}

// --------------------------------------------------------
// -------------------- EbPaddingCommon ---------------------
// --------------------------------------------------------

EbPaddingCommon::EbPaddingCommon() : QueueCommandCommon() {}

EbPaddingCommon::EbPaddingCommon(unsigned numPadding) : QueueCommandCommon()
{
    m_numPadding = numPadding;
}

void EbPaddingCommon::fillPackets(unsigned regOffset)
{
    setValue(0);
    setRegOffset(regOffset);
}

void EbPaddingCommon::Print() const
{
    LOG_DEBUG(QMAN,
              "      Padding before engine-barrier, filling {} WREG_32 writes to address 0x{:x}, pred={}",
              m_numPadding,
              getRegOffset(),
              getPred());
}

void EbPaddingCommon::prepareFieldInfos() {}

unsigned EbPaddingCommon::getEbPaddingNumPadding() const
{
    return m_numPadding;
}
void EbPaddingCommon::setSwitchCQ()
{
    LOG_ERR(QMAN, "Function should not be called");
}

void EbPaddingCommon::resetSwitchCQ()
{
    LOG_ERR(QMAN, "Function should not be called");
}

void EbPaddingCommon::toggleSwitchCQ()
{
    LOG_ERR(QMAN, "Function should not be called");
}

bool EbPaddingCommon::isSwitchCQ() const
{
    // EB Padding should always pad prior to other command and not post
    return false;
}

// --------------------------------------------------------
// ----------------- WriteManyRegistersCommon -------------------
// --------------------------------------------------------

void WriteManyRegistersCommon::fillPackets(unsigned        firstRegOffset,
                                           unsigned&       count32bit,
                                           uint32_t        predicate,
                                           const uint32_t* values)
{
    HB_ASSERT(count32bit > 0, "WriteManyRegisters should contain at least one register");
    unsigned totalRegs = count32bit;
    unsigned offset    = firstRegOffset;
    unsigned curReg    = 0;
    // If the first register offset isn't aligned to 8 bytes, we write it using wreg32 so next register will align
    if ((firstRegOffset & 0x7) != 0)
    {
        m_alignmentReg = createWriteRegister(offset, values[curReg], predicate);
        offset += sizeof(uint32_t);
        curReg++;
        m_incZeroOffset = false;  // If we have a wreg32 here, then patch point in offset 0 should not be updated
        m_incOffsetValue++;       // Header of wreg32
    }
    // We use bulk-write for two or more 32bit registers
    unsigned bulkSize = (totalRegs - curReg) / 2;
    if (bulkSize)
    {
        setSize(bulkSize);
        setPredicate(predicate);
        setOffset(offset);

        unsigned bulkRegs = bulkSize * 2;
        m_valuesBinary.resize(bulkSize);
        memcpy(m_valuesBinary.data(), values + curReg, sizeof(values[0]) * bulkRegs);
        curReg += bulkRegs;
        offset += sizeof(uint32_t) * bulkRegs;
        m_incOffsetValue += 2;  // Headers of WriteBulk
    }
    // If we have odd number of registers (besides the alignment register), the last one is written using wreg32
    if (curReg < totalRegs)
    {
        m_remainderReg = createWriteRegister(offset, values[curReg], predicate);
        curReg++;
    }

    HB_ASSERT(curReg == totalRegs, "Something is wrong, shouldn't have any registers left unprocessed");
}

void WriteManyRegistersCommon::Print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(QMAN)) return;

    if (m_alignmentReg != nullptr)
    {
        m_alignmentReg->Print();
    }
    if (!m_valuesBinary.empty())
    {
        uint64_t numOfDoubleRegsToPrint =
            std::min((uint64_t)GCFG_DEBUG_NUM_OF_DOUBLE_WBULK_REGS_TO_DUMP.value(), (uint64_t)m_valuesBinary.size());
        LOG_DEBUG(QMAN,
                  "      WREG_BULK {} 32bit registers to 0x{:x} pred={} swtc={}. Printing {} registers:",
                  m_valuesBinary.size() * 2,
                  getRegOffset(),
                  getWregBulkPredicate(),
                  getWregBulkSwitchCQ(),
                  numOfDoubleRegsToPrint * 2);

        uint32_t registerOffset = getRegOffset();
        for (const auto& singleValue : m_valuesBinary)
        {
            if (numOfDoubleRegsToPrint == 0)
            {
                break;
            }

            LOG_DEBUG(QMAN, "            value 0x{:x} to register 0x{:x}", singleValue & 0xFFFFFFFF, registerOffset);
            LOG_DEBUG(QMAN,
                      "            value 0x{:x} to register 0x{:x}",
                      (singleValue >> 32) & 0xFFFFFFFF,
                      registerOffset + 4);

            registerOffset += 8;
            numOfDoubleRegsToPrint--;
        }
    }
    if (m_remainderReg != nullptr)
    {
        m_remainderReg->Print();
    }
}

unsigned WriteManyRegistersCommon::GetFirstReg() const
{
    if (m_alignmentReg != nullptr)
    {
        return m_alignmentReg->getRegOffset();
    }
    else if (!m_valuesBinary.empty())
    {
        return getRegOffset();
    }
    else if (m_remainderReg != nullptr)
    {
        return m_remainderReg->getRegOffset();
    }
    else
    {
        HB_ASSERT(0, "How come I don't have even one register?");
        return 0;
    }
}

unsigned WriteManyRegistersCommon::GetCount() const
{
    unsigned numRegs = 0;
    if (m_alignmentReg != nullptr)
    {
        numRegs += 1;
    }
    if (!m_valuesBinary.empty())
    {
        numRegs += m_valuesBinary.size() * 2;
    }
    if (m_remainderReg != nullptr)
    {
        numRegs += 1;
    }
    return numRegs;
}

unsigned WriteManyRegistersCommon::GetBinarySize() const
{
    unsigned size = 0;
    if (m_alignmentReg != nullptr)
    {
        size += m_alignmentReg->GetBinarySize();
    }
    if (!m_valuesBinary.empty())
    {
        size += getPacketWregBulkSize() + m_valuesBinary.size() * sizeof(uint64_t);
    }
    if (m_remainderReg != nullptr)
    {
        size += m_remainderReg->GetBinarySize();
    }
    return size;
}

uint64_t WriteManyRegistersCommon::writeInstruction(void* whereTo) const
{
    // Impossible to perform ptr arithmetic with void*, so we use char*
    char*    whereToChar  = reinterpret_cast<char*>(whereTo);
    unsigned bytesWritten = 0;

    if (nullptr != m_alignmentReg)
    {
        bytesWritten += m_alignmentReg->writeInstruction(whereToChar + bytesWritten);
    }
    if (!m_valuesBinary.empty())
    {
        memcpy(whereToChar + bytesWritten, (void*)getBulkBinaryPacketAddr(), getWregBulkSize());
        bytesWritten += getWregBulkSize();

        unsigned valuesDataSize = sizeof(m_valuesBinary[0]) * m_valuesBinary.size();
        memcpy(whereToChar + bytesWritten, m_valuesBinary.data(), valuesDataSize);
        bytesWritten += valuesDataSize;
    }
    if (nullptr != m_remainderReg)
    {
        bytesWritten += m_remainderReg->writeInstruction(whereToChar + bytesWritten);
    }

    return bytesWritten;
}

void WriteManyRegistersCommon::prepareFieldInfos()
{
    BasicFieldInfoSet    basicSet;
    AddressFieldInfoSet& addressSet = m_addressContainerInfo.retrieveAddressFieldInfoSet();
    transferAddressToBasic(addressSet, basicSet);
    prepareFieldInfos(basicSet);
    transferBasicToAddress(basicSet, addressSet);

    prepareFieldInfos(m_addressContainerInfo.retrieveBasicFieldInfoSet());
}

void WriteManyRegistersCommon::prepareFieldInfos(BasicFieldInfoSet& basicFieldsInfoSet)
{
    /*
        We update the AFCI according to this table (the first three columns
        determine the rest).
        +-----------+--------+-----------+-----------------+------------------+
        | wreg32 #1 | wrBulk | wreg32 #2 | Advance index 0 | Value to advance |
        +-----------+--------+-----------+-----------------+------------------+
        |         0 |      0 |         1 | irrelevant      |                0 |
        |         0 |      1 |         0 | true            |                2 |
        |         0 |      1 |         1 | true            |                2 |
        |         1 |      0 |         0 | false           |                0 |
        |         1 |      0 |         1 | false           |                1 |
        |         1 |      1 |         0 | false           |                3 |
        |         1 |      1 |         1 | false           |                3 |
        +-----------+--------+-----------+-----------------+------------------+
     */
    BasicFieldInfoSet updatedBasicFieldSet;

    for (auto& singleBasicFieldsInfoPair : basicFieldsInfoSet)
    {
        auto copy = singleBasicFieldsInfoPair;
        // Update according to the offset, and update offset 0 only if m_incZeroOffset
        if (0 != copy.first || m_incZeroOffset)
        {
            copy.first += m_incOffsetValue;
            copy.second->setFieldIndexOffset(copy.second->getFieldIndexOffset() + m_incOffsetValue);
        }

        updatedBasicFieldSet.insert(copy);
    }

    basicFieldsInfoSet = std::move(updatedBasicFieldSet);
}

uint32_t WriteManyRegistersCommon::getValue(unsigned i) const
{
    if (m_alignmentReg != nullptr)
    {
        if (i == 0)
        {
            return m_alignmentReg->getValue();
        }
        i--;
    }
    if (m_valuesBinary.size() * 2 > i)
    {
        uint32_t* binaryPtr = (uint32_t*)&m_valuesBinary[i / 2];
        return binaryPtr[i % 2];
    }
    i -= m_valuesBinary.size() * 2;
    HB_ASSERT(i == 0 && m_remainderReg != nullptr, "WriteManyRegisters: Failed to get value");
    return m_remainderReg->getValue();
}

void WriteManyRegistersCommon::setValue(unsigned i, uint32_t value)
{
    if (m_alignmentReg != nullptr)
    {
        if (i == 0)
        {
            m_alignmentReg->setValue(value);
            return;
        }
        i--;
    }
    if (m_valuesBinary.size() * 2 > i)
    {
        uint32_t* binaryPtr = (uint32_t*)&m_valuesBinary[i / 2];
        binaryPtr[i % 2]    = value;
        return;
    }
    i -= m_valuesBinary.size() * 2;
    HB_ASSERT(i == 0 && m_remainderReg != nullptr, "WriteManyRegisters: Failed to set value");
    m_remainderReg->setValue(value);
}

void WriteManyRegistersCommon::addAddressPatchPoint(BasicFieldsContainerInfo& container,
                                                    uint64_t                  memId,
                                                    ptrToInt                  fieldAddress,
                                                    uint64_t                  fieldIndexOffset,
                                                    pNode                     node)
{
    // Check if we are writing to the alignment or the reminder
    // This is 64 bit resolution so when are checking the reminder we have to adjust the offset by +1 to get to the
    // correct offset in 32 bit, we adjust the count by -1 to get an index.
    if ((m_alignmentReg != nullptr && fieldIndexOffset == 0) ||
        (m_remainderReg != nullptr && fieldIndexOffset + 1 == GetCount() - 1))
    {
        // Add 2 PPs Low address for the alignment WREG, and High for the second half (WBULK or reminder WREG)
        container.addAddressEngineFieldInfo(node,
                                            getMemorySectionNameForMemoryID(memId),
                                            memId,
                                            fieldAddress.u32[0],
                                            fieldAddress.u32[1],
                                            fieldIndexOffset,
                                            fieldIndexOffset + 1,
                                            FIELD_MEMORY_TYPE_DRAM);
    }
    else
    {
        container.addAddressEngineFieldInfo(node,
                                            getMemorySectionNameForMemoryID(memId),
                                            memId,
                                            fieldAddress.u64,
                                            (uint32_t)fieldIndexOffset,
                                            FIELD_MEMORY_TYPE_DRAM);
    }
}

void WriteManyRegistersCommon::setSwitchCQ()
{
    // Set the switch on the last command so go from last command (m_remainderReg) to first (m_alignmentReg)
    if (m_remainderReg != nullptr)
    {
        m_remainderReg->setSwitchCQ();
    }
    else if (!m_valuesBinary.empty())
    {
        setWregBulkSwitchCQ(1);
    }
    else if (m_alignmentReg != nullptr)
    {
        m_alignmentReg->setSwitchCQ();
    }
}

void WriteManyRegistersCommon::resetSwitchCQ()
{
    // Reset the switch on the last command so go from last command (m_remainderReg) to first (m_alignmentReg)
    if (m_remainderReg != nullptr)
    {
        m_remainderReg->resetSwitchCQ();
    }
    else if (!m_valuesBinary.empty())
    {
        setWregBulkSwitchCQ(0);
    }
    else if (m_alignmentReg != nullptr)
    {
        m_alignmentReg->resetSwitchCQ();
    }
}

void WriteManyRegistersCommon::toggleSwitchCQ()
{
    // Toggle the switch on the last command so go from last command (m_remainderReg) to first (m_alignmentReg)
    if (m_remainderReg != nullptr)
    {
        m_remainderReg->toggleSwitchCQ();
    }
    else if (!m_valuesBinary.empty())
    {
        setWregBulkSwitchCQ(~getWregBulkSwitchCQ());
    }
    else if (m_alignmentReg != nullptr)
    {
        m_alignmentReg->toggleSwitchCQ();
    }
}

bool WriteManyRegistersCommon::isSwitchCQ() const
{
    // Get the switch from the last command so go from last command (m_remainderReg) to first (m_alignmentReg)
    if (m_remainderReg != nullptr)
    {
        return m_remainderReg->isSwitchCQ();
    }
    else if (!m_valuesBinary.empty())
    {
        return getWregBulkSwitchCQ();
    }
    else if (m_alignmentReg != nullptr)
    {
        return m_alignmentReg->isSwitchCQ();
    }
    return false;
}

// --------------------------------------------------------
// ----------------------- ArcExeWdTpcCommon --------------
// --------------------------------------------------------

void ArcExeWdTpcCommon::Print() const
{
    LOG_DEBUG(QMAN, "      ArcExeWdTpc, swtc={}", getSwtc());
}

uint64_t ArcExeWdTpcCommon::writeInstruction(void* whereTo) const
{
    uint32_t arcExeWdTpcPacketSize = GetBinarySize();
    memcpy(whereTo, getArcExeWdTpcCtxAddr(), arcExeWdTpcPacketSize);
    return arcExeWdTpcPacketSize;
}

void ArcExeWdTpcCommon::setSwitchCQ()
{
    setSwtc(1);
}

void ArcExeWdTpcCommon::resetSwitchCQ()
{
    setSwtc(0);
}

void ArcExeWdTpcCommon::toggleSwitchCQ()
{
    setSwtc(~getSwtc());
}

bool ArcExeWdTpcCommon::isSwitchCQ() const
{
    return getSwtc();
}

// --------------------------------------------------------
// ----------------------- ArcExeWdDmaCommon --------------
// --------------------------------------------------------

void ArcExeWdDmaCommon::Print() const
{
    LOG_DEBUG(QMAN, "      ArcExeWdDma, swtc={}", getSwtc());
}

uint64_t ArcExeWdDmaCommon::writeInstruction(void* whereTo) const
{
    uint32_t arcExeWdDmaPacketSize = GetBinarySize();
    memcpy(whereTo, getArcExeWdDmaCtxAddr(), arcExeWdDmaPacketSize);
    return arcExeWdDmaPacketSize;
}

void ArcExeWdDmaCommon::setSwitchCQ()
{
    setSwtc(1);
}

void ArcExeWdDmaCommon::resetSwitchCQ()
{
    setSwtc(0);
}

void ArcExeWdDmaCommon::toggleSwitchCQ()
{
    setSwtc(~getSwtc());
}

bool ArcExeWdDmaCommon::isSwitchCQ() const
{
    return getSwtc();
}

// --------------------------------------------------------
// ----------------------- ArcExeWdMmeCommon --------------
// --------------------------------------------------------

void ArcExeWdMmeCommon::Print() const
{
    LOG_DEBUG(QMAN, "      ArcExeWdMme, swtc={}", getSwtc());
}

uint64_t ArcExeWdMmeCommon::writeInstruction(void* whereTo) const
{
    uint32_t arcExeWdMmePacketSize = GetBinarySize();
    memcpy(whereTo, getArcExeWdMmeCtxAddr(), arcExeWdMmePacketSize);
    return arcExeWdMmePacketSize;
}

void ArcExeWdMmeCommon::setSwitchCQ()
{
    setSwtc(1);
}

void ArcExeWdMmeCommon::resetSwitchCQ()
{
    setSwtc(0);
}

void ArcExeWdMmeCommon::toggleSwitchCQ()
{
    setSwtc(~getSwtc());
}

bool ArcExeWdMmeCommon::isSwitchCQ() const
{
    return getSwtc();
}

// --------------------------------------------------------
// ----------------------- ArcExeWdRotCommon --------------
// --------------------------------------------------------

void ArcExeWdRotCommon::Print() const
{
    LOG_DEBUG(QMAN, "      ArcExeWdRot, swtc={}, cpl_msg_en={}", getSwtc(), getCplMsgEn());
}

uint64_t ArcExeWdRotCommon::writeInstruction(void* whereTo) const
{
    uint32_t arcExeWdRotPacketSize = GetBinarySize();
    memcpy(whereTo, getArcExeWdRotCtxAddr(), arcExeWdRotPacketSize);
    return arcExeWdRotPacketSize;
}

void ArcExeWdRotCommon::setSwitchCQ()
{
    setSwtc(1);
}

void ArcExeWdRotCommon::resetSwitchCQ()
{
    setSwtc(0);
}

void ArcExeWdRotCommon::toggleSwitchCQ()
{
    setSwtc(~getSwtc());
}

bool ArcExeWdRotCommon::isSwitchCQ() const
{
    return getSwtc();
}

// --------------------------------------------------------
// ------------------- FenceCommon ------------------------
// --------------------------------------------------------

FenceCommon::FenceCommon(unsigned int targetValue) : QueueCommandCommon(), m_targetValue(targetValue)
{
    HB_ASSERT(fitsInBits(targetValue, 14), "Fence targetValue invalid");

    // split to multiple Fence packets if targetValue > 15
    // because there are only 4 bits for dec val
    m_numPkts = div_round_up(targetValue, 0xF);
}

void FenceCommon::fillPackets(WaitID waitID, uint32_t predicate)
{
    unsigned currFenceAggValue = m_targetValue;

    for (unsigned i = 0; i < m_numPkts; ++i)
    {
        if (i != m_numPkts - 1)
        {
            setDecVal(i, 0xF);
            setTargetVal(i, currFenceAggValue);
            currFenceAggValue -= 0xF;
        }
        else
        {
            setDecVal(i, currFenceAggValue);
            setTargetVal(i, currFenceAggValue);
            currFenceAggValue = 0;
        }

        setID(i, waitID);
        setPredicate(i, predicate);
        setSwtc(i, 0x0);
    }
}

void FenceCommon::Print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(QMAN)) return;

    for (int i = 0; i < m_numPkts; i++)
    {
        LOG_DEBUG(QMAN,
                  "      Fence # {}, target value={}, dec value={}, swtc={}",
                  getIdByIndex(i),
                  getTargetValByIndex(i),
                  getDecValByIndex(i),
                  getSwtcByIndex(i));
    }
}

unsigned FenceCommon::GetBinarySize() const
{
    return m_numPkts * getFencePacketSize();
}

uint64_t FenceCommon::writeInstruction(void* whereTo) const
{
    uint32_t fencePacketSize = getFencePacketSize();
    memcpy(whereTo, (void*)getFencePacketAddr(), m_numPkts * fencePacketSize);
    return m_numPkts * fencePacketSize;
}

void FenceCommon::prepareFieldInfos() {}

void FenceCommon::setSwitchCQ()
{
    setSwtc(m_numPkts - 1, 1);  // set the switch on the last packet
}
void FenceCommon::resetSwitchCQ()
{
    setSwtc(m_numPkts - 1, 0);  // reset the switch on the last packet
}
void FenceCommon::toggleSwitchCQ()
{
    setSwtc(m_numPkts - 1, ~getSwtcByIndex(m_numPkts - 1));  // toggle the switch on the last packet
}

bool FenceCommon::isSwitchCQ() const
{
    return getSwtc();
}

// --------------------------------------------------------
// ------------------ SuspendCommon -----------------------
// --------------------------------------------------------

SuspendCommon::SuspendCommon(WaitID waitID, unsigned int waitCycles, unsigned int incrementValue)
: m_waitID(waitID), m_waitCycles(waitCycles), m_incrementValue(incrementValue)
{
}

void SuspendCommon::Print() const
{
    LOG_DEBUG(GC,
              "      Add suspension using wait and fence with ID {}, increment value {} and wait cycles {}, swtc={}",
              m_waitID,
              m_incrementValue,
              m_waitCycles,
              m_fence->getSwtc());
}

unsigned SuspendCommon::GetBinarySize() const
{
    return m_fence->GetBinarySize() + m_wait->GetBinarySize();
}

uint64_t SuspendCommon::writeInstruction(void* whereTo) const
{
    unsigned bytesWritten   = 0;
    char*    whereToCharPtr = reinterpret_cast<char*>(whereTo);

    bytesWritten += m_wait->writeInstruction(whereToCharPtr + bytesWritten);
    bytesWritten += m_fence->writeInstruction(whereToCharPtr + bytesWritten);

    return bytesWritten;
}

void SuspendCommon::prepareFieldInfos() {}

void SuspendCommon::setSwitchCQ()
{
    m_fence->setSwitchCQ();  // set the switch on the last command
}

void SuspendCommon::resetSwitchCQ()
{
    m_fence->resetSwitchCQ();  // reset the switch on the last command
}

void SuspendCommon::toggleSwitchCQ()
{
    m_fence->toggleSwitchCQ();  // toggle the switch on the last command
}

bool SuspendCommon::isSwitchCQ() const
{
    return m_fence->isSwitchCQ();  // get the switch from the last command
}

// --------------------------------------------------------
// --------------------- WriteReg64----------------------
// --------------------------------------------------------

void WriteReg64Common::fillLongBinary(unsigned baseRegIndex,
                                      uint64_t value,
                                      unsigned targetRegisterInBytes,
                                      bool     writeTargetLow,
                                      bool     writeTargetHigh,
                                      uint32_t predicate)
{
    setDwEnable((writeTargetLow ? 0x1 : 0x0) | (writeTargetHigh ? 0x2 : 0x0));
    setPred(predicate);
    setBaseIndex(baseRegIndex);
    setDregOffset(targetRegisterInBytes / sizeof(uint32_t));
    setOpcode();
    setEngBarrier(0x0);
    setMsgBarrier(0x1);
    setSwtc(0x0);
    setValue(value);
    setRel(0x1);
}

void WriteReg64Common::fillShortBinary(unsigned baseRegIndex,
                                       uint32_t value,
                                       unsigned targetRegisterInBytes,
                                       uint32_t predicate)
{
    HB_ASSERT((value & 0x80000000) == 0, "wreg64_short performs sign extension that leads to wrong value");

    setValue(value);
    setPred(predicate);
    setBaseIndex(baseRegIndex);
    setDregOffset(targetRegisterInBytes / sizeof(uint32_t));
    setOpcode();
    setEngBarrier(0x0);
    setMsgBarrier(0x1);
    setSwtc(0x0);
}

unsigned WriteReg64Common::GetBinarySize() const
{
    return getWriteReg64PacketSize();
}

uint64_t WriteReg64Common::writeInstruction(void* whereTo) const
{
    memcpy(whereTo, (void*)getWriteReg64PacketAddr(), getWriteReg64PacketSize());
    return getWriteReg64PacketSize();
}

void WriteReg64Common::prepareFieldInfos()
{
    HB_ASSERT(0 == m_addressContainerInfo.size(), "WriteReg64 command should have no patch-points.");
}

void WriteReg64Common::Print() const
{
    std::string packet("");
    std::string parts("");
    unsigned    target  = 0xFFFFFFFF;
    unsigned    baseIdx = 0xFFFFFFFF;
    uint64_t    value   = 0xFFFFFFFFFFFFFFFF;
    unsigned    pred    = 0xFFFFFFFF;
    unsigned    swtc    = 0;

    if (m_useLongBinary)
    {
        packet = std::string("wreg64_long");
        switch (getDwEnable())
        {
            case 1:
                parts = std::string("low only");
                break;
            case 2:
                parts = std::string("high only");
                break;
            case 3:
                parts = std::string("low and high");
                break;
            default:
                HB_ASSERT(0, "invalid dw_enable value");
        };
    }
    else
    {
        packet = std::string("wreg64_short");
        parts  = std::string("low and high");
    }
    target  = getDregOffset() * sizeof(uint32_t);
    baseIdx = getBaseIndex();
    value   = getValue();
    pred    = getPred();
    swtc    = getSwtc();

    LOG_DEBUG(QMAN,
              "      WriteReg64 ({}) to 0x{:x}, baseRegIndex={}, value=0x{:x}, parts={}, pred={}, swtc={}",
              packet,
              target,
              baseIdx,
              value,
              parts,
              pred,
              swtc);
}

uint64_t WriteReg64Common::getBinForTesting() const
{
    // For testing, returns the binary as 64bit integer. In case of the long packet, return only the first 64bit
    ptrToInt ret;
    ret.u64 = 0;
    if (m_useLongBinary)
    {
        ret.u32[0] = getDwEnable();
        ret.u32[1] = getCtl();
    }
    else
    {
        ret.u32[0] = getValue();
        ret.u32[1] = getCtl();
    }
    return ret.u64;
}

void WriteReg64Common::setSwitchCQ()
{
    setSwtc(1);
}

void WriteReg64Common::resetSwitchCQ()
{
    setSwtc(0);
}

void WriteReg64Common::toggleSwitchCQ()
{
    setSwtc(~getSwtc());
}

bool WriteReg64Common::isSwitchCQ() const
{
    return getSwtc();
}

// --------------------------------------------------------
// --------------- UploadKernelsAddrCommon ----------------
// --------------------------------------------------------

UploadKernelsAddrCommon::UploadKernelsAddrCommon(uint32_t uploadToLow,
                                                 uint32_t uploadToHigh,
                                                 uint32_t predicate,
                                                 uint32_t prefetchAlignmentMask)
: QueueCommandCommon(),
  m_highAddress(uploadToHigh),
  m_lowAddress(uploadToLow & prefetchAlignmentMask),  // Prefetch address must aligned to 13 bit
  m_predicate(predicate)
{
}

void UploadKernelsAddrCommon::fillPackets(uint32_t regTpcCmd)
{
    // Will be patching as BasicFieldsContainerInfo in the following order: (low, high)
    for (int i = 0; i < m_numOfPackets; ++i)
    {
        setPredicate(i, m_predicate);
        setSwtc(i, 0);
    }

    setRegOffset(0, getAddrOfTpcBlockField("icache_base_adderess_low"));
    setValue(0, m_lowAddress);

    setRegOffset(1, getAddrOfTpcBlockField("icache_base_adderess_high"));
    setValue(1, m_highAddress);

    setRegOffset(2, getAddrOfTpcBlockField("tpc_cmd"));
    setValue(2, regTpcCmd);
}

void UploadKernelsAddrCommon::Print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(QMAN)) return;

    ptrToInt p;
    p.u32[0] = m_lowAddress;
    p.u32[1] = m_highAddress;
    LOG_DEBUG(QMAN, "      Prefetch icache at base address 0x{:x}, swtc={}", p.u64, getSwtcByIndex(m_numOfPackets - 1));
}

uint64_t UploadKernelsAddrCommon::writeInstruction(void* whereTo) const
{
    memcpy(whereTo, getUploadKernelPacketAddr(), GetBinarySize());
    return GetBinarySize();
}

void UploadKernelsAddrCommon::prepareFieldInfos()
{
    if (0 == m_addressContainerInfo.size())
    {
        return;
    }

    HB_ASSERT(2 == m_addressContainerInfo.size(), "Unexpected number of patching points for UploadKernelAddr command");
    BasicFieldInfoSet    basicSet;
    AddressFieldInfoSet& addressSet = m_addressContainerInfo.retrieveAddressFieldInfoSet();
    transferAddressToBasic(addressSet, basicSet);
    prepareFieldInfos(basicSet);
    transferBasicToAddress(basicSet, addressSet);

    prepareFieldInfos(m_addressContainerInfo.retrieveBasicFieldInfoSet());
}

void UploadKernelsAddrCommon::prepareFieldInfos(BasicFieldInfoSet& basicFieldsInfoSet)
{
    BasicFieldInfoSet updatedBasicFieldSet;
    for (auto& singleBasicFieldsInfoPair : basicFieldsInfoSet)
    {
        auto copy = singleBasicFieldsInfoPair;
        if (copy.second->getFieldIndexOffset() == 1)
        {
            // Accounting for msg_short header
            copy.first = 2;
            copy.second->setFieldIndexOffset(2);
        }
        else if (copy.second->getFieldIndexOffset() != 0)
        {
            HB_ASSERT(false, "Unexpected field index offset in UploadKernelAddr command");
        }

        updatedBasicFieldSet.insert(copy);
    }

    basicFieldsInfoSet.clear();
    basicFieldsInfoSet.insert(updatedBasicFieldSet.begin(), updatedBasicFieldSet.end());
}

void UploadKernelsAddrCommon::setSwitchCQ()
{
    setSwtc(m_numOfPackets - 1, 1);  // set the switch on the last packet
}

void UploadKernelsAddrCommon::resetSwitchCQ()
{
    setSwtc(m_numOfPackets - 1, 0);  // reset the switch on the last packet
}

void UploadKernelsAddrCommon::toggleSwitchCQ()
{
    setSwtc(m_numOfPackets - 1, ~getSwtcByIndex(m_numOfPackets - 1));  // toggle the switch on the last packet
}

bool UploadKernelsAddrCommon::isSwitchCQ() const
{
    return getSwtcByIndex(m_numOfPackets - 1);  // get the switch from the last packet
}

// --------------------------------------------------------
// -------------------- MsgLongCommon ---------------------
// --------------------------------------------------------

unsigned MsgLongCommon::GetBinarySize() const
{
    return getMsgLongPacketSize();
}

uint64_t MsgLongCommon::writeInstruction(void* whereTo) const
{
    uint32_t msgLongPacketSize = getMsgLongPacketSize();
    memcpy(whereTo, (void*)getMsgLongPacketAddr(), msgLongPacketSize);
    return msgLongPacketSize;
}

void MsgLongCommon::prepareFieldInfos()
{
    HB_ASSERT(2 == m_addressContainerInfo.size(), "Unexpected number of patching points for MsgLong command");
    prepareFieldInfosTwoDwordsHeader(m_addressContainerInfo.retrieveBasicFieldInfoSet(),
                                     m_addressContainerInfo.retrieveAddressFieldInfoSet());
}

void MsgLongCommon::setSwitchCQ()
{
    setSwtc(1);
}

void MsgLongCommon::resetSwitchCQ()
{
    setSwtc(0);
}

void MsgLongCommon::toggleSwitchCQ()
{
    setSwtc(~getSwtc());
}

bool MsgLongCommon::isSwitchCQ() const
{
    return getSwtc();
}

// --------------------------------------------------------
// --------------------- QmanDelay ------------------------
// --------------------------------------------------------

// Add QmanDelay command as a WA for a HW issue - H6-3262 (https://jira.habana-labs.com/browse/H6-3262)
// To avoid race between updating regs in cache to read them using wreg64

void QmanDelayCommon::Print() const
{
    LOG_DEBUG(QMAN, "      QmanDelay (blocking command using fence with ID_2), swtc={}", m_fence->getSwtc());
}

unsigned QmanDelayCommon::GetBinarySize() const
{
    return m_fence->GetBinarySize() + m_wreg->GetBinarySize();
}

uint64_t QmanDelayCommon::writeInstruction(void* whereTo) const
{
    unsigned bytesWritten   = 0;
    char*    whereToCharPtr = reinterpret_cast<char*>(whereTo);

    bytesWritten += m_wreg->writeInstruction(whereToCharPtr + bytesWritten);
    bytesWritten += m_fence->writeInstruction(whereToCharPtr + bytesWritten);

    return bytesWritten;
}

void QmanDelayCommon::prepareFieldInfos() {}

void QmanDelayCommon::setSwitchCQ()
{
    m_fence->setSwitchCQ();  // set the switch on the last command
}

void QmanDelayCommon::resetSwitchCQ()
{
    m_fence->resetSwitchCQ();  // reset the switch on the last command
}

void QmanDelayCommon::toggleSwitchCQ()
{
    m_fence->toggleSwitchCQ();  // toggle the switch on the last command
}

bool QmanDelayCommon::isSwitchCQ() const
{
    return m_fence->isSwitchCQ();  // get the switch from the last command
}

// --------------------------------------------------------
// ----------------------- ResetSobs ----------------------
// --------------------------------------------------------

ResetSobs::ResetSobs(unsigned target, unsigned totalNumEngs, unsigned targetXps)
: m_switchBit(false), m_target(target), m_totalNumEngs(totalNumEngs), m_targetXps(targetXps)
{
    m_isDynamic = true;
}

void ResetSobs::Print() const
{
    if (getTarget() == 0 && getTargetXps() == 0)
    {
        LOG_DEBUG(QMAN, "      ResetSobs(virtual, canceled), swtc={}", getSwtc());
    }
    else
    {
        std::string targetXps("");
        if (getTargetXps() != 0) targetXps = std::string(", targetXps=") + std::to_string(getTargetXps());
        LOG_DEBUG(QMAN,
                "      ResetSobs(virtual), swtc={}, target={}, totalNumEngs={}{}",
                getSwtc(),
                getTarget(),
                getTotalNumEngs(),
                targetXps);
    }
}

unsigned ResetSobs::GetBinarySize() const
{
    return 0;  // virtual command has no binary size from the program perspective
}

uint64_t ResetSobs::writeInstruction(void* whereTo) const
{
    return 0;  // virtual command has no binary size from the program perspective
}

void ResetSobs::setSwitchCQ()
{
    setSwtc(true);
}

void ResetSobs::resetSwitchCQ()
{
    setSwtc(false);
}

void ResetSobs::toggleSwitchCQ()
{
    setSwtc(!getSwtc());
}

bool ResetSobs::isSwitchCQ() const
{
    return getSwtc();
}

bool ResetSobs::getSwtc() const
{
    return m_switchBit;
}

void ResetSobs::setSwtc(bool val)
{
    m_switchBit = val;
}
