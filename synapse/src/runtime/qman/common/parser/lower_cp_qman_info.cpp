#include "lower_cp_qman_info.hpp"

#include <sstream>

using namespace common;

void LowerCpQmanInfo::reset(uint64_t cpDmaPacketIndex)
{
    m_cpDmaPacketIndex = cpDmaPacketIndex;
    QmanCpInfo::reset(true);
}

void LowerCpQmanInfo::printStartParsing()
{
    QmanCpInfo::printStartParsing(getIndentation() + "   ==> Parsing Lower-CP");
}

std::string LowerCpQmanInfo::getIndentation()
{
    return std::string(INDENTATION_SIZE, ' ');
}

std::string LowerCpQmanInfo::getCtrlBlockIndentation()
{
    return std::string(CONTROL_BLOCK_INDENTATION_SIZE, ' ');
};

std::string LowerCpQmanInfo::getPacketIndexDesc() const
{
    std::stringstream descriptionStream;
    descriptionStream << m_cpDmaPacketIndex << "." << m_packetIndex;

    return descriptionStream.str();
}