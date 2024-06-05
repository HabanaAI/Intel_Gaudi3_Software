#include "lower_cp_qman_info.hpp"

#include "gaudi/gaudi_packets.h"

using namespace gaudi;

bool LowerCpQmanInfo::isValidPacket(uint64_t packetId)
{
    return ((packetId != PACKET_CP_DMA) && (packetId != PACKET_ARB_POINT));
}