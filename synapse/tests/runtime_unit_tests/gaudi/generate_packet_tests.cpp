#include <stdint.h>
#include <vector>
#include <iostream>
#include <sstream>
#include "runtime/qman/gaudi/generate_packet.hpp"
#include "synapse_test.hpp"
#include "gaudi/gaudi_packets.h"

using namespace std;

using namespace gaudi;
using BasePacket = CommonPktGen::BasePacket;

const uint64_t ANY_DATA_64   = std::numeric_limits<uint64_t>::max();
const uint32_t ANY_DATA_32   = std::numeric_limits<uint32_t>::max();
const uint16_t ANY_DATA_16   = std::numeric_limits<uint16_t>::max();
const uint8_t  ANY_DATA_8    = std::numeric_limits<uint8_t>::max();
const bool     ANY_DATA_BOOL = true;

class SynGaudiPacketGen : public ::testing::Test
{
public:
    SynGaudiPacketGen() = default;

    virtual ~SynGaudiPacketGen() = default;
};
TEST_F(SynGaudiPacketGen, wreg32)
{
    BasePacket* pkt  = new GenWreg32(ANY_DATA_32, ANY_DATA_32, ANY_DATA_32, ANY_DATA_32);
    char*       buf  = new char[pkt->getPacketSize()];
    char*       head = buf;

    pkt->generatePacket(buf);
    ASSERT_NE(buf, head) << "Pointer was not promoted";

    packet_wreg32* res_cast = (packet_wreg32*)head;
    ASSERT_EQ(res_cast->opcode, PACKET_WREG_32) << "Packet wrong opcode";

    delete[] head;
    delete pkt;
}

TEST_F(SynGaudiPacketGen, wregbulk)
{
    const uint32_t BUFF_SIZE = 0xff;

    uint64_t* valuesArr = new uint64_t[BUFF_SIZE];
    for (uint32_t i = 0; i < BUFF_SIZE; i++)
    {
        valuesArr[i] = 0xff;
    }
    BasePacket* pkt  = new GenWregBulk(BUFF_SIZE, ANY_DATA_32, ANY_DATA_32, ANY_DATA_32, valuesArr);
    char*       buf  = new char[pkt->getPacketSize()];
    char*       head = buf;

    pkt->generatePacket(buf);
    ASSERT_NE(buf, head) << "Pointer was not promoted";

    packet_wreg_bulk* res_cast = (packet_wreg_bulk*)head;
    ASSERT_EQ(res_cast->opcode, PACKET_WREG_BULK) << "Packet wrong opcode";

    delete[] head;
    delete pkt;
    delete[] valuesArr;
}

TEST_F(SynGaudiPacketGen, msgLong)
{
    BasePacket* pkt  = new GenMsgLong(ANY_DATA_32, ANY_DATA_32, ANY_DATA_32, ANY_DATA_32, ANY_DATA_32, ANY_DATA_64);
    char*       buf  = new char[pkt->getPacketSize()];
    char*       head = buf;

    pkt->generatePacket(buf);
    ASSERT_NE(buf, head) << "Pointer was not promoted";

    packet_msg_long* res_cast = (packet_msg_long*)head;
    ASSERT_EQ(res_cast->opcode, PACKET_MSG_LONG) << "Packet wrong opcode";

    delete[] head;
    delete pkt;
}

TEST_F(SynGaudiPacketGen, msgShort)
{
    BasePacket* pkt =
        new GenMsgShort(ANY_DATA_32, ANY_DATA_32, ANY_DATA_32, ANY_DATA_32, ANY_DATA_32, ANY_DATA_32, ANY_DATA_32);
    char* buf  = new char[pkt->getPacketSize()];
    char* head = buf;

    pkt->generatePacket(buf);
    ASSERT_NE(buf, head) << "Pointer was not promoted";

    packet_msg_short* res_cast = (packet_msg_short*)head;
    ASSERT_EQ(res_cast->opcode, PACKET_MSG_SHORT) << "Packet wrong opcode";

    delete[] head;
    delete pkt;
}

TEST_F(SynGaudiPacketGen, msgProt)
{
    BasePacket* pkt  = new GenMsgProt(ANY_DATA_32, ANY_DATA_32, ANY_DATA_32, ANY_DATA_32, ANY_DATA_32);
    char*       buf  = new char[pkt->getPacketSize()];
    char*       head = buf;

    pkt->generatePacket(buf);
    ASSERT_NE(buf, head) << "Pointer was not promoted";

    packet_msg_prot* res_cast = (packet_msg_prot*)head;
    ASSERT_EQ(res_cast->opcode, PACKET_MSG_PROT) << "Packet wrong opcode";

    delete[] head;
    delete pkt;
}

TEST_F(SynGaudiPacketGen, fence)
{
    BasePacket* pkt  = new GenFence(ANY_DATA_32, ANY_DATA_32, ANY_DATA_32, ANY_DATA_32, ANY_DATA_32, ANY_DATA_32);
    char*       buf  = new char[pkt->getPacketSize()];
    char*       head = buf;

    pkt->generatePacket(buf);
    ASSERT_NE(buf, head) << "Pointer was not promoted";

    packet_fence* res_cast = (packet_fence*)head;
    ASSERT_EQ(res_cast->opcode, PACKET_FENCE) << "Packet wrong opcode";

    delete[] head;
    delete pkt;
}

TEST_F(SynGaudiPacketGen, linDma)
{
    BasePacket* pkt  = new GenLinDma(ANY_DATA_32,
                                    ANY_DATA_32,
                                    ANY_DATA_32,
                                    ANY_DATA_32,
                                    ANY_DATA_64,
                                    ANY_DATA_64,
                                    ANY_DATA_8,
                                    ANY_DATA_8,
                                    ANY_DATA_BOOL,
                                    ANY_DATA_BOOL,
                                    ANY_DATA_BOOL,
                                    ANY_DATA_BOOL,
                                    ANY_DATA_BOOL,
                                    ANY_DATA_BOOL);
    char*       buf  = new char[pkt->getPacketSize()];
    char*       head = buf;

    pkt->generatePacket(buf);
    ASSERT_NE(buf, head) << "Pointer was not promoted";

    packet_lin_dma* res_cast = (packet_lin_dma*)head;
    ASSERT_EQ(res_cast->opcode, PACKET_LIN_DMA) << "Packet wrong opcode";

    delete[] head;
    delete pkt;
}

TEST_F(SynGaudiPacketGen, nop)
{
    BasePacket* pkt  = new GenNop(ANY_DATA_32, ANY_DATA_32, ANY_DATA_32);
    char*       buf  = new char[pkt->getPacketSize()];
    char*       head = buf;

    pkt->generatePacket(buf);
    ASSERT_NE(buf, head) << "Pointer was not promoted";

    packet_nop* res_cast = (packet_nop*)head;
    ASSERT_EQ(res_cast->opcode, PACKET_NOP) << "Packet wrong opcode";

    delete[] head;
    delete pkt;
}

TEST_F(SynGaudiPacketGen, stop)
{
    BasePacket* pkt  = new GenStop(ANY_DATA_32);
    char*       buf  = new char[pkt->getPacketSize()];
    char*       head = buf;

    pkt->generatePacket(buf);
    ASSERT_NE(buf, head) << "Pointer was not promoted";

    packet_stop* res_cast = (packet_stop*)head;
    ASSERT_EQ(res_cast->opcode, PACKET_STOP) << "Packet wrong opcode";

    delete[] head;
    delete pkt;
}

TEST_F(SynGaudiPacketGen, cp_dma)
{
    BasePacket* pkt  = new GenCpDma(ANY_DATA_32, ANY_DATA_32, ANY_DATA_32, ANY_DATA_64);
    char*       buf  = new char[pkt->getPacketSize()];
    char*       head = buf;

    pkt->generatePacket(buf);
    ASSERT_NE(buf, head) << "Pointer was not promoted";

    packet_cp_dma* res_cast = (packet_cp_dma*)head;
    ASSERT_EQ(res_cast->opcode, PACKET_CP_DMA) << "Packet wrong opcode";

    delete[] head;
    delete pkt;
}

TEST_F(SynGaudiPacketGen, arb_point)
{
    BasePacket* pkt  = new GenArbitrationPoint(ANY_DATA_8, ANY_DATA_BOOL);
    char*       buf  = new char[pkt->getPacketSize()];
    char*       head = buf;

    pkt->generatePacket(buf);
    ASSERT_NE(buf, head) << "Pointer was not promoted";

    packet_arb_point* res_cast = (packet_arb_point*)head;
    ASSERT_EQ(res_cast->opcode, PACKET_ARB_POINT) << "Packet wrong opcode";

    delete[] head;
    delete pkt;
}

TEST_F(SynGaudiPacketGen, repeat)
{
    BasePacket* pkt  = new GenRepeat(ANY_DATA_BOOL, ANY_DATA_BOOL, ANY_DATA_16, ANY_DATA_16, ANY_DATA_32);
    char*       buf  = new char[pkt->getPacketSize()];
    char*       head = buf;

    pkt->generatePacket(buf);
    ASSERT_NE(buf, head) << "Pointer was not promoted";

    packet_repeat* res_cast = (packet_repeat*)head;
    ASSERT_EQ(res_cast->opcode, PACKET_REPEAT) << "Packet wrong opcode";

    delete[] head;
    delete pkt;
}

TEST_F(SynGaudiPacketGen, wait)
{
    BasePacket* pkt  = new GenWait(ANY_DATA_BOOL, ANY_DATA_BOOL, ANY_DATA_32, ANY_DATA_32, ANY_DATA_32);
    char*       buf  = new char[pkt->getPacketSize()];
    char*       head = buf;

    pkt->generatePacket(buf);
    ASSERT_NE(buf, head) << "Pointer was not promoted";

    packet_wait* res_cast = (packet_wait*)head;
    ASSERT_EQ(res_cast->opcode, PACKET_WAIT) << "Packet wrong opcode";

    delete[] head;
    delete pkt;
}

TEST_F(SynGaudiPacketGen, load_and_execute)
{
    BasePacket* pkt  = new GenLoadAndExecute(ANY_DATA_BOOL,
                                            ANY_DATA_BOOL,
                                            ANY_DATA_BOOL,
                                            ANY_DATA_BOOL,
                                            ANY_DATA_32,
                                            ANY_DATA_32,
                                            ANY_DATA_64);
    char*       buf  = new char[pkt->getPacketSize()];
    char*       head = buf;

    pkt->generatePacket(buf);
    ASSERT_NE(buf, head) << "Pointer was not promoted";

    packet_load_and_exe* res_cast = (packet_load_and_exe*)head;
    ASSERT_EQ(res_cast->opcode, PACKET_LOAD_AND_EXE) << "Packet wrong opcode";

    delete[] head;
    delete pkt;
}
