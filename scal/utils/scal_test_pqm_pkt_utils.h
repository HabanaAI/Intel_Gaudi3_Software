#ifndef _SCAL_TEST_PQM_PKT_UTILS_H_
#define _SCAL_TEST_PQM_PKT_UTILS_H_

#include <string>
#include "scal_internal/pkt_macros.hpp"

class PqmPktUtils{
    public:
        enum pdmaDir
        {
            DEVICE_TO_HOST,
            HOST_TO_DEVICE,
            INVALID
        };

        static void sendPdmaCommand(bool       isDirectMode,
                                    std::variant<G2Packets , G3Packets> buildPkt,
                                    char*       pktBuffer,
                                    uint64_t    src,
                                    uint64_t    dst,
                                    uint32_t    size,
                                    uint8_t     engineGroupType,
                                    int32_t     workloadType,
                                    uint8_t     ctxId,
                                    uint32_t    payload,
                                    uint64_t    payloadAddr,
                                    bool        bMemset,
                                    uint32_t    signalToCg,
                                    bool        wr_comp,
                                    uint32_t    completionGroupIndex,
                                    uint64_t    longSoSmIdx,
                                    unsigned    longSoIndex);

        static uint64_t getPdmaCmdSize( bool isDirectMode,
                                        std::variant<G2Packets , G3Packets> buildPkt,
                                        bool     wr_comp,
                                        unsigned paramsCount);

        static uint64_t getFenceCmdSize();

        static uint64_t getMsgLongCmdSize();

        static void buildPqmFenceCmd(uint8_t* pktBuffer,
                                     uint8_t  fenceId,
                                     uint32_t fenceDecVal,
                                     uint32_t fenceTarget);

        static uint32_t getPayloadDataFenceInc(FenceIdType fenceId);

        static void buildPqmMsgLong(void*    pktBuffer,
                                    uint32_t val,
                                    uint64_t address);

        static void buildNopCmd(void* pktBuffer);

        static uint64_t getNopCmdSize();


    private:
        static uint32_t getCqLongSoValue();

        static void buildPqmLpdmaPacket(uint8_t* pktBuffer,
                                        uint64_t src,
                                        uint64_t dst,
                                        uint32_t size,
                                        bool     bMemset,
                                        uint8_t  direction,
                                        bool     useBarrier,
                                        bool     signalToCg,
                                        uint64_t barrierAddress,
                                        uint32_t barrierData,
                                        uint32_t fenceDecVal,
                                        uint32_t fenceTarget,
                                        uint32_t fenceId);

        static uint64_t getLpdmaCmdSize(bool useBarrier);

};


#endif