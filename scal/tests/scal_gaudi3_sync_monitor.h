#pragma once

#define ARC_LBU_BASE_ADDR                0xF0000000
#define ARC_LBU_ADDR(reg)                (ARC_LBU_BASE_ADDR + reg)
#define NUM_OF_DWORDS                    4
#define ARC_ACC_ENGS_VIRTUAL_ADDR_OFFSET 0x73c

int gaudi3_configMonitorForLongSO(uint64_t           smBase,
                                  scal_core_handle_t coreHandle,
                                  unsigned           monitorID,
                                  unsigned           longSoID,
                                  bool               compareEQ,
                                  uint64_t           payloadAddr,
                                  uint32_t           payloadData,
                                  uint64_t           addr[7],
                                  uint32_t           value[7]);

int gaudi3_armMonitorForLongSO(uint64_t smBase, unsigned monitorID, unsigned longSoID, uint64_t longSOtargetValue, uint64_t prevLongSOtargetValue, bool compareEQ, uint64_t addr[7],
                                              uint32_t value[7], unsigned &numPayloads);

int32_t gaudi3_getOffsetVal(uint32_t offset);

#ifndef QMAN_EB_MB_DEFINED
#define QMAN_EB_MB_DEFINED
enum QMAN_EB
{
    EB_FALSE = 0,
    EB_TRUE
};

enum QMAN_MB
{
    MB_FALSE = 0,
    MB_TRUE
};
#endif

uint32_t gaudi3_qman_add_nop_pkt(void* buffer, uint32_t buf_off, enum QMAN_EB eb, enum QMAN_MB mb);

uint32_t gaudi3_getAcpRegAddr(unsigned dcoreID);

uint32_t gaudi3_createPayload(uint32_t fence_id);

void gaudi3_getAcpPayload(uint64_t& payloadAddress, uint32_t& payloadData, uint32_t fence_id, uint64_t acpEngBaseAddr);