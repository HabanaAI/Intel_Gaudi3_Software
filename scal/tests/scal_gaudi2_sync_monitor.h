#pragma once

#define varoffsetof(t, m) ((size_t)(&(((t*)0)->m)))

#define ARC_LBU_BASE_ADDR                0xF0000000
#define ARC_LBU_ADDR(reg)                (ARC_LBU_BASE_ADDR + reg)
#define NUM_OF_DWORDS                    4
#define ARC_ACC_ENGS_VIRTUAL_ADDR_OFFSET 0x73c


int gaudi2_configMonitorForLongSO(uint64_t smBase, scal_core_handle_t coreHandle, unsigned monitorID, unsigned longSoID, unsigned fenceID, bool compareEQ,
                                  unsigned syncObjIdForDummyPayload, uint64_t smBaseForDummyPayload,
                                  uint64_t addr[10], uint32_t value[10]);

int gaudi2_armMonitorForLongSO(uint64_t smBase, unsigned monitorID, unsigned longSoID, uint64_t longSOtargetValue, uint64_t prevLongSOtargetValue, bool compareEQ,
                        uint64_t addr[7], uint32_t value[7], unsigned &numPayloads);

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
uint32_t gaudi2_qman_add_nop_pkt(void* buffer, uint32_t buf_off, enum QMAN_EB eb, enum QMAN_MB mb);

uint32_t gaudi2_getAcpRegAddr(unsigned dcoreID);

uint32_t gaudi2_createPayload(uint32_t fence_id);
