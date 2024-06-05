#pragma once
#include <stdint.h>

#include <mutex>

class FS_Pcap
{
   public:
    FS_Pcap();
    ~FS_Pcap();

    void dump(const uint8_t* buffer, unsigned len);

   private:
    void*      m_pcap;
    void*      m_pcapFile;
    std::mutex m_pcapMutex;
};