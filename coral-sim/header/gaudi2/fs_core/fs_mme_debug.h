#pragma once
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <mutex>
#include <thread>

#include "fs_assert.h"

//#define MME_DEADLOCK_DEBUG
#define MME_DEADLOCK_DEBUG_FILE_NAME_ENV_VAR "MME_DEADLOCK_DEBUG_FILE_NAME"

#ifdef MME_DEADLOCK_DEBUG

namespace gaudi3
{

namespace Mme
{

class Debug
{
   public:
    static Debug* getInstance();
    static void   terminate();

   private:
    Debug();
    ~Debug();
    void        main();
    static void dump(const std::string& filename);

    static std::mutex s_mutex;
    static Debug*     s_instance;
    static unsigned   refCtr;
    static void       handler(int s);

    bool                    m_terminate;
    bool                    m_exec;
    std::mutex              m_mutex;
    std::condition_variable m_cond;
    std::thread*            m_thread;
};
} // namespace Mme
} // namespace Gaudi2

#endif