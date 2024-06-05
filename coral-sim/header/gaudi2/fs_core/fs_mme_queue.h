#pragma once
#include <stdint.h>

#include <condition_variable>
#include <map>
#include <string>
#include <thread>

#include "fs_mme_condition_variable.h"
#include "fs_mme_debug.h"
#include "fs_mme_unit.h"

namespace Gaudi2
{
namespace Mme
{
class QueueBase : public Unit
{
   public:
#ifdef MME_DEADLOCK_DEBUG
    struct WaitInfo
    {
        QueueBase*  idleQueue;
        QueueBase** inputQueues;
        QueueBase** outputQueues;
        unsigned    numInputQueues;
        unsigned    numOutputQueues;
    };
#endif

    static const unsigned c_select_max_queues_nr = 4;

    QueueBase(FS_Mme* mme, unsigned size, const std::string& name)
        : Unit(mme, name),
          m_size(size),
          m_terminate(false),
          m_rdIdx(0),
          m_wrIdx(0),
          m_inputFlag(nullptr),
          m_outputFlag(nullptr)
    {
#ifdef MME_DEADLOCK_DEBUG
        std::lock_guard<std::mutex> lock(s_registrationMutex);
        s_instances.push_back(this);
#endif
    }

    virtual ~QueueBase() {}

    const std::string& getInstanceName() { return m_name; }

    unsigned getSize() const { return m_size; }

    bool getStatus(unsigned* availabeElements = nullptr,
                   unsigned* availabeSpace    = nullptr,
                   uint64_t* rdIdx            = nullptr,
                   uint64_t* wrIdx            = nullptr);

    void terminate(bool queue_reset = false);
    void reset(bool queue_reset = false);
    // checks the readiness of IO queues
    // returns false when at least one of the queues is terminated. otherwise true.
    static bool select(QueueBase**    inputQueues,
                       QueueBase**    outputQueues,
                       const unsigned numInputQueues,
                       const unsigned numOutputQueues,
                       unsigned*      availableInputElements = nullptr,
                       unsigned*      availableOutputSpace   = nullptr);

    bool waitUntilEmpty();
#ifdef MME_DEADLOCK_DEBUG
    static void lock() { s_activeWaitsMutex.lock(); }
    static void unlock() { s_activeWaitsMutex.unlock(); }

    static const std::map<std::thread::id, const WaitInfo*>& getActiveWaits() { return s_activeWaits; }
    static const std::vector<QueueBase*>&                    getInstances() { return s_instances; }
#endif

   protected:
    virtual void get(uint64_t rdIdx, void* pElement) = 0;

    virtual void set(uint64_t wrIdx, const void* pElement) = 0;

    bool basePop(void* pElement, unsigned* availableInputs);

    bool basePush(const void* pElement, unsigned* availableSpcae);

   private:
    struct Flag
    {
        QueueBase*              readyInstance;
        std::mutex              mutex;
        std::condition_variable cond;
    };

    const unsigned          m_size;
    bool                    m_terminate;
    uint64_t                m_rdIdx;
    uint64_t                m_wrIdx;
    std::mutex              m_mutex;
    std::condition_variable m_pushEvent;
    std::condition_variable m_popEvent;
    Flag*                   m_inputFlag;
    Flag*                   m_outputFlag;
#ifdef MME_DEADLOCK_DEBUG
    static std::map<std::thread::id, const WaitInfo*> s_activeWaits;
    static std::vector<QueueBase*>                    s_instances;
    static std::mutex                                 s_activeWaitsMutex;
    static std::mutex                                 s_registrationMutex;
#endif
};

template <typename T>
class Queue : public QueueBase
{
   public:
    Queue(const unsigned size, FS_Mme* mme = nullptr, const std::string& name = std::string())
        : QueueBase(mme, size, name), m_queue(new T[size])
    {
    }

    ~Queue() override { delete[] m_queue; }

    // pop an element from the queue.
    // returns false when the queue is terminated. otherwise true.
    bool pop(T& element, unsigned* availableInputs = nullptr) { return basePop(&element, availableInputs); }

    // push an element to the queue.
    // returns false when the queue is terminated. otherwise true.
    bool push(const T& element, unsigned* availableSpcae = nullptr) { return basePush(&element, availableSpcae); }

   protected:
    void get(uint64_t rdIdx, void* pElement) override { memcpy(pElement, &m_queue[rdIdx], sizeof(T)); }

    void set(uint64_t wrIdx, const void* pElement) override { memcpy(&m_queue[wrIdx], pElement, sizeof(T)); }

   private:
    T* const m_queue;
};
} // namespace Mme
} // namespace Gaudi2
