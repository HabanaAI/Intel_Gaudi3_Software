#pragma once

//**********************************************************************
//   This file includes a class (Statistics) that should help when you need to collect statistics.
//   The class assumes all collection points are known during compilation. If it is not the case
//   you can use StatisticsVec class (see below), which has a little bigger performance penalty
//   1) Define the point to collect as an enum. For example: enum StatPoints {p1, p2, p3};
//   2) Define an array with the description of each enum. For example:
//       static constexpr auto enumNamePoints = toStatArray<StatPoints>(
//    {
//        {p1, "description p1"},
//        {p2, "description p2"},
//        {p3, "description p3"}
//    });
//
//   3) add a member in your class: "Statistics<StatPoints, enumNamePoints.size()> m_stat;"
//   4) In the constructor of your class, construct m_stat. You need to provide:
//      Name, the array you created before, frequency (how often to push it to trace. 0 means only on destruct), enable
//          For example: m_stat("my Name", enumNamePoints, 1, true);
//      NOTE: to enable the statistics both enable bit should be set and also one of the GCFG:
//      GCFG_ENABLE_STATS || GCFG_ENABLE_STATS_TBL || GCFG_STATS_PER_SYNLAUNCH || GCFG_STATS_FREQ

//   5) Add collection where you need. For example, if p1 collects size of something, use
//          For example: m_stat.collectSum(p1, sizeToCollect);
//   6) If you want to add time, do:
//          For example: TimeTools::StdTime start = TimeTools::timeNow();
//                       ....Some code to measure....
//                       m_stat.collectSum(p2, TimeTools::timeFrom(start)
//************************************************************************
#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include "infra/containers/to_array.hpp"
#include "timer.h"

template<class T>
struct StatEnumMsg
{
    T           pointEnum;
    const char* pointName;
};

template <class T, std::size_t N>
constexpr std::array<StatEnumMsg<T>, N> toStatArray(StatEnumMsg<T> (&&a)[N])
{
    return cpp20::detail::to_array_impl<StatEnumMsg<T>>(std::move(a), std::make_index_sequence<N>{});
}

class StatisticsBase
{
public:
    struct GetData
    {
        uint64_t    count;
        uint64_t    sum;
        const char* name;
    };

    void flush();
    void setEnableState(bool enable);

    struct CollectedData
    {
        std::atomic<uint64_t> sum;
        std::atomic<uint64_t> count;
    };

    bool isEnabled() { return m_enabled; }
protected:
    StatisticsBase(std::string_view        statName,
                   uint32_t                dumpFreq,
                   bool                    enable,
                   int                     maxEnum,
                   const StatEnumMsg<int>* statEnumMsg,
                   CollectedData*          collectedData,
                   uint16_t                maxMsgLength)
    : m_statName(statName),
      m_dumpFreq(dumpFreq),
      m_maxEnum(maxEnum),
      m_statEnumMsg(statEnumMsg),
      m_collectedData(collectedData),
      m_maxMsgLength(maxMsgLength)
    {
        if (enable)
        {
            updateEnable();
        }
    }

    StatisticsBase(const StatisticsBase& other)
    {
        m_statName      = other.m_statName + " Cloned";
        m_dumpFreq      = other.m_dumpFreq;
        m_enabled       = other.m_enabled;
        m_isFlushed     = other.m_isFlushed;
        m_headerPrinted = other.m_headerPrinted;
        m_isTbl         = other.m_isTbl;
        m_maxEnum       = other.m_maxEnum;
        m_statEnumMsg   = other.m_statEnumMsg;
    }

    StatisticsBase& operator=(const StatisticsBase& other) = delete;

    virtual ~StatisticsBase();

    inline void collect(int point, uint64_t sum)
    {
        m_collectedData[point].sum += sum;
        m_collectedData[point].count++;

        uint64_t cnt = m_collectedData[point].count;

        if((point == 0)              &&
           (m_dumpFreq != 0)         &&
           ((cnt % m_dumpFreq) == 0) &&
           (cnt > 0))
        {
            printToLog(" periodic " + std::to_string(cnt), false/*dump all*/, true/*clear*/);
        }
    }

    void setPtrs(StatEnumMsg<int>* ptrMsg, CollectedData* ptrData) { m_statEnumMsg = ptrMsg; m_collectedData = ptrData; }
    GetData get(int point) { return {m_collectedData[point].count.load(),
                                    m_collectedData[point].sum.load(),
                                           m_statEnumMsg[point].pointName}; }

    void printToLog(const std::string& rMsg = "", bool dumpAll = false, bool clear = true);

    void updateEnable();

    void clearAll();

    void outputHeader();

    std::string             m_statName;
    uint32_t                m_dumpFreq      = 0;
    bool                    m_enabled       = false;
    bool                    m_isFlushed     = false;
    bool                    m_headerPrinted = false;
    bool                    m_isTbl         = false;
    int                     m_maxEnum       = 0;
    const StatEnumMsg<int>* m_statEnumMsg   = nullptr;
    CollectedData*          m_collectedData = nullptr;
    const uint16_t          m_maxMsgLength  = 0;

    static constexpr char   m_grep[] = "zzz~"; // something to grep and cut on
};

// W/A init order issue:
// If m_db is a member of the Statistics class, it's UB to pass it's .data() to the StatisticsBase c'tor since it
// technically wasn't initialized yet. The W/A is to inherit from this struct first so that it's guaranteed to be
// initialized.
template<size_t N>
struct StatisticsArrayContainer
{
public:
    std::array<StatisticsBase::CollectedData, N> m_db {};  // This is the data-base for the counters
};

template<auto& arr>
class Statistics
: public StatisticsArrayContainer<arr.size()>
, public StatisticsBase
{
    using Container = StatisticsArrayContainer<arr.size()>;
    using EnumT = decltype(arr[0].pointEnum);

public:
    Statistics(std::string_view statName, uint32_t dumpFreq, bool enable)
    : StatisticsBase(statName,
                     dumpFreq,
                     enable,
                     arr.size(),
                     reinterpret_cast<const StatEnumMsg<int>*>(arr.data()),
                     Container::m_db.data(),
                     maxMsgLength())
    {
    }

    Statistics(const Statistics& src) : StatisticsBase(src)
    {
        // Do not copy m_db, start with zero
        StatisticsBase::m_collectedData = Container::m_db.data();  // point the base class to the data
    }

    virtual ~Statistics() override = default;

    inline void collect(EnumT point, uint64_t sum)
    {
        if (!m_enabled) return;
        StatisticsBase::collect((int)point, sum);
    }

    inline void collectTime(EnumT point, StatTimeStart time)
    {
        if (m_enabled)
        {
            StatisticsBase::collect((int)point, TimeTools::timeFromNs(time.startTime));
        }
    }

    GetData get(EnumT point) { return StatisticsBase::get((int)point); }

private:
    static constexpr size_t NotInOrder()
    {
        for (std::size_t i = 0; i < arr.size(); i++)
        {
            if ((int)arr[i].pointEnum != i)
            {
                return i;
            }
        }
        return arr.size();
    }

    static constexpr uint16_t maxMsgLength()
    {
        uint16_t maxLength = 0;

        for (int i = 0; i < arr.size(); i++)
        {
            uint16_t l  = std::string_view(arr[i].pointName).size();
            maxLength = (l > maxLength) ? l : maxLength;
        }
        return maxLength;
    }

    // We cast the enum to int for the base class to avoid code duplication per enum in the base class,
    // for that we assume the enum size is the same as int. Verify it here.
    static_assert(sizeof(EnumT) == sizeof(int));

    // Make sure that all the enums are represented it the array (the array is enum->message pairs).
    // For that the size of the array should be the same of the enum
    static_assert((size_t)EnumT::LAST == arr.size());

    // And verify all the enum->message pairs are in the order of the enums
    static_assert(NotInOrder() == arr.size(), "The stat arr given is not in order of the enums");
};

//////////////////////////////////////////////////////////
// The class belows supoort statistics when the collection points
// are unknow during compilation
/////////////////////////////////////////////////////////
class StatisticsVec : public StatisticsBase
{
    public:
        StatisticsVec(std::string_view statName, std::vector<StatEnumMsg<int>> points, uint32_t dumpFreq, bool enable)
        : StatisticsBase(statName, dumpFreq, enable, points.size(), points.data(), nullptr, 55), m_points(points)
        {
            m_db = std::unique_ptr<CollectedData[]>(new CollectedData[points.size()]{});
            StatisticsBase::setPtrs(m_points.data(), m_db.get());
        }

         virtual ~StatisticsVec() override { StatisticsBase::flush(); };

        inline void collect(int point, uint64_t sum)
        {
            if (!m_enabled) return;
            StatisticsBase::collect(point, sum);
        }

        GetData get(int point) { return StatisticsBase::get(point); }

private:
    std::vector<StatEnumMsg<int>>    m_points;
    std::unique_ptr<CollectedData[]> m_db;
};
