/**********************************************************************************************************
 * This file includes a tool to help measure time. Logging a message takes about 9-30 micro second (not
 * release mode). So using the logs to measure time impacts the results.
 * This file defines an object that can collect the time and dump it later.
 *
 * NOTE: THIS IS NOT THREAD SAFE
 *
 * To use it do the following:
 *
 * 1) diffMeasure myMeasure("Prfix"); //Create the object diffMeasure, with prefix message
 * 2) myMeasure.collect("message");   //Collects the time here. Message is displayed in the log file
 * 3) myMeasure.dump();               //Dumps the collected points (and restart the timer)
 *
 * Some more options:
 * 1) Each object has up to 20 points to collect by default. If you want more, use diffMeasureN<100> myMesuare("Prfix");
 * 2) myMeasure.dump() happens also during destructor.
 * 3) There is one global object with 200 points called perfCollect that you can use (uncomment it in statistics.cpp)
 *
 * Logs go to file perf_measure.log (one file only), with lever INFO. Use env LOG_LEVEL_PERF=2 to really dump to this
 *file
 *************************************************************************************************************/

#ifndef DIFF_MEASURE_H
#define DIFF_MEASURE_H

#include <string>
#include <chrono>


/**************************************************************************************************************************/
/******                                     Diff time collection *********/
/**************************************************************************************************************************/

template<int N>
class diffMeasureN
{
public:
    explicit diffMeasureN(std::string prefix = "") : m_prefix(std::move(prefix)) { reset("Created"); }
    ~diffMeasureN()
    {
        if (m_curr != 1) dump();
    }

    void collect(std::string_view msg)
    {
        if (LOG_LEVEL_AT_LEAST_INFO(PERF))
        {
            m_pointsTime[m_curr] = std::chrono::steady_clock::now();
            //            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &m_pointsTimeThread[m_curr]);
            m_pointsStr[m_curr] = msg;
            m_curr++;
            if (m_curr >= MAX_POINTS)
            {
                LOG_INFO(PERF, "Reached max num of cllection points, dumping. Note, this effects the performance");
                dump();
            }
        }
    }

    void reset(std::string_view msg)
    {
        m_curr          = 1;
        m_pointsTime[0] = std::chrono::steady_clock::now();
        //        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &m_pointsTimeThread[0]);
        m_pointsStr[0] = msg;
    }

    void dump();

private:
    static const int MAX_POINTS = N + 3;  // Because it is not thread safe, make sure m_curr will never overflow

#if 0
    timespec m_pointsTimeThread[MAX_POINTS];
    uint64_t timeSpecDiff(timespec t1, timespec t2)
    {
        return (((t1.tv_sec - t2.tv_sec) * 1000000) + (t1.tv_nsec - t2.tv_nsec))/1000;
    }
#endif

    std::chrono::time_point<std::chrono::steady_clock> m_pointsTime[MAX_POINTS];
    std::string                                        m_pointsStr[MAX_POINTS];
    int                                                m_curr = 0;
    std::string                                        m_prefix;
};  // class diffMeasureN

template<int N>
void diffMeasureN<N>::dump()
{
    for (int i = 0; i < m_curr; i++)
    {
        uint64_t diff_strt =
            std::chrono::duration_cast<std::chrono::nanoseconds>(m_pointsTime[i] - m_pointsTime[0]).count();
        uint64_t diff_prev = 0;
        if (i > 0)
            diff_prev =
                std::chrono::duration_cast<std::chrono::nanoseconds>(m_pointsTime[i] - m_pointsTime[i - 1]).count();

        //        uint64_t diffStrtThread  = timeSpecDiff(m_pointsTimeThread[i], m_pointsTimeThread[0]);
        //        uint64_t diffPrevThread  = 0;
        //        if(i > 0) diffPrevThread = timeSpecDiff(m_pointsTimeThread[i], m_pointsTimeThread[i - 1]);

        //        LOG_INFO(PERF, "{} {} {} {} (thread {} {})", m_prefix, m_pointsStr[i], diff_strt, diff_prev,
        //        diffStrtThread, diffPrevThread);
        LOG_INFO(PERF, "{} {} {} {}", m_prefix, m_pointsStr[i], diff_strt, diff_prev);
    }
    reset("Restart");
}

typedef diffMeasureN<20> diffMeasure;

#define NUM_GLOBAL_COLLECT_POINTS 200
extern diffMeasureN<NUM_GLOBAL_COLLECT_POINTS> perfCollect;

#define INT2STR_DETAIL(x) #x
#define INT2STR(x)        INT2STR_DETAIL(x)

// TIME_COLLECT_HERE macro collects the time, message includes file:line
#define TIME_COLLECT_HERE() perfCollect.collect(__FILE__ ":" INT2STR(__LINE__))

#endif  // DIFF_MEASURE_H
