#include <gtest/gtest.h>
#include "infra/event_dispatcher.hpp"

// globallly define all the TEvents
struct Ev1
{
};
struct Ev2
{
};
struct Ev3
{
    int         v;
    std::string s;
};
struct Ev4
{
    int         v;
    std::string s;
};

TEST(EventDispatcherTest, empty_events)
{
    EventDispatcher<Ev1, Ev2, Ev3, Ev4> ed;

    int cnt = 0;
    {
        auto conn1 = ed.addEventHandler<Ev1>([&]() { cnt++; });
        auto conn2 = ed.addEventHandler<Ev1>([&](Ev1 const&) { cnt++; });
        ed.send<Ev1>();
        ASSERT_EQ(cnt, 2);
        ed.send<Ev2>();
        ASSERT_EQ(cnt, 2);
        conn1.disconnect();
        ed.send<Ev1>();
        ASSERT_EQ(cnt, 3);
    }
    ed.send<Ev1>();
    ASSERT_EQ(cnt, 3);
}

TEST(EventDispatcherTest, not_empty_events)
{
    EventDispatcher<Ev1, Ev2, Ev3, Ev4> ed;

    static const std::string msgStr = "hello";
    static const int         msgVal = 55;
    int                      cnt    = 0;
    {
        auto conn1 = ed.addEventHandler<Ev3>([&](Ev3 const& v) {
            if (v.s == msgStr && v.v == msgVal) cnt++;
        });
        auto conn2 = ed.addEventHandler<Ev3>([&](Ev3 const& v) {
            if (v.s == msgStr && v.v == msgVal) cnt++;
        });
        ed.send(Ev3 {msgVal, msgStr});
        ASSERT_EQ(cnt, 2);
        conn1.disconnect();
        ed.send(Ev3 {msgVal, msgStr});
        ASSERT_EQ(cnt, 3);
        ed.send(Ev4 {msgVal, msgStr});
        ASSERT_EQ(cnt, 3);
        ed.send(Ev3 {msgVal + 1, msgStr});
        ASSERT_EQ(cnt, 3);
    }
    ed.send(Ev3 {msgVal, msgStr});
    ASSERT_EQ(cnt, 3);
}
