#pragma once

/************************************************************************/
/*                                                                      */
/* This module was written by Alex Vaisman                              */
/*                                                                      */
/************************************************************************/

#include <mutex>
#include <tuple>
#include <functional>
#include <type_traits>
#include <vector>
#include <atomic>

/**
 * @class EventConnection provides connection holder of event handler
 * supports move semantics only
 * disconnects the underlying connection in dtor
 */
class EventConnection
{
public:
    EventConnection()                  = default;
    EventConnection(EventConnection&&) = default;
    EventConnection& operator          =(EventConnection&& other)
    {
        disconnect();
        _disconnectFunc = std::move(other._disconnectFunc);
        return *this;
    }

    EventConnection(EventConnection const&) = delete;
    EventConnection& operator=(EventConnection const&) = delete;

    // disconnect the underlying connection
    void disconnect()
    {
        if (_disconnectFunc)
        {
            _disconnectFunc();
            _disconnectFunc = nullptr;
        }
    }

    ~EventConnection() { disconnect(); }

private:
    EventConnection(std::function<void()> disconnectFunc) : _disconnectFunc(std::move(disconnectFunc)) {}

    template<class...>
    friend class EventDispatcher;
    std::function<void()> _disconnectFunc;
};

/**
 * @class EventDispatcher - an implementation of mediator design pattern. thread-safe.
 * provides events subscription and synchronous events delivery to the subscribers
 *  event - any user-provided struct
 * @param TEvents list of supported events.
 *
 * @code{.cpp}
 * struct Event1{};
 * struct Event2{std::string s;};
 * EventDispatcher<Event1, Event2> eventDispatcher;
 * // for empty events handlers with and without parameter are supported:
 * auto conn1 = eventDispatcher.addEventHandler<Event1>([](){ std::cout << "Event1 handler no param\n"; });
 * auto conn2 = eventDispatcher.addEventHandler<Event1>([](Event1 const &){ std::cout << "Event1 handler with param\n"; });
 * // for empty event send with and without parameter are supported:
 * eventDispatcher.send<Event1>();
 * // output:
 * // Event1 handler no param
 * // Event1 handler with param
 * eventDispatcher.send(Event1{}); // does the the same as eventDispatcher.send<Event1>();
 * auto conn3 = eventDispatcher.addEventHandler<Event2>([](Event2 const & v){ std::cout << "Event2 : " << v.s << "\n"; });
 * eventDispatcher.send(Event2{"hello world"});
 * // output:
 * // Event2 : hello world
 * conn3.disconnect(); // disconnect Event2 handler from EventDispatcher
 * eventDispatcher.send(Event2{"hello world"});
 * // no output (no event handlers for Event2)
 *
 * * if a class has several connections - add them into a vector
 * std::vector<EventConnection> connections;
 * connections.push_back(eventDispatcher.addEventHandler...);
 * connections.push_back(eventDispatcher.addEventHandler...);
 * @endcode
 *
 */
template<class... TEvents>
class EventDispatcher
{
public:
    template<class TEvent>
    void send()
    {
        static_assert(std::is_empty_v<TEvent>, "TEvent is not empty. use: send(TEvent{init values});");
        sendImpl(TEvent {});
    }

    template<class TEvent>
    void send(TEvent const& event)
    {
        sendImpl(event);
    }

    template<class TEvent>
    [[nodiscard]] EventConnection addEventHandler(std::function<void()> eventHandler)
    {
        static_assert(std::is_empty_v<TEvent>,
                      "TEvent is not empty. the handler's signature must be: void (TEvent const &)");
        return addEventHandlerImpl<TEvent>([eventHandler = std::move(eventHandler)](TEvent const&) { eventHandler(); });
    }

    template<class TEvent>
    [[nodiscard]] EventConnection addEventHandler(std::function<void(TEvent const&)> eventHandler)
    {
        return addEventHandlerImpl<TEvent>(std::move(eventHandler));
    }

private:
    template<class TEvent>
    void sendImpl(TEvent const& event)
    {
        static_assert(IsSupportedEvent<TEvent>::value, "TEvent must be one of TEvents...");

        auto& eventHandlersDB = std::get<EventHandlersDB<TEvent>>(_eventHandlersTuple);
        std::lock_guard guard(eventHandlersDB.mtx);
        for (auto& handlerWithId : eventHandlersDB.handlerWithIds)
        {
            handlerWithId.second(event);
        }
    }

    template<class TEvent, class TEventHandler>
    [[nodiscard]] EventConnection addEventHandlerImpl(TEventHandler eventHandler)
    {
        static_assert(IsSupportedEvent<TEvent>::value, "TEvent must be one of TEvents...");

        uint64_t        newUniqueId     = _uniqueId++;
        auto&           eventHandlersDB = std::get<EventHandlersDB<TEvent>>(_eventHandlersTuple);
        EventConnection connection([&eventHandlersDB, newUniqueId]() { erase(eventHandlersDB, newUniqueId); });
        std::lock_guard guard(eventHandlersDB.mtx);
        eventHandlersDB.handlerWithIds.push_back(std::make_pair(newUniqueId, std::move(eventHandler)));
        return connection;
    }

    template<class... Ts>
    struct AreTypesDifferent
    {
        static constexpr bool value = false;
    };
    template<class T, class... Ts>
    struct AreTypesDifferent<T, Ts...>
    {
        static constexpr bool value = (!std::is_same_v<T, Ts> && ...) && AreTypesDifferent<Ts...>::value;
    };
    template<class T>
    struct AreTypesDifferent<T>
    {
        static constexpr bool value = true;
    };
    static_assert(sizeof...(TEvents) > 0, "TEvents... must be not empty");
    static_assert(AreTypesDifferent<TEvents...>::value, "All TEvents... must be different");

    template<class T>
    struct IsSupportedEvent
    {
        static constexpr bool value = (std::is_same_v<T, TEvents> || ...);
    };

    template<class Event>
    struct EventHandlersDB
    {
        using TEventHandler = std::function<void(Event const&)>;
        using HandlerWithId = std::pair<uint64_t, TEventHandler>;
        std::vector<HandlerWithId> handlerWithIds;
        std::mutex                 mtx;
    };

    friend class EventConnection;

    template<class Event>
    static void erase(EventHandlersDB<Event>& eventHandlersDB, uint64_t uniqueId)
    {
        std::lock_guard guard(eventHandlersDB.mtx);

        auto& handlerWithIds = eventHandlersDB.handlerWithIds;
        auto  eraseIt        = std::remove_if(handlerWithIds.begin(), handlerWithIds.end(), [uniqueId](auto const& v) {
            return v.first == uniqueId;
        });
        handlerWithIds.erase(eraseIt, handlerWithIds.end());
    }

    using EventHandlersTuple = std::tuple<EventHandlersDB<TEvents>...>;
    EventHandlersTuple    _eventHandlersTuple;
    std::atomic<uint64_t> _uniqueId {0};
};