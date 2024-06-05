#pragma once

#include "infra/event_dispatcher.hpp"
#include "internal/dfa_defines.hpp"

struct EventDfaPhase
{
    DfaPhase dfaPhase;
};

// global instance of event dispatcher
inline EventDispatcher<EventDfaPhase> synEventDispatcher;
