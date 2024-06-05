#pragma once

#include "stdint.h"
#include <deque>

using WcmPhysicalQueuesId = uint32_t;
using WcmCsHandle         = uint64_t;
using WcmCsHandleQueue    = std::deque<WcmCsHandle>;
