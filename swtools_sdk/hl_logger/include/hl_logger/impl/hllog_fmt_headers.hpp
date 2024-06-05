#pragma once
// enable simple ostream<< operator support (like in old versions of fmt)
// this support can be disabled in the future by fmt. see fmt docs
#define FMT_DEPRECATED_OSTREAM

#ifndef HLLOG_DISABLE_FMT_COMPILE
#include <fmt-9.1.0/include/fmt/compile.h>
#else
#define FMT_COMPILE(v) v
#endif
#include <fmt-9.1.0/include/fmt/format.h>
#include <fmt-9.1.0/include/fmt/chrono.h>
#include <fmt-9.1.0/include/fmt/ostream.h>
