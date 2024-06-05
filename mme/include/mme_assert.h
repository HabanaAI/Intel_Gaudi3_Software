#ifndef MME__ASSERT_H
#define MME__ASSERT_H

#include <assert.h>
#include <string.h>
#include <string>

#ifdef likely
#undef likely
#endif
#ifdef unlikely
#undef unlikely
#endif
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#ifndef __FILENAME__
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif
#ifdef NDEBUG
constexpr bool mmeReleaseMode = true;
#else
constexpr bool mmeReleaseMode = false;
#endif

namespace MmeCommon
{
class MMEStackException : public std::exception
{
public:
    MMEStackException(std::string exceptionString) { m_exceptionString = exceptionString; }
    const char* what() const noexcept { return m_exceptionString.c_str(); }

private:
    std::string m_exceptionString;
};
// this function is only for MACROs not use it without them
void hbAssert(bool throwException,
              const char* conditionString,
              const std::string& message,
              const char* file,
              const int line,
              const char* func);
}  // namespace MmeCommon

// Guideline for using MACROs:
// if it possible use only MME_ASSERT or MME_ASSERT_PTR
// in destructors and noexcept functions use only MME_ASSERT_DEBUG_ONLY, its forbidden to use MME_ASSERT
// if the error is recoverable and handled but assert is needed use MME_ASSERT_DEBUG_ONLY
// if running validation function that cost in performance and should not effect on Release use HB_DEBUG_VALIDATE
#define MME_ASSERT(condition, message)                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (unlikely(!(condition))) /* likely not taken, hint to optimize runtime */                                   \
        {                                                                                                              \
            MmeCommon::hbAssert(mmeReleaseMode, #condition, message, __FILENAME__, __LINE__, __FUNCTION__);               \
            assert(condition);                                                                                         \
        }                                                                                                              \
    } while (false)

#define MME_ASSERT_DEBUG_ONLY(condition, message)                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (unlikely(!(condition))) /* likely not taken, hint to optimize runtime */                                   \
        {                                                                                                              \
            MmeCommon::hbAssert(false, #condition, message, __FILENAME__, __LINE__, __FUNCTION__);                     \
            assert(condition);                                                                                         \
        }                                                                                                              \
    } while (false)

#define MME_ASSERT_PTR(pointer) MME_ASSERT(pointer != nullptr, "got null pointer")

#endif //MME__ASSERT_H
