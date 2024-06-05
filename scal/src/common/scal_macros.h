#pragma once

// clang-format off
#define varoffsetof(t, m)    ((size_t)(&(((t*)0)->m)))
#define upper_32_bits(n)     ((uint32_t) (((n) >> 16) >> 16))
#define lower_32_bits(n)     ((uint32_t) (n))
#define lower_29_bits(n)     (((uint32_t) (n)) & 0x1fffffff)
#define lower_27_bits(n)     (((uint32_t) (n)) & 0x7ffffff)
#define bits_12_to_26(n)     lower_27_bits(n) >> 12
#define lbw_block_address(n) bits_12_to_26(n)

#define scal_assert(cond,msg,...)        \
do {                                     \
    if(!(cond))                          \
        LOG_ERR(SCAL,msg,## __VA_ARGS__);\
} while(0)

#define scal_assert_return(cond,fail_status,msg,...) \
do {                                                 \
    if(!(cond))                                      \
    {                                                \
        LOG_ERR(SCAL,msg,## __VA_ARGS__);            \
        return fail_status;                          \
    }                                                \
} while(0)

#define UNUSED(expr) do { (void)(expr); } while (0)
#define scal_assert_equal_and_set(var,assert_val,fail_status,set_val,msg,...) \
do {\
    if(var != assert_val)\
    {\
        LOG_ERR(SCAL,msg,## __VA_ARGS__);\
        return fail_status;\
    }\
    var = set_val;\
} while(0)
