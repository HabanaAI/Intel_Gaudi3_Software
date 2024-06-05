#pragma once
HLLOG_BEGIN_NAMESPACE
namespace details{

template<typename T>
struct remove_cvref
{
    typedef std::remove_cv_t<std::remove_reference_t<T>> type;
};
template<typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

template <class TEnum>
struct EnumPack{
    TEnum value;
};
} // details
HLLOG_END_NAMESPACE

template <class T>
struct fmt::formatter<hl_logger::HLLOG_INLINE_API_NAMESPACE::details::EnumPack<T>> : formatter<std::string_view>
{
    bool useIntParser = false;
    formatter<int> intFormatter;
    // Parses format specifications of the form ['f' | 'e'].
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    auto end = intFormatter.parse(ctx);
#if MAGIC_ENUM_SUPPORTED
        useIntParser = false;
        for (auto it = ctx.begin(); it != end; ++it)
        {
            switch (*it)
            {
                case 'X' :
                case 'x' :
                case 'b' :
                case 'B' :
                case 'o' :
                case 'd' :
                case '#' :
                    useIntParser = true;
                    break;
                default: ;
            }
        }
#else
        useIntParser = true;
#endif

        return useIntParser ? end : formatter<std::string_view>::parse(ctx);
    }
    template <typename FormatContext>
    auto format(hl_logger::HLLOG_INLINE_API_NAMESPACE::details::EnumPack<T> v, FormatContext& ctx) const -> decltype(ctx.out())
    {
#if MAGIC_ENUM_SUPPORTED
        if (!useIntParser)
        {
            fmt::memory_buffer buf;
            fmt::format_to(std::back_inserter(buf), FMT_COMPILE("{}[{}]"), uint64_t(v.value), magic_enum::enum_name(v.value));
            return formatter<std::string_view>::format(std::string_view(buf.data(), buf.size()), ctx);
        }
#endif
        return intFormatter.format(uint64_t(v.value), ctx);
    }
};

HLLOG_BEGIN_NAMESPACE
namespace details{
/**
 * spdlog requires to pass all the parameters with perfect forwarding
 * unfortunately it does not work for bit fields
 * in order to distinguish b/w bit fields and other variables duplicate function is used
 * duplicate has the following options:
 * 1. for integral types (e.i. applicable to bit fields)
 * 2. for not integral types
 * 3. enums - return a EnumPack that will print enum value and its string
 * 4. pointers - return uint64_t (for now)
 * overload resolution b/w them is resolved with enable_if<is_integral<T>>
 *
 * @tparam T type of a parameter
 * @param v  parameter
 * @return the same parameter or EnumPack for enums
 */
template<typename T, std::enable_if_t<std::is_integral_v<remove_cvref_t<T>>>* = nullptr>
constexpr auto duplicate(T v) ->std::conditional_t<std::is_enum_v<remove_cvref_t<T>>, EnumPack<remove_cvref_t<T>>, T>
{
    if constexpr (std::is_enum_v<remove_cvref_t<T>>)
    {
        return {v};
    }
    else
    {
        return v;
    }
}

template<typename T, std::enable_if_t<!std::is_integral_v<remove_cvref_t<T>>>* = nullptr>
constexpr auto duplicate(T&& v) ->std::conditional_t<std::is_enum_v<remove_cvref_t<T>>, EnumPack<remove_cvref_t<T>>, T>
{
    if constexpr (std::is_enum_v<remove_cvref_t<T>>)
    {
        return {v};
    }
    else
    {
        return std::forward<T>(v);
    }
}
// fmt handles pointers ina special way. in our case we wait them to be uint64_t
// we can convert into a special type that has a output as a hex value
template<typename T>
constexpr auto duplicate(T* v)
{
    return (void*)v;
}

// C strings should stay the same
constexpr const char* duplicate(const char* v)
{
    return v;
}

constexpr char* duplicate(char* v)
{
    return v;
}
}  // namespace synapse::details
HLLOG_END_NAMESPACE