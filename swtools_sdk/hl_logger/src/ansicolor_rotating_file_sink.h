// temporary solution for transition to spldlog 1.10.0
// a custom pattern with %<%> should be used instead
#pragma once
#include <spdlog/details/console_globals.h>
#include <spdlog/details/null_mutex.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/details/file_helper.h>
#include <memory>
#include <string>
#include <array>
namespace spdlog {
namespace sinks {
/**
 * This sink prefixes the output with an ANSI escape sequence color code
 * depending on the severity
 * of the message.
 * If no color terminal detected, omit the escape codes.
 */
template<typename Mutex>
class ansicolor_rotating_file_sink : public base_sink<Mutex>
{
public:
    ansicolor_rotating_file_sink(filename_t                 base_filename,
                                 std::size_t                max_size,
                                 std::size_t                max_files,
                                 bool                       rotate_on_open   = false,
                                 bool                       should_do_colors = true,
                                 const file_event_handlers& event_handlers   = {});
    static filename_t calc_filename(const filename_t& filename, std::size_t index);
    const filename_t& filename() const;
    void              set_color(level::level_enum color_level, string_view_t color);
    bool              should_color();

protected:
    void sink_it_(const details::log_msg& msg) override;
    void flush_() override;

public:
    // Formatting codes
    const string_view_t reset      = "\033[m";
    const string_view_t bold       = "\033[1m";
    const string_view_t dark       = "\033[2m";
    const string_view_t underline  = "\033[4m";
    const string_view_t blink      = "\033[5m";
    const string_view_t reverse    = "\033[7m";
    const string_view_t concealed  = "\033[8m";
    const string_view_t clear_line = "\033[K";
    // Foreground colors
    const string_view_t black   = "\033[30m";
    const string_view_t red     = "\033[31m";
    const string_view_t green   = "\033[32m";
    const string_view_t yellow  = "\033[33m";
    const string_view_t blue    = "\033[34m";
    const string_view_t magenta = "\033[35m";
    const string_view_t cyan    = "\033[36m";
    const string_view_t white   = "\033[37m";
    /// Background colors
    const string_view_t on_black   = "\033[40m";
    const string_view_t on_red     = "\033[41m";
    const string_view_t on_green   = "\033[42m";
    const string_view_t on_yellow  = "\033[43m";
    const string_view_t on_blue    = "\033[44m";
    const string_view_t on_magenta = "\033[45m";
    const string_view_t on_cyan    = "\033[46m";
    const string_view_t on_white   = "\033[47m";
    /// Bold colors
    const string_view_t yellow_bold = "\033[33m\033[1m";
    const string_view_t red_bold    = "\033[31m\033[1m";
    const string_view_t bold_on_red = "\033[1m\033[41m";

private:
    // Rotate files:
    // log.txt -> log.1.txt
    // log.1.txt -> log.2.txt
    // log.2.txt -> log.3.txt
    // log.3.txt -> delete
    void rotate_();
    // delete the target if exists, and rename the src file  to target
    // return true on success, false otherwise.
    bool rename_file_(const filename_t& src_filename, const filename_t& target_filename);

    filename_t           base_filename_;
    std::size_t          max_size_;
    std::size_t          max_files_;
    std::size_t          current_size_;
    details::file_helper file_helper_;
    bool                                     should_do_colors_;
    std::array<std::string, level::n_levels> colors_;
    void                                     print_ccode_(const string_view_t& color_code);
    static std::string                       to_string_(const string_view_t& sv);
};
using ansicolor_rotating_file_sink_mt = ansicolor_rotating_file_sink<std::mutex>;
using ansicolor_rotating_file_sink_st = ansicolor_rotating_file_sink<details::null_mutex>;
}  // namespace sinks
}  // namespace spdlog
#ifdef SPDLOG_HEADER_ONLY
#include "ansicolor_rotating_file_sink.inl"
#endif