// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
// Distributed under the MIT License (http://opensource.org/licenses/MIT)
#pragma once
#ifndef SPDLOG_HEADER_ONLY
#include "ansicolor_rotating_file_sink.h"
#endif
#include <spdlog/pattern_formatter.h>
#include <spdlog/details/os.h>
namespace spdlog {
namespace sinks {
template<typename Mutex>
SPDLOG_INLINE
ansicolor_rotating_file_sink<Mutex>::ansicolor_rotating_file_sink(filename_t  base_filename,
                                                                  std::size_t max_size,
                                                                  std::size_t max_files,
                                                                  bool        rotate_on_open /* = false*/,
                                                                  bool should_do_colors /* = true*/,
                                                                  const file_event_handlers& event_handlers /*= {}*/)
: base_filename_(std::move(base_filename)),
  max_size_(max_size),
  max_files_(max_files),
  file_helper_ {event_handlers},
  should_do_colors_(should_do_colors)
{
    if (max_size == 0)
    {
        throw_spdlog_ex("rotating sink constructor: max_size arg cannot be zero");
    }
    if (max_files > 200000)
    {
        throw_spdlog_ex("rotating sink constructor: max_files arg cannot exceed 200000");
    }
    file_helper_.open(calc_filename(base_filename_, 0));
    current_size_ = file_helper_.size();  // expensive. called only once
    if (rotate_on_open && current_size_ > 0)
    {
        rotate_();
        current_size_ = 0;
    }
    colors_[level::trace]    = to_string_(white);
    colors_[level::debug]    = to_string_(cyan);
    colors_[level::info]     = to_string_(green);
    colors_[level::warn]     = to_string_(yellow_bold);
    colors_[level::err]      = to_string_(red_bold);
    colors_[level::critical] = to_string_(bold_on_red);
    colors_[level::off]      = to_string_(reset);
}
template<typename Mutex>
SPDLOG_INLINE void ansicolor_rotating_file_sink<Mutex>::set_color(level::level_enum color_level, string_view_t color)
{
    std::lock_guard<typename base_sink<Mutex>::mutex_t> lock(this->mutex_);
    colors_[static_cast<size_t>(color_level)] = to_string_(color);
}
// calc filename according to index and file extension if exists.
// e.g. calc_filename("logs/mylog.txt, 3) => "logs/mylog.3.txt".
template<typename Mutex>
SPDLOG_INLINE filename_t ansicolor_rotating_file_sink<Mutex>::calc_filename(const filename_t& filename,
                                                                            std::size_t       index)
{
    if (index == 0u)
    {
        return filename;
    }
    filename_t basename, ext;
    std::tie(basename, ext) = details::file_helper::split_by_extension(filename);
    return fmt_lib::format(SPDLOG_FILENAME_T("{}.{}{}"), basename, index, ext);
}
template<typename Mutex>
SPDLOG_INLINE filename_t const & ansicolor_rotating_file_sink<Mutex>::filename() const
{
    return file_helper_.filename();
}
template<typename Mutex>
SPDLOG_INLINE void ansicolor_rotating_file_sink<Mutex>::sink_it_(const details::log_msg& msg)
{
    // Wrap the originally formatted message in color codes.
    // If color is not supported in the terminal, log as is instead.
    memory_buf_t formatted;
    this->formatter_->format(msg, formatted);
    auto new_size = current_size_ + formatted.size();
    // rotate if the new estimated file size exceeds max size.
    // rotate only if the real size > 0 to better deal with full disk (see issue #2261).
    // we only check the real size when new_size > max_size_ because it is relatively expensive.
    if (new_size > max_size_)
    {
        file_helper_.flush();
        if (file_helper_.size() > 0)
        {
            rotate_();
            new_size = formatted.size();
        }
    }
    if (should_do_colors_ && msg.color_range_end > msg.color_range_start)
    {
        string_view_t level_color = colors_[static_cast<size_t>(msg.level)];
        print_ccode_(level_color);
        file_helper_.write(formatted);
        print_ccode_(reset);
        new_size += reset.size() + level_color.size();
    }
    else  // no color
    {
        file_helper_.write(formatted);
    }
    current_size_ = new_size;
}
template<typename Mutex>
SPDLOG_INLINE void ansicolor_rotating_file_sink<Mutex>::flush_()
{
    file_helper_.flush();
}
// Rotate files:
// log.txt -> log.1.txt
// log.1.txt -> log.2.txt
// log.2.txt -> log.3.txt
// log.3.txt -> delete
template<typename Mutex>
SPDLOG_INLINE void ansicolor_rotating_file_sink<Mutex>::rotate_()
{
    using details::os::filename_to_str;
    using details::os::path_exists;
    file_helper_.close();
    std::string rotation_error;
    try
    {
        for (auto i = max_files_; i > 0; --i)
        {
            filename_t src = calc_filename(base_filename_, i - 1);
            if (!path_exists(src))
            {
                continue;
            }
            filename_t target = calc_filename(base_filename_, i);
            if (!rename_file_(src, target))
            {
                // if failed try again after a small delay.
                // this is a workaround to a windows issue, where very high rotation
                // rates can cause the rename to fail with permission denied (because of antivirus?).
                details::os::sleep_for_millis(100);
                if (!rename_file_(src, target))
                {
                    file_helper_.reopen(true);  // truncate the log file anyway to prevent it to grow beyond its limit!
                    current_size_ = 0;
                    throw_spdlog_ex("rotating_file_sink: failed renaming " + filename_to_str(src) + " to " +
                                        filename_to_str(target),
                                    errno);
                }
            }
        }
    }
    catch (std::exception const &e)
    {
        rotation_error = e.what();
    }
    catch(...)
    {
        rotation_error = "unknown exception";
    }
    file_helper_.reopen(true);
    if (!rotation_error.empty())
    {
        memory_buf_t buf;
        buf.append(rotation_error.data(), rotation_error.data() + rotation_error.size());
        file_helper_.write(buf);
    }
}
// delete the target if exists, and rename the src file  to target
// return true on success, false otherwise.
template<typename Mutex>
SPDLOG_INLINE bool ansicolor_rotating_file_sink<Mutex>::rename_file_(const filename_t& src_filename,
                                                                     const filename_t& target_filename)
{
    // try to delete the target file in case it already exists.
    (void)details::os::remove(target_filename);
    return details::os::rename(src_filename, target_filename) == 0;
}
template<typename Mutex>
SPDLOG_INLINE bool ansicolor_rotating_file_sink<Mutex>::should_color()
{
    return should_do_colors_;
}
template<typename Mutex>
SPDLOG_INLINE void ansicolor_rotating_file_sink<Mutex>::print_ccode_(const string_view_t& color_code)
{
    memory_buf_t buf;
    buf.append(color_code.begin(), color_code.end());
    file_helper_.write(buf);
}
template<typename ConsoleMutex>
SPDLOG_INLINE std::string ansicolor_rotating_file_sink<ConsoleMutex>::to_string_(const string_view_t& sv)
{
    return std::string(sv.data(), sv.size());
}
}  // namespace sinks
}  // namespace spdlog