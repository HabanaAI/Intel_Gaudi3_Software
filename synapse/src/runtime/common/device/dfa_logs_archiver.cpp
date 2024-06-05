#include <libarchive/libarchive/archive.h>
#include <libarchive/libarchive/archive_entry.h>
#include <stdio.h>
#include <tuple>
#include <string>
#include <vector>
#include <thread>
#include <hl_logger/hllog_core.hpp>
#include "dfa_defines.hpp"
#include <limits.h>
#include <sys/stat.h>

static unsigned getEnvVarValue(const char * envvarName, unsigned defaultValue)
{
    unsigned value = defaultValue;
    char* newValueStr = getenv(envvarName);
    if (newValueStr)
    {
        value = atoi(newValueStr);
    }
    return value;
}

const unsigned dfaLogsArchiveFileAmount = getEnvVarValue("SYNAPSE_DFA_ARCHIVE_AMOUNT", 9);

static bool pathExists(const std::string &filename)
{
    struct stat buffer;
    return (::stat(filename.c_str(), &buffer) == 0);
}

static std::tuple<std::string, std::string> splitByExtension(const std::string &fname)
{
    auto ext_index = fname.rfind('.');

    // no valid extension found - return whole path and empty string as
    // extension
    if (ext_index == std::string::npos || ext_index == 0 || ext_index == fname.size() - 1)
    {
        return std::make_tuple(fname, std::string());
    }

    // treat cases like "/etc/rc.d/somelogfile or "/abc/.hiddenfile"
    auto folder_index = fname.find_last_of("/");
    if (folder_index != std::string::npos && folder_index >= ext_index - 1)
    {
        return std::make_tuple(fname, std::string());
    }

    // finally - return a valid base and extension tuple
    return std::make_tuple(fname.substr(0, ext_index), fname.substr(ext_index));
}

// calc filename according to index and file extension if exists.
// e.g. calc_filename("logs/mylog.txt, 3) => "logs/mylog.3.txt".
static std::string calcFilename(const std::string& filename, std::size_t index)
{
    if (index == 0u)
    {
        return filename;
    }
    std::string basename, ext;
    std::tie(basename, ext) = splitByExtension(filename);
    return basename + "." + std::to_string(index) + ext;
}

// delete the target if exists, and rename the src file  to target
// return true on success, false otherwise.
static bool renameFile(const std::string& src_filename, const std::string& target_filename)
{
    // try to delete the target file in case it already exists.
    std::remove(target_filename.c_str());
    return std::rename(src_filename.c_str(), target_filename.c_str()) == 0;
}

// Rotate files:
// log.txt -> log.1.txt
// log.1.txt -> log.2.txt
// log.2.txt -> log.3.txt
// log.3.txt -> delete
static void rotateFile(std::string const & filename)
{
    try
    {
        std::string base_filename_;
        std::string ext_;
        std::tie(base_filename_, ext_) = splitByExtension(filename);
        for (auto i = dfaLogsArchiveFileAmount; i > 0; --i)
        {
            auto src = calcFilename(filename, i - 1);
            if (!pathExists(src))
            {
                continue;
            }
            auto target = calcFilename(filename, i);
            if (!renameFile(src, target))
            {
                // if failed try again after a small delay.
                // this is a workaround to a windows issue, where very high rotation
                // rates can cause the rename to fail with permission denied (because of antivirus?).
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (!renameFile(src, target))
                {
                    // double rename failed
                }
            }
        }
    }
    catch(...)
    {
    }
}

static std::string getFilenameFromFd(int fd)
{
    char path[PATH_MAX];
    char resolved_path[PATH_MAX];

    snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
    ssize_t len = readlink(path, resolved_path, sizeof(resolved_path) - 1);
    if (len != -1) {
        resolved_path[len] = '\0';
        return std::string(resolved_path);
    }

    return ""; // return empty string in case of failure
}

static void archiveFiles(std::string const & dir, std::string const & archFilename, std::vector<std::string> const & filenames)
{
    unsigned validFilesCount = 0;
    for (auto const & filename : filenames)
    {
        std::string fullFilename = dir + "/" + filename;
        struct stat st;
        if (stat(fullFilename.c_str(), &st) == 0)
        {
            validFilesCount++;
        }
    }

    if (validFilesCount == 0)
    {
        return;
    }

    std::string fullArchiveFilename = dir + "/" + archFilename;
    std::vector<char> buff(1 * 1024 * 1024);
    int err = 0;
    std::string tmpArchNameTemplate = fullArchiveFilename + "_XXXXXX";
    int archFd = mkstemp(tmpArchNameTemplate.data());
    std::string tmpArchFilename = getFilenameFromFd(archFd);
    if (archFd != -1 && !tmpArchFilename.empty())
    {
        archive * arch = archive_write_new();

        err = archive_write_set_format_zip(arch);
        err     = err | archive_write_open_fd(arch, archFd);

        for (auto const & filename : filenames)
        {
            std::string fullFilename = dir + "/" + filename;
            struct stat st;
            if (stat(fullFilename.c_str(), &st) != 0)
            {
                continue;
            }
            FILE * f = fopen(fullFilename.c_str(), "r");
            if (f == nullptr)
            {
                continue;
            }
            archive_entry *entry = archive_entry_new();
            if (entry == nullptr)
            {
                fclose(f);
                continue;
            }
            archive_entry_set_pathname(entry, filename.c_str());
            archive_entry_set_size(entry, st.st_size);
            archive_entry_set_atime(entry, st.st_atim.tv_sec, st.st_atim.tv_nsec);
            archive_entry_set_birthtime(entry, st.st_ctim.tv_sec, st.st_ctim.tv_nsec);
            archive_entry_set_ctime(entry, st.st_ctim.tv_sec, st.st_ctim.tv_nsec);
            archive_entry_set_mtime(entry, st.st_mtim.tv_sec, st.st_mtim.tv_nsec);
            archive_entry_set_filetype(entry, AE_IFREG);
            archive_entry_set_perm(entry, 0644);
            archive_write_header(arch, entry);
            int len = fread(buff.data(), 1, buff.size(), f);
            while ( len > 0 ) {
                auto writtenLen = archive_write_data(arch, buff.data(), len);
                if (writtenLen != len)
                {
                    err = 1;
                    break;
                }
                len = fread(buff.data(), 1, buff.size(), f);
            }
            fclose(f);
            archive_entry_free(entry);
        }
        err = err | archive_write_close(arch);
        err = err | archive_write_free(arch);
    }
    else
    {
        err = 1;
    }

    if (err == 0)
    {
        rotateFile(fullArchiveFilename);
        if (renameFile(tmpArchFilename, fullArchiveFilename))
        {
            for (auto const & filename : filenames)
            {
                std::string fullFilename = dir + "/" + filename;
                std::remove(fullFilename.c_str());
            }
        }
    }
}

void archiveDfaLogs()
{
    auto dir = hl_logger::getLogsFolderPath();
    archiveFiles(dir, "dfa_logs.zip",
                 {DFA_API_FILE, DEVICE_FAIL_ANALYSIS_FILE, SUSPECTED_RECIPES, DMESG_COPY_FILE, DFA_NIC_INFO_FILE});
}
// before module logs initialization zip dfa files into an archive
// then rotate dfaLogsArchive
// we don't want to keep old dfa file because unzipped because it will add noise while grep'ing logs
// process forks, then child crashes (writes dfa) and then parent crashes (should write into the same dfa log file)
struct DfaLogsArchiver
{
    DfaLogsArchiver()
    {
        archiveDfaLogs();
    }
};

static DfaLogsArchiver dfaLogsArchiver;