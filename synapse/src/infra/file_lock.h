#pragma once

#include "log_manager.h"
#include "types_exception.h"
#include "types.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

class FileLock
{
public:
    static FileLock lock(const std::string& path)
    {
        int fd = open(path.c_str(), O_RDWR);
        if (fd < 0)
        {
            throw SynapseException(fmt::format("failed to open file lock, name: {}, error: {}", path, errno));
        }

        LOG_DEBUG(SYNREC, "try lock file: {}, pid: {}", path, getpid());

        int sts = lockf(fd, F_LOCK, 0);

        if (sts == 0)
        {
            LOG_DEBUG(SYNREC, "lock acuired file name: {}, pid: {}", path, getpid());
            return FileLock(fd);
        }

        close(fd);

        throw SynapseException(
            fmt::format("failed to lock file, file name: {}, error: {}, pid: {}", path, errno, getpid()));
    }

    ~FileLock()
    {
        if (m_fileDescriptor != -1)
        {
            int sts = lockf(m_fileDescriptor, F_ULOCK, 0);
            if (sts != 0)
            {
                LOG_WARN(SYNREC,
                         "failed to unlock file, fd: {}, error: {}, pid: {}",
                         m_fileDescriptor,
                         errno,
                         getpid());
            }

            LOG_DEBUG(SYNREC, "file unlocked, fd: {}, pid: {}", m_fileDescriptor, getpid());
            close(m_fileDescriptor);
        }
    }

private:
    FileLock(int fileDescriptor) : m_fileDescriptor(fileDescriptor) {}

    int m_fileDescriptor = -1;
};
