#include "defs.h"
#include "sql_db.h"

#include <chrono>
#include <functional>
#include <memory>
#include <sqlite/sqlite3.h>
#include <sys/file.h>
#include <thread>
#include <unistd.h>

using namespace data_serialize;

static const uint64_t SLEEP_DURATION = 100;

Sql::File::File(const std::string& filePath, bool readOnly) : m_path(filePath), m_descriptor(-1)
{
    int flags    = readOnly ? O_RDONLY : (O_RDWR | O_CREAT);
    m_descriptor = open(m_path.c_str(), flags, 0666);
    HB_ASSERT(m_descriptor != -1, "failed to open file: {}, pid: {}, ", m_path, getpid());
}

Sql::File::~File()
{
    if (m_descriptor != -1)
    {
        close(m_descriptor);
    }
}

const std::string& Sql::File::path() const
{
    return m_path;
}

int Sql::File::descriptor() const
{
    return m_descriptor;
}

Sql::SqlDb::SqlDb(const std::string& filePath, bool readOnly)
{
    int flags =
        SQLITE_OPEN_FULLMUTEX | (readOnly ? SQLITE_OPEN_READONLY : (SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE));
    HB_ASSERT(sqlite3_open_v2(filePath.c_str(), &m_db, flags, nullptr) == SQLITE_OK,
              "failed to open database: {}, error {}",
              filePath,
              sqlite3_errmsg(m_db));
}

Sql::SqlDb::~SqlDb()
{
    if (m_db != nullptr && sqlite3_close(m_db) != SQLITE_OK)
    {
        LOG_ERR(SYNREC, "failed to close sql DB");
    }
}

sqlite3* Sql::SqlDb::db()
{
    return m_db;
}

Sql::Lock::Lock(const std::shared_ptr<Sql::File>& file, bool readOnly, uint64_t millisecondsTimeout) : m_dbFile(file)
{
    using namespace std::chrono;
    const auto begin         = steady_clock::now();
    auto       sleepDuration = milliseconds(std::clamp<uint64_t>(millisecondsTimeout / 16, 1, SLEEP_DURATION));

    int rc = -1;
    while (duration_cast<milliseconds>(steady_clock::now() - begin).count() < millisecondsTimeout)
    {
        // using flock instead sql lock due to this known issue:
        // "This is because fcntl() file locking is broken on many NFS implementations. You should avoid putting SQLite
        // database files on NFS if multiple processes might try to access the file at the same time"
        rc = flock(file->descriptor(), LOCK_EX);
        if (rc == 0) break;
        std::this_thread::sleep_for(sleepDuration);
        sleepDuration = milliseconds(std::max(uint64_t(sleepDuration.count() / 2),
                                              uint64_t(1)));  // reduce starvation by reducing the sleep duration
    }

    HB_ASSERT(rc == 0, "failed to lock file: {}, pid: {}, ", file->path(), getpid());
    m_sqlDb = std::make_unique<SqlDb>(file->path(), readOnly);
}

Sql::Lock::~Lock()
{
    if (flock(m_dbFile->descriptor(), LOCK_UN) != 0)
    {
        LOG_ERR(SYNREC, "failed to unlock fd: {}, pid: {}, ", m_dbFile->descriptor(), getpid());
    }
}

sqlite3* Sql::Lock::db()
{
    return m_sqlDb->db();
}

Sql::Sql(const std::string& dbName, bool readOnly)
: m_dbFile(std::make_shared<File>(dbName, readOnly)), m_readOnly(readOnly)
{
    HB_ASSERT(sqlite3_threadsafe() == 1, "sqlite wasn't compile in thread safe mode");
}

Sql::~Sql() = default;