#pragma once

#include "defs.h"
#include "data_serializer/ds_types.h"
#include "synapse_common_types.h"
#include <functional>
#include <memory>
#include <set>
#include <sqlite/sqlite3.h>
#include <vector>

namespace data_serialize
{
static const int      INVALID_CALLBACK_RESULT = -1;
static const uint64_t DEFAULT_TIMEOUT         = 60e3;

struct TensorTableColumn
{
    synTensorType        type;
    synDataType          dataType;
    std::vector<TSize>   shape;
    std::vector<uint8_t> permutation;
    uint64_t             dataSize;
    uint8_t*             data;
    bool                 invalidData;
    Compression          compression;
    uint64_t             compressedDataSize;
    std::set<uint64_t>   dataIterations;
    uint8_t              dimSizeInBytes;
};

template<typename To, typename From>
inline std::enable_if_t<sizeof(To) == sizeof(From) &&
                            (std::is_trivially_copyable_v<From> && std::is_trivially_copyable_v<To> &&
                             std::is_trivially_constructible_v<To>),
                        To>
bit_cast(const From& src)
{
    To res;
    std::memcpy(&res, &src, sizeof(To));
    return res;
}

class Sql
{
protected:
    class File
    {
    public:
        File(const std::string& filePath, bool readOnly);
        ~File();

        const std::string& path() const;
        int                descriptor() const;

    private:
        std::string m_path;
        int         m_descriptor;
    };

    class SqlDb
    {
    public:
        SqlDb(const std::string& filePath, bool readOnly);
        ~SqlDb();

        sqlite3* db();

    private:
        sqlite3* m_db = nullptr;
    };

    class Lock
    {
    public:
        Lock(const std::shared_ptr<File>& file, bool readOnly, uint64_t millisecondsTimeout = DEFAULT_TIMEOUT);
        ~Lock();

        sqlite3* db();

    private:
        std::shared_ptr<File>  m_dbFile;
        std::unique_ptr<SqlDb> m_sqlDb;
    };

    Sql(const std::string& dbName, bool readOnly);
    virtual ~Sql();

    template<class T>
    static int singleIntCallback(void* data, int argc, char** argv, char** colNames)
    {
        if (argv[0] == nullptr) return INVALID_CALLBACK_RESULT;
        *static_cast<T*>(data) = std::stoll(argv[0]);
        return SQLITE_OK;
    }

    template<class F>
    static int exeCmd(sqlite3* db, const std::string& cmd, void* data, F callback, bool throwOnFailure = true)
    {
        char* error = nullptr;
        auto  sts   = sqlite3_exec(db, cmd.c_str(), callback, data, &error);
        if (sts != SQLITE_OK)
        {
            const std::string err = error ? error : "unknown";
            if (error) sqlite3_free(error);
            HB_ASSERT(!throwOnFailure, "failed to exe sql command: {}, error: {}", cmd, err);
        }
        return sts;
    }

    template<class F>
    int exe(const std::string& cmd, void* data, F callback, bool throwOnFailure = true) const
    {
        if (m_readOnly)
        {
            SqlDb sqlDb(m_dbFile->path(), m_readOnly);
            return exeCmd(sqlDb.db(), cmd, data, callback, throwOnFailure);
        }
        else
        {
            Lock lk(m_dbFile, m_readOnly);
            return exeCmd(lk.db(), cmd, data, callback, throwOnFailure);
        }
    }

    std::shared_ptr<File> m_dbFile;
    bool                  m_readOnly;
};
}  // namespace data_serialize
