#pragma once

#include "runtime/scal/common/device_scal.hpp"

class DeviceGaudi2scal : public common::DeviceScal
{
public:
    DeviceGaudi2scal(const DeviceConstructInfo& deviceConstructInfo);

    virtual ~DeviceGaudi2scal() = default;

    virtual synStatus acquire(const uint16_t numSyncObj) override;

    virtual const std::vector<uint64_t> getTpcAddrVector() override
    {
        return s_tpcAddrVector;
    }
private:
    static const std::vector<uint64_t> s_tpcAddrVector;
};
