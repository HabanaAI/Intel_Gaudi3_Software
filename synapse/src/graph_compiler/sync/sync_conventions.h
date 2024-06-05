#pragma once
#include <string>
#include "sync_types.h"
#include "habana_device_types.h"

enum SyncObjOp
{
    SYNC_OP_SET = 0x0,
    SYNC_OP_ADD = 0x4,
};

class SyncConventions
{
public:
    virtual ~SyncConventions() {}

    virtual std::string getSyncObjName(unsigned int objId) const      = 0;

    virtual unsigned getSyncObjMinId() const                          { return 0; }
    virtual unsigned getSyncObjMinSavedId() const                     = 0;
    virtual unsigned getSyncObjMaxSavedId() const                     = 0;

    virtual unsigned getSyncObjDmaDownFeedback() const                = 0;
    virtual unsigned getSyncObjDmaUpFeedback() const                  = 0;
    virtual unsigned getSyncObjDmaStaticDramSramFeedback() const      = 0;
    virtual unsigned getSyncObjDmaSramDramFeedback() const            = 0;
    virtual unsigned getSyncObjFirstComputeFinish() const             = 0;
    virtual unsigned getSyncObjHostDramDone() const                   = 0;
    virtual unsigned getSyncObjDbgCtr() const                         = 0;
    virtual unsigned getSyncObjDmaActivationsDramSramFeedback() const = 0;
    virtual unsigned getSyncObjEngineSem(unsigned engineIdx) const    = 0;

    virtual unsigned getMonObjMinId() const                           { return 0; }
    virtual unsigned getMonObjMinSavedId() const                      = 0;
    virtual unsigned getMonObjMaxSavedId() const                      = 0;
    virtual unsigned getMonObjEngineSemBase() const                   = 0;

    virtual unsigned getMonObjDmaDownFeedbackReset() const            = 0;
    virtual unsigned getMonObjDmaUpFeedbackReset() const              = 0;

    virtual int getSyncAddOp() const                                  = 0;
    virtual int getSyncSetOp() const                                  = 0;

    virtual unsigned getGroupSize() const                             = 0;

    // Gaudi-specific, Greco-specific
    virtual unsigned getLowerQueueID(HabanaDeviceType deviceType, unsigned engineId) const; // empty implementation

    virtual unsigned getSignalOutGroup() const;
    virtual bool     isSignalOutGroupSupported() const;
    virtual unsigned getNumOfSignalGroups() const { return 0; }
};
