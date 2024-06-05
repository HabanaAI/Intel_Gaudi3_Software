#pragma once

namespace synapse
{

class ThreadWorkItem
{
public:
    virtual void call()
    {
        doWork();
        if (shouldDeleteItem())
        {
            delete this;
        }
    }

    virtual void doWork() = 0;

    virtual bool shouldDeleteItem() const
    {
        return true;
    }

protected:
    virtual ~ThreadWorkItem() {}
};

}
