#ifndef MME__SKIP_DATA_H
#define MME__SKIP_DATA_H

struct SkipData
{
    bool skipActivation = false;
    bool skipNorthActivation = false;
    unsigned skipDescsNr = 0;
    unsigned skipSignalsNr = 0;

    bool operator==(const SkipData& other) const
    {
        return (skipActivation == other.skipActivation && skipNorthActivation == other.skipNorthActivation &&
                skipDescsNr == other.skipDescsNr && skipSignalsNr == other.skipSignalsNr);
    }
};

#endif //MME__SKIP_DATA_H
