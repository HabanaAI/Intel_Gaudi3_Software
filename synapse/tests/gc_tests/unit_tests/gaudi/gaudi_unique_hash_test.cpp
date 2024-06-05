#include "gtest/gtest.h"
#include "unique_hash_test.h"
#include "gaudi_graph.h"

class GaudiUniqueHash : public UniqueHash {};

TEST_F(GaudiUniqueHash, blob_equality)
{
    GaudiGraph g;
    // compile here and in other tests so a queue cmd factory will exist for NOP addition in addBlob
    g.compile();
    blob_equality(&g);
}

TEST_F(GaudiUniqueHash, blob_inequality)
{
    GaudiGraph g;
    g.compile();
    blob_inequality(&g);
}
