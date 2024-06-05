#include "gc_dynamic_shapes_infra.h"

struct tensorSizes
{
    std::vector<unsigned> in1MinSizes;
    std::vector<unsigned> in1MaxSizes;
    std::vector<unsigned> in1ActualSizes;
    std::vector<unsigned> in2MinSizes;
    std::vector<unsigned> in2MaxSizes;
    std::vector<unsigned> in2ActualSizes;
    std::vector<unsigned> outMinSizes;
    std::vector<unsigned> outMaxSizes;
    std::vector<unsigned> outActualSizes;
};

class SynGaudiDynamicEinsum
: public SynGaudiDynamicShapesTestsInfra
, public testing::WithParamInterface<const char*>
{
public:
    void runTest(tensorSizes& ts, synEinsumParams& params)
    {
        TensorIndices inputs;
        unsigned      in1Tensor = createPersistTensor(INPUT_TENSOR,
                                                 MEM_INIT_ALL_ONES,
                                                 nullptr,
                                                 ts.in1MaxSizes.data(),
                                                 ts.in1MaxSizes.size(),
                                                 syn_type_float,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 0,
                                                 nullptr,
                                                 ts.in1MinSizes.data());
        inputs.push_back(in1Tensor);

        if (ts.in2MaxSizes.size() > 0)
        {
            unsigned in2Tensor = createPersistTensor(INPUT_TENSOR,
                                                     MEM_INIT_ALL_ONES,
                                                     nullptr,
                                                     ts.in2MaxSizes.data(),
                                                     ts.in2MaxSizes.size(),
                                                     syn_type_float,
                                                     nullptr,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     ts.in2MinSizes.data());
            inputs.push_back(in2Tensor);
        }

        unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 ts.outMaxSizes.data(),
                                                 ts.outMaxSizes.size(),
                                                 syn_type_float,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 0,
                                                 nullptr,
                                                 ts.outMinSizes.data());

        addNodeToGraph(NodeFactory::einsumTypeName, inputs, {outTensor}, &params, sizeof(params), "Einsum0");

        compileTopology();
        ASSERT_FALSE(HasFailure()) << "Compilation failed";

        setActualSizes(inputs[0], ts.in1ActualSizes.data());
        if (ts.in2MaxSizes.size() > 0)
        {
            setActualSizes(inputs[1], ts.in2ActualSizes.data());
        }
        setActualSizes(outTensor, ts.outActualSizes.data());

        runTopology(0, true);
        ASSERT_FALSE(HasFailure()) << "Launch failed";
    }
};


class SynGaudiDynamicEinsumUB : public SynGaudiDynamicEinsum
{
};

INSTANTIATE_TEST_SUITE_P( , SynGaudiDynamicEinsumUB, ::testing::Values("ab->", "ab->ab", "abc->b", "abc->ca", "abc->cab"));

TEST_P_GC(SynGaudiDynamicEinsumUB, unary_basic)
{
    // Tensor sizes are in reverse, i.e. "abc" {3,4,5} means a=5, b=4, c=3
    std::map<std::string, tensorSizes> pMap = {
    // key                  in1: Min        Max        Actual in2: N/A     out: Min        Max        Actual
    {std::string("ab->"),      { {1, 2},    {8, 9},    {3, 4},     {}, {}, {},  {1},       {1},       {1}       } },
    {std::string("ab->ab"),    { {1, 2},    {8, 9},    {3, 4},     {}, {}, {},  {1, 2},    {8, 9},    {3, 4}    } },
    {std::string("abc->b"),    { {1, 2, 3}, {7, 8, 9}, {3, 4, 5},  {}, {}, {},  {2},       {8},       {4}       } },
    {std::string("abc->ca"),   { {1, 2, 3}, {7, 8, 9}, {3, 4, 5},  {}, {}, {},  {3, 1},    {9, 7},    {5, 3}    } },
    {std::string("abc->cab"),  { {1, 2, 3}, {7, 8, 9}, {3, 4, 5},  {}, {}, {},  {2, 3, 1}, {8, 9, 7}, {4, 5, 3} } }
    };

    synEinsumParams params(GetParam());

    runTest(pMap[std::string(params.equation)], params);
}

class SynGaudiDynamicEinsumUE : public SynGaudiDynamicEinsum
{
};

INSTANTIATE_TEST_SUITE_P( , SynGaudiDynamicEinsumUE, ::testing::Values("...ijk->...ki", "...jkl->...lj"));

TEST_P_GC(SynGaudiDynamicEinsumUE, unary_ellipsis)
{
    // Tensor sizes are in reverse, i.e. "abc" {5,4,3} means a=3, b=4, c=5
    std::map<std::string, tensorSizes> pMap = {
    // key                       in1: Min              Max              Actual       in2: N/A     out: Min           Max           Actual
    {std::string("...ijk->...ki"),  { {4, 3, 2, 1},    {9, 8, 7, 6},    {6, 5, 4, 3},     {}, {}, {},  {2, 4, 1},    {7, 9, 6},    {4, 6, 3}    } },
    {std::string("...jkl->...lj"),  { {5, 4, 3, 2, 1}, {9, 8, 7, 6, 5}, {7, 6, 5, 4, 3},  {}, {}, {},  {3, 5, 2, 1}, {7, 9, 6, 5}, {5, 7, 4, 3} } }
    };

    synEinsumParams params(GetParam());

    runTest(pMap[std::string(params.equation)], params);
}

class SynGaudiDynamicEinsumBB : public SynGaudiDynamicEinsum
{
};

INSTANTIATE_TEST_SUITE_P( , SynGaudiDynamicEinsumBB,
    ::testing::Values("a,a->", "a,a->a", "ab,b->a", "ab,ab->", "ab,bc->ac", "abc,bad->abcd"));

TEST_P_GC(SynGaudiDynamicEinsumBB, binary_basic)
{
    // Tensor sizes are in reverse, i.e. "abc" {5,4,3} means a=3, b=4, c=5
    std::map<std::string, tensorSizes> pMap = {
    // key                       in1: Min        Max        Actual in2: Min        Max        Actual out: Min           Max           Actual
    {std::string("a,a->"),          { {2},       {9},       {3},        {2},       {9},       {3},        {1},          {1},          {1}          } },
    {std::string("a,a->a"),         { {2},       {9},       {3},        {2},       {9},       {3},        {2},          {9},          {3}          } },
    {std::string("ab,b->a"),        { {2, 1},    {9, 8},    {4, 3},     {2},       {9},       {4},        {1},          {8},          {3}          } },
    {std::string("ab,ab->"),        { {2, 1},    {9, 8},    {4, 3},     {2, 1},    {9, 8},    {4, 3},     {1},          {1},          {1}          } },
    {std::string("ab,bc->ac"),      { {2, 1},    {8, 7},    {4, 3},     {3, 2},    {9, 8},    {5, 4},     {3, 1},       {9, 7},       {5, 3}       } },
    {std::string("abc,bad->abcd"),  { {3, 2, 1}, {8, 7, 6}, {5, 4, 3},  {4, 1, 2}, {9, 6, 7}, {6, 3, 4},  {4, 3, 2, 1}, {9, 8, 7, 6}, {6, 5, 4, 3} } }
    };

    synEinsumParams params(GetParam());

    runTest(pMap[std::string(params.equation)], params);
}

class SynGaudiDynamicEinsumBE : public SynGaudiDynamicEinsum
{
};

INSTANTIATE_TEST_SUITE_P( , SynGaudiDynamicEinsumBE,
    ::testing::Values("...mk,...kn->...mn", "...ml,...ln->...mn", /*"...ija,aijb...->ba...ij",*/ "...mk,...kn->mn", "...mk,kn->mn", "mk,...kn->mn"));

TEST_P_GC(SynGaudiDynamicEinsumBE, binary_ellipsis)
{
    // Tensor sizes are in reverse, i.e. "abc" {5,4,3} means a=3, b=4, c=5
    std::map<std::string, tensorSizes> pMap = {
    // key                                  in1: Min          Max           Actual    in2: Min              Max              Actual       out: Min          Max          Actual
    // Batch matmul with ellipsis but without broadcasting
    {std::string("...mk,...kn->...mn"),       { {4, 3, 2, 1}, {8, 7, 6, 5}, {6, 5, 4, 3},  {5, 4, 2, 1},    {9, 8, 6, 5},    {7, 6, 4, 3},     {5, 3, 2, 1},    {9, 7, 6, 5},    {7, 5, 4, 3}    } },
    // Empty batch dimensions
    {std::string("...ml,...ln->...mn"),       { {2, 1},       {8, 7},       {4, 3},        {3, 2},          {9, 8},          {5, 4},           {3, 1},          {9, 7},          {5, 3}          } },
    // Tensor contraction with transpose - unsupporded atm, creates internal tensor with rank 6
    {std::string("...ija,aijb...->ba...ij"),  { {5, 4, 3, 2}, {9, 8, 7, 6}, {7, 6, 5, 4},  {2, 1, 4, 3, 5}, {6, 5, 8, 7, 9}, {4, 3, 6, 5, 7},  {4, 3, 2, 5, 1}, {8, 7, 6, 9, 5}, {6, 5, 4, 7, 3} } },
    // Output subscripts may omit ellipsis when batch shape is empty
    {std::string("...mk,...kn->mn"),          { {2, 1},       {8, 7},       {4, 3},        {3, 2},          {9, 8},          {5, 4},           {3, 1},          {9, 7},          {5, 3}          } },
    {std::string("...mk,kn->mn"),             { {2, 1},       {8, 7},       {4, 3},        {3, 2},          {9, 8},          {5, 4},           {3, 1},          {9, 7},          {5, 3}          } },
    {std::string("mk,...kn->mn"),             { {2, 1},       {8, 7},       {4, 3},        {3, 2},          {9, 8},          {5, 4},           {3, 1},          {9, 7},          {5, 3}          } }
    };

    synEinsumParams params(GetParam());

    runTest(pMap[std::string(params.equation)], params);
}
#if 0  // Broadcast is not fully supported yet
class SynGaudiDynamicEinsumBBr : public SynGaudiDynamicEinsum
{
};

INSTANTIATE_TEST_SUITE_P( , SynGaudiDynamicEinsumBBr,
    ::testing::Values("...ab,...bc->...ac", "...bc,...cd->...bd", "...cd,...de->...ce", "...de,...ef->...df", "...ij,...jk->...ik", "i...j,j...k->...ik", "...abc,...abcd->...d", "ab...,b->ab...", "i...j,j...k->i...k"));

TEST_P_GC(SynGaudiDynamicEinsumBBr, binary_broadcast)
{
    // Tensor sizes are in reverse, i.e. "...abc" {5,4,3,2,1} means a=3, b=4, c=5, other=2,1
    std::map<std::string, tensorSizes> pMap = {
    // key                              in1: Min              Max              Actual       in2: Min              Max              Actual       out: Min              Max              Actual
    {std::string("...ab,...bc->...ac"),    { {2, 2, 1},       {9, 9, 1},       {4, 3, 1},        {2, 2},          {9, 9},          {5, 4},           {2, 2, 1},       {9, 9, 1},       {5, 3, 1}       } },  // pass
    {std::string("...bc,...cd->...bd"),    { {2, 2},          {9, 9},          {4, 3},           {2, 2, 1},       {9, 9, 1},       {5, 4, 1},        {2, 2, 1},       {9, 9, 1},       {5, 3, 1}       } },  // pass
    {std::string("...cd,...de->...ce"),    { {2, 2, 2},       {9, 9, 9},       {4, 3, 5},        {2, 2},          {9, 9},          {5, 4},           {2, 2, 2},       {9, 9, 9},       {5, 3, 5}       } },  // pass
    {std::string("...de,...ef->...df"),    { {2, 2},          {9, 9},          {4, 3},           {2, 2, 2},       {9, 9, 9},       {5, 4, 5},        {2, 2, 2},       {9, 9, 9},       {5, 3, 5}       } },  // pass
    {std::string("...ij,...jk->...ik"),    { {4, 3, 1, 6},    {9, 9, 1, 10},   {4, 3, 1, 6},     {5, 4, 7, 1, 1}, {9, 9, 11, 1, 1}, {5, 4, 7, 1, 1}, {5, 3, 7, 6, 1}, {9, 9, 11, 10, 1},  {5, 3, 7, 6, 1} } },  // fail
    {std::string("i...j,j...k->...ik"),    { {2, 2, 2, 2},    {9, 9, 9, 9},    {7, 6, 5, 4},     {2, 2, 2, 2, 2}, {9, 9, 9, 9, 9}, {3, 6, 5, 8, 7},  {2, 2, 2, 2, 2}, {9, 9, 9, 9, 9}, {3, 4, 6, 5, 8} } },  // pass
    {std::string("...abc,...abcd->...d"),  { {2, 2, 2, 1, 1}, {9, 9, 9, 1, 1}, {5, 4, 3, 1, 1},  {2, 2, 2, 2, 2}, {9, 9, 9, 9, 9}, {6, 5, 4, 3, 7},  {2, 2, 1},       {9, 9, 1},       {6, 7, 1}       } },  // fail
    {std::string("ab...,b->ab..."),        { {2, 1, 1, 2, 2}, {9, 1, 1, 9, 9}, {5, 1, 1, 4, 3},  {2},             {9},             {4},              {2, 1, 1, 2, 2}, {9, 1, 1, 9, 9}, {5, 1, 1, 4, 3} } },  // fail
    {std::string("i...j,j...k->i...k"),    { {2, 2, 1, 2},    {9, 9, 1, 9},    {4, 6, 1, 3},     {2, 1, 2, 2, 2}, {9, 1, 9, 9, 9}, {5, 1, 7, 8, 4},  {2, 2, 2, 2, 2}, {9, 9, 9, 9, 9}, {5, 6, 7, 8, 3} } }   // fail
    };

    synEinsumParams params(GetParam());

    runTest(pMap[std::string(params.equation)], params);
}
#endif // 0
