#include "node_access_cache_setter.h"

#include "../gaudi2/layered_brain_test.h"
#include "scoped_configuration_change.h"

using namespace gc::layered_brain;

class NodeAccessCacheTest : public LayeredBrainTest
{
public:
    NodePtr createNode(unsigned numInputs, unsigned numOutputs)
    {
        TensorVector inputs;
        for (unsigned i = 0; i < numInputs; i++)
            inputs.push_back(newTensor());

        TensorVector outputs;
        for (unsigned i = 0; i < numOutputs; i++)
            outputs.push_back(newTensor());

        NodePtr n = NodeFactory::createNode(inputs, outputs, nullptr, TPCNode::NOP_KERNEL_NAME, "nop");

        n->getNodeAnnotation().inputsCacheMetaData.resize(numInputs, defaultMetadataForTest());
        n->getNodeAnnotation().outputsCacheMetaData.resize(numOutputs, defaultMetadataForTest());
        return n;
    }

    static CacheMetaData defaultMetadataForTest()
    {
        return CacheMetaData {CacheDirective::NoAllocate, CacheMaintenanceAction::NOP, LogicalMcid {}, CacheClass::Low};
    }

    //
    // Abbreviation: CMD = Cache Meta Data
    //

    void validateDirectives(const NodePtr&                               n,
                            const std::initializer_list<CacheDirective>& expInputDirectives,
                            const std::initializer_list<CacheDirective>& expOutputDirectives) const
    {
        std::function<CacheDirective(const CacheMetaData&)> getter = [](const CacheMetaData& cmd) {
            return cmd.cacheDirective;
        };
        validateCMDVecField(n->getNodeAnnotation().inputsCacheMetaData, getter, expInputDirectives, "input");
        validateCMDVecField(n->getNodeAnnotation().outputsCacheMetaData, getter, expOutputDirectives, "output");
    }

    void validateClasses(const NodePtr&                           n,
                         const std::initializer_list<CacheClass>& expInputClasses,
                         const std::initializer_list<CacheClass>& expOutputClasses) const
    {
        std::function<CacheClass(const CacheMetaData&)> getter = [](const CacheMetaData& cmd) {
            return cmd.cacheClass;
        };
        validateCMDVecField(n->getNodeAnnotation().inputsCacheMetaData, getter, expInputClasses, "input");
        validateCMDVecField(n->getNodeAnnotation().outputsCacheMetaData, getter, expOutputClasses, "output");
    }

    void validateCMActions(const NodePtr&                                       n,
                           const std::initializer_list<CacheMaintenanceAction>& expInputCMEs,
                           const std::initializer_list<CacheMaintenanceAction>& expOutputCMEs) const
    {
        std::function<CacheMaintenanceAction(const CacheMetaData&)> getter = [](const CacheMetaData& cmd) {
            return cmd.cmAction;
        };
        validateCMDVecField(n->getNodeAnnotation().inputsCacheMetaData, getter, expInputCMEs, "input");
        validateCMDVecField(n->getNodeAnnotation().outputsCacheMetaData, getter, expOutputCMEs, "output");
    }

    void validateMCIDs(const NodePtr&                            n,
                       const std::initializer_list<LogicalMcid>& expInputMCIDs,
                       const std::initializer_list<LogicalMcid>& expOutputMCIDs) const
    {
        std::function<LogicalMcid(const CacheMetaData&)> getter = [](const CacheMetaData& cmd) { return cmd.mcid; };
        validateCMDVecField(n->getNodeAnnotation().inputsCacheMetaData, getter, expInputMCIDs, "input");
        validateCMDVecField(n->getNodeAnnotation().outputsCacheMetaData, getter, expOutputMCIDs, "output");
    }

    template<typename FieldType>
    void validateCMDVecField(const std::vector<CacheMetaData>&              cmds,
                             std::function<FieldType(const CacheMetaData&)> getter,
                             const std::initializer_list<FieldType>&        fieldVals,
                             const std::string_view&                        inputOrOutput) const
    {
        size_t cmdIdx = 0;
        ASSERT_EQ(cmds.size(), fieldVals.size());
        for (const auto& val : fieldVals)
        {
            EXPECT_EQ(val, getter(cmds[cmdIdx++])) << "Failure at " << inputOrOutput << " cmd " << cmdIdx - 1;
        }
    }
};

TEST_F(NodeAccessCacheTest, nacs_should_allow_setting_input_access_directive)
{
    NodePtr n = createNode(3, 4);

    NodeAccessCacheSetter nacs(n);

    nacs.input(0).directive(HomeAllocate).set();
    nacs.input(2).set();

    validateDirectives(n,
                       {
                           // Expected Inputs directives
                           HomeAllocate,
                           NoAllocate,
                           HomeAllocate,
                       },
                       {
                           // Expected outputs directives
                           NoAllocate,
                           NoAllocate,
                           NoAllocate,
                           NoAllocate,
                       });
}

TEST_F(NodeAccessCacheTest, nacs_should_allow_setting_output_access_directive)
{
    NodePtr n = createNode(3, 4);

    NodeAccessCacheSetter nacs(n);

    nacs.output(1).directive(HomeAllocate).set();
    nacs.output(0).set();
    validateDirectives(n,
                       {
                           // Expected Inputs directives
                           NoAllocate,
                           NoAllocate,
                           NoAllocate,
                       },
                       {
                           // Expected outputs directives
                           HomeAllocate,
                           HomeAllocate,
                           NoAllocate,
                           NoAllocate,
                       });
}

TEST_F(NodeAccessCacheTest, nacs_should_allow_setting_input_access_class)
{
    NodePtr n = createNode(3, 4);

    NodeAccessCacheSetter nacs(n);

    nacs.input(0).cacheClass(CacheClass::Top).set();
    nacs.input(2).set();

    validateClasses(n,
                    {
                        // Expected Inputs classes
                        CacheClass::Top,
                        CacheClass::Low,  // This is the cache metadata default in the test
                        CacheClass::Top,
                    },
                    {
                        // Expected outputs classes
                        CacheClass::Low,
                        CacheClass::Low,
                        CacheClass::Low,
                        CacheClass::Low,
                    });
}

TEST_F(NodeAccessCacheTest, nacs_should_allow_setting_output_access_class)
{
    NodePtr n = createNode(3, 4);

    NodeAccessCacheSetter nacs(n);

    nacs.output(1).cacheClass(CacheClass::Normal).set();
    nacs.output(0).set();
    validateClasses(n,
                    {
                        // Expected Inputs classes
                        CacheClass::Low,
                        CacheClass::Low,
                        CacheClass::Low,
                    },
                    {
                        // Expected outputs classes
                        CacheClass::Normal,
                        CacheClass::Normal,
                        CacheClass::Low,
                        CacheClass::Low,
                    });
}

TEST_F(NodeAccessCacheTest, nacs_should_allow_setting_input_cme)
{
    NodePtr n = createNode(3, 4);

    NodeAccessCacheSetter nacs(n);

    nacs.input(0).cmAction(CacheMaintenanceAction::DISCARD).set();
    nacs.input(2).set();

    validateCMActions(n,
                      {
                          // Expected Inputs classes
                          CacheMaintenanceAction::DISCARD,
                          CacheMaintenanceAction::NOP,  // This is the cache metadata default in the test
                          CacheMaintenanceAction::DISCARD,
                      },
                      {
                          // Expected outputs classes
                          CacheMaintenanceAction::NOP,
                          CacheMaintenanceAction::NOP,
                          CacheMaintenanceAction::NOP,
                          CacheMaintenanceAction::NOP,
                      });
}

TEST_F(NodeAccessCacheTest, nacs_should_allow_setting_output_cme)
{
    NodePtr n = createNode(3, 4);

    NodeAccessCacheSetter nacs(n);

    nacs.output(1).cmAction(CacheMaintenanceAction::DEGRADE).set();
    nacs.output(0).set();
    validateCMActions(n,
                      {
                          // Expected Inputs classes
                          CacheMaintenanceAction::NOP,
                          CacheMaintenanceAction::NOP,
                          CacheMaintenanceAction::NOP,
                      },
                      {
                          // Expected outputs classes
                          CacheMaintenanceAction::DEGRADE,
                          CacheMaintenanceAction::DEGRADE,
                          CacheMaintenanceAction::NOP,
                          CacheMaintenanceAction::NOP,
                      });
}

TEST_F(NodeAccessCacheTest, nacs_should_allow_setting_input_mcid)
{
    NodePtr n = createNode(3, 4);

    NodeAccessCacheSetter nacs(n);

    nacs.input(0).mcid(82).set();
    nacs.input(2).mcid(33).set();

    validateMCIDs(n, {82, 0, 33}, {0, 0, 0, 0});
}

TEST_F(NodeAccessCacheTest, nacs_should_allow_setting_output_mcid)
{
    NodePtr n = createNode(3, 4);

    NodeAccessCacheSetter nacs(n);

    nacs.output(1).mcid(75).set();
    nacs.output(0).mcid(91).set();

    validateMCIDs(n, {0, 0, 0}, {91, 75, 0, 0});
}