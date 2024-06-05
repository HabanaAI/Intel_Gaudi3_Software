#pragma once

#include "synapse_types.h"
#include "layout.h"

class Node;
class TPCNode;
class LogicalOpNode;
using gc::Layout;
/**
 * Node Inputs/Outputs Manager
 *
 * Handle node's IO with relation to the node's requirements.
 * Right now, only handles IO layouts and node's parameters due to layout changes.
 *
 */
class NodeIOManager
{
public:
    NodeIOManager(Node* node);
    virtual ~NodeIOManager();

    void                 setDefaultIOLayouts();
    virtual bool         setSupportedIOLayouts(synDeviceType deviceType);
    virtual LayoutVector getInputInferLayouts(const LayoutVector& outputLayouts, synDeviceType deviceType);

    const LayoutVector& getInputSupportedLayouts() const;
    const LayoutVector& getOutputSupportedLayouts() const;

    bool isInputRestrictedAtIndex(unsigned index) const { return m_supportedInputLayouts[index].isRestricted(); }
    bool isOutputRestrictedAtIndex(unsigned index) const { return m_supportedOutputLayouts[index].isRestricted(); }
    void setInputRestrictedAtIndex(unsigned index) { m_supportedInputLayouts[index].setAsRestricted(); }
    void setOutputRestrictedAtIndex(unsigned index) { m_supportedOutputLayouts[index].setAsRestricted(); }

    virtual void permute(PermutationVector& inputPermutations, PermutationVector& outputPermutations) const;

    bool validateAndSetActualIOLayouts();
    bool nodeisAllDontCare() const;
    bool permutationsRequired() const;
    bool validateLayouts() const;

    virtual void setSupportedLayouts(const LayoutVector& supportedInputLayouts,
                                     const LayoutVector& supportedOutputLayouts);

    bool isAllDontCare() const { return m_allDontCare; }
    void markAdjusted() { m_isAdjusted = true; }
    bool isAdjusted() const { return m_isAdjusted; }

protected:
    Node* m_node;

    void setSupportedLayoutsHelper(LayoutVector&       inputLayouts,
                                   LayoutVector&       outputLayouts,
                                   const LayoutVector& supportedInputLayouts,
                                   const LayoutVector& supportedOutputLayouts) const;

    void setSupportedLayoutsConv(LayoutVector&       inputLayouts,
                                 LayoutVector&       outputLayouts,
                                 const LayoutVector& supportedInputLayouts,
                                 const LayoutVector& supportedOutputLayouts,
                                 const LayoutVector& supported3DInputLayouts,
                                 const LayoutVector& supported3DOutputLayouts) const;

private:
    void permuteInternal(const LayoutVector& fromLayouts,
                         const LayoutVector& toLayouts,
                         PermutationVector&  permutations) const;
    bool validateAllRequiredLayoutsExist(const LayoutVector& supportedLayouts, const LayoutVector& userLayouts) const;
    bool permutationRequired(const LayoutVector& supportedLayouts, const LayoutVector& userLayouts) const;
    void selectLayoutsByType(LayoutVector& inputLayouts, LayoutVector& outputLayouts) const;
    void selectLayoutsConvNode(LayoutVector& inputLayouts, LayoutVector& outputLayouts) const;
    void selectLayoutsDedwNode(LayoutVector& inputLayouts, LayoutVector& outputLayouts) const;
    void selectLayoutsDedxNode(LayoutVector& inputLayouts, LayoutVector& outputLayouts) const;
    bool isDontCareNode() const;

    bool         m_allDontCare = true;
    bool         m_isAdjusted  = false;
    LayoutVector m_supportedInputLayouts;
    LayoutVector m_supportedOutputLayouts;
};

/**
 * TPC Node Inputs/Outputs Manager
 *
 * Specific implementation for TPC nodes where GUID should be taken into consideration.
 *
 */
class TPCNodeIOManager : public NodeIOManager
{
public:
    TPCNodeIOManager(TPCNode* node);

    virtual bool setSupportedIOLayouts(synDeviceType deviceType) override;

    virtual LayoutVector getInputInferLayouts(const LayoutVector& outputLayouts, synDeviceType deviceType) override;

private:
    bool setGuidSupportedLayouts(const TPCNode& node, synDeviceType deviceType);
};

/**
 * Logical Node Inputs/Outputs Manager
 *
 * Specific implementation for logical nodes where there is an internal parameter to handle.
 *
 */
class LogicalNodeIOManager : public NodeIOManager
{
public:
    LogicalNodeIOManager(LogicalOpNode* node);

    virtual void permute(PermutationVector& inputPermutations, PermutationVector& outputPermutations) const override;
};