#pragma once
#include <types.h>
#include "habana_graph.h"

class RolePattern
{
public:
    RolePattern();
    virtual ~RolePattern();
    GraphPtr                     getPattern();
    virtual bool                 isSymmetricRole(const pNode n);
    virtual bool                 rolesMatch(const HabanaGraph& g, const pNode& a, const pNode& b) const = 0;
    virtual std::pair<int, int>  numInputsRange(const pNode& n) const;


protected:
    template<class T>
    bool addNode(const TensorVector& inputs,
                 const TensorVector& outputs,
                 const T*            userParams,
                 const char*         guid,
                 const std::string&  name);

    GraphPtr m_pattern;
};

typedef std::vector<std::pair<NodeList, RolePattern*>> NodesPatternVector;
typedef std::vector<RolePattern*> RolePatternVector;

NodesPatternVector matchMultiplePatternsWithSingleOutputNode(HabanaGraph& g, const RolePatternVector& patterns);
