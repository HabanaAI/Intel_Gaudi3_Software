#pragma once

#include "hpp/syn_context.hpp"
#include "hpp/syn_graph.hpp"
#include "json.hpp"
#include "synapse_common_types.h"
#include <memory>
#include <string>
#include <optional>
#include "compiler_types.h"

class JsonGraphLoader
{
public:
    JsonGraphLoader(syn::Context&                  ctx,
                    synDeviceType                  deviceType,
                    std::optional<CompilationMode> compilationMode,
                    const nlohmann_hcl::json&      jsonGraph,
                    const std::string&             constTensorsFilePath);
    virtual ~JsonGraphLoader();

    virtual uint64_t                                  getGroup() const;
    virtual std::string                               getName() const;
    virtual uint16_t                                  getRecipeId() const;
    virtual bool                                      isEager() const;
    virtual const syn::GraphBase&                     getGraph() const;
    virtual const std::map<std::string, syn::Tensor>& getTensors() const;
    virtual nlohmann_hcl::json                        getConfig() const;
    virtual nlohmann_hcl::json                        getGraphAttributes() const;
    virtual const nlohmann_hcl::json&                 getJsonGraph() const;

    static synDataType          dataTypeFromString(const std::string& str);
    static synTensorType        tensorTypeFromString(const std::string& str);
    static std::vector<uint8_t> decompress(const std::vector<uint8_t>& src, int dstSize);

private:
    syn::Section
    getSection(std::vector<unsigned>& sectionsIndices, bool isPersistent, bool isRmwSection, bool isConstSection);
    syn::Section createSection(bool isPersistent, bool isRmwSection, bool isConstSection);

    // The graph content is accessed by a reference,
    // therefore it should not be released before JsonGraphLoader is released.
    const nlohmann_hcl::json& m_jsonGraph;

    uint32_t                        m_fileVersion;
    std::unique_ptr<syn::GraphBase> m_graph;
    std::string                     m_constTensorsFilePath;

    std::map<std::string, syn::Tensor> m_tensors;
    std::map<unsigned, syn::Section>   m_persistentSectionsMap;
    std::map<unsigned, syn::Section>   m_nonPersistentSectionsMap;
    virtual void                       allocateTensors();
    virtual void                       generateModel();
    virtual void                       loadGraphAttributes();
    virtual void                       loadTensorQuantParams(const nlohmann_hcl::json& t, syn::Tensor& tensor);
};