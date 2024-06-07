#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/session_state.h"
#include "tensorflow/core/NCheck/persist.h"
#include "tensorflow/core/graph/graph.h"
#include <string>
#include <vector>

class ModelMetadataCollector {
public:
    void CollectAndSaveMetadata(tensorflow::OpKernelContext* context) {
        const tensorflow::Graph& graph = context->session_state()->graph();
        std::vector<std::string> op_metadata;
        for (const auto& node : graph.nodes()) {
            std::string metadata = CreateOpMetadata(node);
            op_metadata.push_back(metadata);
            PersistenceManager::SaveTensorMetadata(metadata.data(), metadata.size());
        }

        std::string combined_metadata;
        for (const auto& meta : op_metadata) {
            combined_metadata += meta + "\n";
        }
        PersistenceManager::SaveModelState(combined_metadata.data(), combined_metadata.size());
    }

private:
    std::string CreateOpMetadata(const tensorflow::Node* node) {
        std::string op_type = node->type_string();
        std::string op_name = node->name();
        std::string inputs;
        for (const auto& edge : node->in_edges()) {
            if (!edge->IsControlEdge()) {
                inputs += edge->src()->name() + " ";
            }
        }
        return op_name + "|" + op_type + "|" + inputs;
    }
};
