#ifndef TENSORFLOW_CORE_NCHECK_MODEL_METADATA_H_
#define TENSORFLOW_CORE_NCHECK_MODEL_METADATA_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"


namespace tensorflow {

class ModelMetadataCollector {
public:
    void CollectAndSaveMetadata(tensorflow::OpKernelContext* context);

private:
    std::string CreateOpMetadata(const tensorflow::Node* node);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_NCHECK_MODEL_METADATA_H_
