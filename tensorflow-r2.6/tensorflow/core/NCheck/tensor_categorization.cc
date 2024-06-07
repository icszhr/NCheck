#include "tensorflow/core/NCheck/tensor_categorization.h"

namespace tensorflow {

void TensorCategorizer::RegisterTensor(const Tensor* tensor, float priority, bool persistence, TensorStatus status) {
    TensorAttributes attributes {priority, persistence, status};
    tensor_attributes_map_[tensor] = attributes;
}

void TensorCategorizer::UpdateTensorStatus(const Tensor* tensor, TensorStatus new_status) {
    auto it = tensor_attributes_map_.find(tensor);
    if (it != tensor_attributes_map_.end()) {
        it->second.status = new_status;
    }
}

TensorAttributes TensorCategorizer::GetTensorAttributes(const Tensor* tensor) {
    auto it = tensor_attributes_map_.find(tensor);
    if (it != tensor_attributes_map_.end()) {
        return it->second;
    }
    return TensorAttributes{};  // Return default if not found
}

}  // namespace tensorflow
