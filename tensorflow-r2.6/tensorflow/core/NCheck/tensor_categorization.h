#ifndef TENSORFLOW_CORE_UTIL_TENSOR_CATEGORIZATION_H_
#define TENSORFLOW_CORE_UTIL_TENSOR_CATEGORIZATION_H_

#include "tensorflow/core/framework/tensor.h"
#include <unordered_map>
#include <memory>

namespace tensorflow {

enum class TensorStatus {
    GPU,
    NVM,
    DRAM
};

struct TensorAttributes {
    float priority;
    bool persistence;
    TensorStatus status;
};

class TensorCategorizer {
 public:
    TensorCategorizer() = default;
    ~TensorCategorizer() = default;

    void RegisterTensor(const Tensor* tensor, float priority, bool persistence, TensorStatus status);
    void UpdateTensorStatus(const Tensor* tensor, TensorStatus new_status);
    TensorAttributes GetTensorAttributes(const Tensor* tensor);

 private:
    std::unordered_map<const Tensor*, TensorAttributes> tensor_attributes_map_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_TENSOR_CATEGORIZATION_H_
