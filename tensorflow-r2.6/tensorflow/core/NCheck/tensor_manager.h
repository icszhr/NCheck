#ifndef TENSORFLOW_CORE_UTIL_TENSOR_MANAGER_H_
#define TENSORFLOW_CORE_UTIL_TENSOR_MANAGER_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_categorization.h"

namespace tensorflow {

class TensorManager {
 public:
    TensorManager() = default;
    ~TensorManager() = default;

    // Methods to handle tensor transfers
    Status TransferTensorToDevice(const Tensor& tensor, Device* device);
    Status TransferTensorToHost(const Tensor& tensor, Device* device);
    Status DirectPersistentAccess(const Tensor& tensor, Device* gpu_device);

 private:
    TensorCategorizer categorizer_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_TENSOR_MANAGER_H_
