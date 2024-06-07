#include "tensorflow/core/NCheck/tensor_manager.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"

namespace tensorflow {

Status TensorManager::TransferTensorToDevice(const Tensor& tensor, Device* device) {
    auto status = device->CopyTensorFromHostToDevice(tensor, tensor, nullptr);
    if (status.ok()) {
        categorizer_.UpdateTensorStatus(&tensor, TensorStatus::GPU);
    }
    return status;
}

Status TensorManager::TransferTensorToHost(const Tensor& tensor, Device* device) {
    auto status = device->CopyTensorFromDeviceToHost(tensor, tensor, nullptr);
    if (status.ok()) {
        categorizer_.UpdateTensorStatus(&tensor, TensorStatus::DRAM);
    }
    return status;
}

Status TensorManager::DirectPersistentAccess(const Tensor& tensor, Device* gpu_device) {
    // This method assumes that the tensor is already located in NVM and is accessible by the GPU through a direct access path
    TensorAttributes attributes = categorizer_.GetTensorAttributes(&tensor);
    if (attributes.status != TensorStatus::NVM) {
        return Status(tensorflow::error::INTERNAL, "Tensor is not in NVM for Direct Persistent Access.");
    }

    // Depending on the GPU and NVM setup, this might involve specific API calls or configurations
    // to enable direct access. Here, we assume that such configurations are already in place.
    // No actual data transfer is performed. The operation is assumed to be handled by the hardware
    // and driver configurations that allow the GPU to access NVM directly.

    return Status::OK();
}

}  // namespace tensorflow
