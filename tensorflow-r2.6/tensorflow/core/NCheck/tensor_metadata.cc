#include "tensor_metadata.h"
#include "persist.h"  

namespace tensorflow {

TensorMetadataCollector::TensorMetadataCollector() : tensor_id_(0) {}

void TensorMetadataCollector::SaveTensorMetadata(void* base, void* nptr, const std::string& dtype, const std::string& shape, const std::string& name) {
    int tensor_id = ++tensor_id_;  
    std::string metadata = CreateTensorMetadata(dtype, shape, name, tensor_id, nptr);
    PersistenceManager::SaveTensorMetadata(base, metadata.data(), metadata.size());
}

std::string TensorMetadataCollector::CreateTensorMetadata(const std::string& dtype, const std::string& shape, const std::string& name, int tensor_id, void* nptr) {
    return std::to_string(tensor_id) + "|" + name + "|" + dtype + "|" + shape+ "|" + std::to_string(size_t(nptr));
}

}  // namespace tensorflow
