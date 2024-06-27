#ifndef TENSOR_METADATA_COLLECTOR_H_
#define TENSOR_METADATA_COLLECTOR_H_

#include <string>

namespace tensorflow {

class TensorMetadataCollector {
private:
    int tensor_id_;  // 用于跟踪每个张量的唯一ID

public:
    TensorMetadataCollector();

    void SaveTensorMetadata(void* base, void* nptr, const std::string& dtype, const std::string& shape, const std::string& name, int count);
    void* LoadTensor(void* base, const std::string& name, int count);
private:
    std::string CreateTensorMetadata(const std::string& dtype, const std::string& shape, const std::string& name, int tensor_id, void* nptr, int count);
};

}  // namespace tensorflow

#endif  // TENSOR_METADATA_COLLECTOR_H_
