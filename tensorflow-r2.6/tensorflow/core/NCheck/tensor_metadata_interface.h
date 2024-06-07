#ifndef TENSORFLOW_CORE_NCHECK_METADATA_COLLECTOR_INTERFACE_H_
#define TENSORFLOW_CORE_NCHECK_METADATA_COLLECTOR_INTERFACE_H_

#include <string>

namespace tensorflow {



class IMetadataCollector {
public:
    virtual ~IMetadataCollector() = default;
    virtual void SaveTensorMetadata(const std::string& dtype, const std::string& shape, int tensor_id) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_NCHECK_METADATA_COLLECTOR_INTERFACE_H_
