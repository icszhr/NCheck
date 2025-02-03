import re

# 示例日志字符串，实际情况中这些数据将从文件中读取
log_data = []
file_path = "demo1"  # 文件路径

# 打开并读取文件内容
with open(file_path, 'r') as file:
    log_data = file.readlines()

# 正则表达式用于匹配GPU的内存行为日志
gpu_fragment_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}): I tensorflow/core/common_runtime/bfc_allocator.cc:415] GPU, allocated: (\d+), peak_memory: (\d+), reserved: (\d+), GPU fragment: ([\d.]+)")
data_transfer_pattern = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}): I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:\d+] (GPU2CPU_C|CPU2GPU_C): (\d+)B"
)


# 用于存储解析结果
gpu_fragments = []
data_transfers = []

# 分行处理日志数据
with open(file_path, 'r') as file:
    for line in file:
        # 检查是否匹配GPU内存碎片的日志
        match = gpu_fragment_pattern.search(line)
        if match:
            timestamp, allocated, peak_memory, reserved, fragment = match.groups()
            gpu_fragments.append({
                "timestamp": timestamp,
                "allocated": int(allocated),
                "peak_memory": int(peak_memory),
                "reserved": int(reserved),
                "fragment": float(fragment)
            })
        
        # 检查是否匹配数据传输的日志
        match = data_transfer_pattern.search(line)
        if match:
            timestamp, direction, bytes_transferred = match.groups()
            data_transfers.append({
                "timestamp": timestamp,
                "direction": direction,
                "bytes": int(bytes_transferred)
            })

# 展示收集到的数据
print(data_transfers)
#print(gpu_fragments)
def get_data():
    return gpu_fragments, data_transfers
