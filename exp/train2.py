import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, BertTokenizer, TFBertModel, RobertaTokenizer, TFRobertaModel, DistilBertTokenizer, TFDistilBertModel
import random
import string
import os
import subprocess
import time

# 设置 TensorFlow 日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# 禁用统一内存
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, False)

# 定义生成随机文本的函数
def generate_random_text(min_length=5, max_length=1000):
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(string.ascii_lowercase, k=length))

# 生成训练数据
def generate_training_data(tokenizer, min_length=5, max_length=10, batch_size=32):
    texts = [generate_random_text(min_length, max_length) for _ in range(batch_size)]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="tf")
    labels = inputs["input_ids"]
    return inputs, labels

# 编译模型
def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn)
    return model

# 获取 GPU 内存使用情况
def get_gpu_memory():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
    return int(result.decode('utf-8').strip())

# 模型配置
models_config = [
    ("GPT-2", GPT2Tokenizer, TFGPT2LMHeadModel, "gpt2"),
    ("BERT", BertTokenizer, TFBertModel, "bert-base-uncased"),
    ("RoBERTa", RobertaTokenizer, TFRobertaModel, "roberta-base"),
    ("DistilBERT", DistilBertTokenizer, TFDistilBertModel, "distilbert-base-uncased")
]

def benchmark_model(model_name, tokenizer_class, model_class, pretrained_model_name, batch_size=1):
    print(f"Running benchmark for {model_name}...")
    
    # 初始化 tokenizer 和 model
    tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
    model = model_class.from_pretrained(pretrained_model_name)

    # 添加 padding token 如果没有的话
    if model_name in ["BERT", "DistilBERT"]:
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer.pad_token = tokenizer.eos_token

    model = compile_model(model)

    # 生成一个批次的训练数据
    train_inputs, train_labels = generate_training_data(tokenizer, batch_size=batch_size)
    train_inputs = {key: tf.constant(value) for key, value in train_inputs.items()}
    train_labels = tf.constant(train_labels)

    # 记录训练过程中的 GPU 内存使用峰值
    peak_memory_usage = 0

    def update_peak_memory():
        nonlocal peak_memory_usage
        current_memory_usage = get_gpu_memory()
        peak_memory_usage = max(peak_memory_usage, current_memory_usage)

    # 在训练过程中定期更新内存使用情况
    update_interval = 0.1  # 检查内存使用情况的时间间隔（秒）

    def train_step():
        nonlocal train_inputs, train_labels
        model.train_on_batch(train_inputs, train_labels)
        update_peak_memory()

    # 启动训练，并在过程中定期检查内存使用情况
    for _ in range(int(1 / update_interval)):  # 假设训练过程持续 1 秒
        train_step()
        time.sleep(update_interval)

    # 获取 GPU 和 DRAM 的总内存
    total_memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'])
    total_memory = int(total_memory.decode('utf-8').strip())
    dram_memory_allocated = total_memory - peak_memory_usage

    return peak_memory_usage, dram_memory_allocated

# 运行基准测试并汇总结果
results = []

for model_config in models_config:
    model_name, tokenizer_class, model_class, pretrained_model_name = model_config
    gpu_memory_allocated, dram_memory_allocated = benchmark_model(model_name, tokenizer_class, model_class, pretrained_model_name)
    results.append({
        "model_name": model_name,
        "gpu_memory_allocated": gpu_memory_allocated,
        "dram_memory_allocated": dram_memory_allocated
    })

# 输出结果
for result in results:
    print(f"Model: {result['model_name']}")
    print(f"Peak GPU Memory Allocated: {result['gpu_memory_allocated']} MB")
    print(f"DRAM Memory Allocated: {result['dram_memory_allocated']} MB\n")
