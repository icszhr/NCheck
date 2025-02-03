import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置 TensorFlow 日志级别
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import random
import string

# 设置 TensorFlow 日志级别
tf.get_logger().setLevel('ERROR')

# 模拟 NVM 文件路径
nvm_file_path = "nvm_simulation"

# 创建模拟 NVM 文件
def create_nvm_file(path, size):
    with open(path, "wb") as f:
        f.truncate(size)

# 初始化 NVM 文件大小
nvm_size = 10 * 1024 * 1024  # 10MB
create_nvm_file(nvm_file_path, nvm_size)

# 生成随机文本数据
def generate_random_texts(num_texts, min_length=5, max_length=100):
    texts = []
    for _ in range(num_texts):
        length = random.randint(min_length, max_length)
        text = ''.join(random.choices(string.ascii_lowercase + ' ', k=length))
        texts.append(text)
    return texts

# 生成随机标签
def generate_random_labels(num_labels, num_classes=2):
    return np.random.randint(0, num_classes, size=(num_labels,))

# 加载预训练的 BERT 模型和分词器
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 生成训练数据
texts = generate_random_texts(100, min_length=20, max_length=50)
labels = generate_random_labels(100, num_classes=2)

# 编码文本
train_encodings = tokenizer(texts, truncation=True, padding=True, max_length=50, return_tensors="tf")
input_ids = train_encodings['input_ids']
attention_mask = train_encodings['attention_mask']

# 将标签转换为需要的形状
labels = tf.convert_to_tensor(labels)

# 准备 TensorFlow 数据集
train_dataset = tf.data.Dataset.from_tensor_slices(({
    'input_ids': input_ids,
    'attention_mask': attention_mask
}, labels)).shuffle(len(texts)).batch(8)

# 编译模型
optimizer = Adam(learning_rate=3e-5)
loss_fn = SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# 模拟卸载到NVM的操作
def simulate_unload_to_nvm(tensor, nvm_path):
    if isinstance(tensor, tf.IndexedSlices):
        tensor = tf.convert_to_tensor(tensor)
    tensor_np = tensor.numpy()
    with open(nvm_path.numpy().decode('utf-8'), 'ab') as f:
        f.write(tensor_np.tobytes())
    return np.array(0, dtype=np.int32)  # 返回一个空数组表示成功执行

# 计算张量的内存使用量
def tensor_memory_usage(tensor):
    if isinstance(tensor, tf.IndexedSlices):
        tensor = tf.convert_to_tensor(tensor)
    return tensor.numpy().nbytes

class MemoryManager:
    def __init__(self, dram_cache_limit=3):
        self.dram_cache = []
        self.dram_cache_limit = dram_cache_limit

    def cache_to_dram(self, tensor):
        self.dram_cache.append(tensor)
        if len(self.dram_cache) > self.dram_cache_limit:
            return self.dram_cache.pop(0)  # Remove the oldest tensor if cache limit is exceeded
        return None

    def unload_to_nvm(self, tensor, nvm_path):
        if tensor is not None:
            simulate_unload_to_nvm(tensor, nvm_path)

    def get_dram_usage(self):
        return sum(tensor_memory_usage(tensor) for tensor in self.dram_cache)

memory_manager = MemoryManager()

def train_step(input_ids, attention_mask, labels, nvm_path, memory_logger):
    with tf.GradientTape() as tape:
        # 将输入移动到 GPU
        with tf.device('/GPU:0'):
            input_ids_gpu = tf.identity(input_ids)
            attention_mask_gpu = tf.identity(attention_mask)
            labels_gpu = tf.identity(labels)

            # 前向传播计算
            logits = model(input_ids_gpu, attention_mask=attention_mask_gpu, training=True).logits
            loss = loss_fn(labels_gpu, logits)

        # 计算梯度并应用优化器
        gradients = tape.gradient(loss, model.trainable_variables)

    # 缓存前向传播的激活到DRAM
    activations = model.bert(input_ids_gpu, attention_mask=attention_mask_gpu)[0]
    to_unload = memory_manager.cache_to_dram(activations)
    memory_manager.unload_to_nvm(to_unload, nvm_path)

    # 缓存梯度到DRAM并将多余的卸载到NVM
    for grad in gradients:
        to_unload = memory_manager.cache_to_dram(grad)
        memory_manager.unload_to_nvm(to_unload, nvm_path)

    # 将梯度和权重移动回 GPU 进行更新
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 计算并返回当前批次的 DRAM 使用量
    total_dram_usage = memory_manager.get_dram_usage()
    return loss, total_dram_usage

# 使用 TensorFlow 的内存日志功能记录 GPU 内存使用情况
class MemoryLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MemoryLoggerCallback, self).__init__()
        self.peak_dram_usage = 0
        self.peak_nvm_usage = 0
        self.dram_usage_log = []

    def update_dram_usage(self, usage, context=""):
        self.dram_usage_log.append((context, usage))
        if usage > self.peak_dram_usage:
            self.peak_dram_usage = usage

    def on_train_batch_end(self, batch, dram_usage, logs=None):
        details_gpu = tf.config.experimental.get_memory_info('GPU:0')
        print(f"Batch {batch + 1} - GPU memory usage: {details_gpu['current']} bytes (peak: {details_gpu['peak']} bytes)")

        self.update_dram_usage(dram_usage, f"Batch {batch + 1}")
        print(f"Batch {batch + 1} - Simulated DRAM usage: {dram_usage} bytes")

        nvm_usage = os.path.getsize(nvm_file_path)
        if nvm_usage > self.peak_nvm_usage:
            self.peak_nvm_usage = nvm_usage
        print(f"Batch {batch + 1} - NVM memory usage: {nvm_usage} bytes")

    def on_epoch_end(self, epoch, logs=None):
        details_gpu = tf.config.experimental.get_memory_info('GPU:0')
        print(f"Epoch {epoch + 1} - GPU memory usage: {details_gpu['current']} bytes (peak: {details_gpu['peak']} bytes)")

        print(f"Epoch {epoch + 1} - Peak simulated DRAM usage: {self.peak_dram_usage} bytes")
        print(f"Epoch {epoch + 1} - Peak NVM usage: {self.peak_nvm_usage} bytes")
        print("\nDetailed DRAM usage log:")
        for context, usage in self.dram_usage_log:
            print(f"{context}: {usage} bytes")

# 训练和评估模型
memory_logger = MemoryLoggerCallback()

# 训练模型
nvm_path = tf.constant(nvm_file_path, dtype=tf.string)
for epoch in range(1):
    for batch, (features, labels) in enumerate(train_dataset):
        input_ids = features['input_ids']
        attention_mask = features['attention_mask']
        loss, dram_usage = train_step(input_ids, attention_mask, labels, nvm_path, memory_logger)
        memory_logger.on_train_batch_end(batch, dram_usage)

memory_logger.on_epoch_end(0)
model.evaluate(train_dataset, callbacks=[memory_logger])
