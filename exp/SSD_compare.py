import os
import tensorflow as tf
import time
import mmap
import pickle
from transformers import TFAutoModelForSequenceClassification, TFGPT2Model, AutoTokenizer

# 设置环境变量以减少日志输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 配置GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 模型和数据集名称
model_names = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]
gpt2_name = "gpt2"

# 创建随机数据集
def create_random_dataset(batch_size, seq_length, num_batches, num_labels=2, is_gpt2=False):
    input_ids = tf.random.uniform([batch_size, seq_length], minval=0, maxval=30522, dtype=tf.int32)
    attention_mask = tf.ones([batch_size, seq_length], dtype=tf.int32)
    if is_gpt2:
        labels = tf.random.uniform([batch_size, seq_length], minval=0, maxval=30522, dtype=tf.int32)
    else:
        labels = tf.random.uniform([batch_size], minval=0, maxval=num_labels, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_mask}, labels))
    return dataset.batch(batch_size).repeat(num_batches)

# 检查点保存路径
ssd_checkpoint_dir = "./checkpoints"
nvm_checkpoint_file = "/pmem/zhr/NCheck"

# 训练配置
batch_size = 8
seq_length = 128
num_batches = 100

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 确保NVM文件存在并且大小足够大
mapped_len = 1024 * 1024 * 1024  # 1GB
if not os.path.exists(nvm_checkpoint_file):
    with open(nvm_checkpoint_file, 'wb') as f:
        f.write(b'\x00' * mapped_len)
else:
    file_size = os.path.getsize(nvm_checkpoint_file)
    if file_size < mapped_len:
        with open(nvm_checkpoint_file, 'ab') as f:
            f.write(b'\x00' * (mapped_len - file_size))

# 自定义回调函数，用于SSD检查点保存并测量序列化开销
class SSDCheckpointCallback(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, **kwargs):
        super(SSDCheckpointCallback, self).__init__(filepath, **kwargs)
        self.total_serialize_time = 0
        self.total_checkpoint_time = 0

    def on_epoch_end(self, epoch, logs=None):
        start_time = time.time()
        weights = self.model.get_weights()
        serialize_start_time = time.time()
        data = pickle.dumps(weights)
        serialize_time = time.time() - serialize_start_time
        self.total_serialize_time += serialize_time
        super(SSDCheckpointCallback, self).on_epoch_end(epoch, logs)
        checkpoint_time = time.time() - start_time
        self.total_checkpoint_time += checkpoint_time
        print(f"Epoch {epoch + 1} SSD serialization time: {serialize_time} seconds")
        print(f"Epoch {epoch + 1} SSD checkpoint time: {checkpoint_time} seconds")

# 自定义回调函数，用于NVM检查点保存
class NVMCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, file_path, mapped_len):
        super(NVMCheckpointCallback, self).__init__()
        self.file_path = file_path
        self.mapped_len = mapped_len
        self.mmap_file = None

    def on_train_begin(self, logs=None):
        with open(self.file_path, "r+b") as f:
            self.mmap_file = mmap.mmap(f.fileno(), self.mapped_len)

    def on_epoch_end(self, epoch, logs=None):
        start_time = time.time()
        weights = self.model.get_weights()
        data = pickle.dumps(weights)
        data_len = len(data)
        if data_len > self.mapped_len:
            raise ValueError("Data size exceeds mapped NVM region size")
        self.mmap_file.seek(0)
        self.mmap_file.write(data)
        self.mmap_file.flush()
        checkpoint_time = time.time() - start_time
        print(f"Epoch {epoch + 1} NVM checkpoint time: {checkpoint_time} seconds")

    def on_train_end(self, logs=None):
        if self.mmap_file is not None:
            self.mmap_file.close()

# 创建分类模型和数据集
for model_name in model_names:
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    dataset = create_random_dataset(batch_size, seq_length, num_batches)

    # 配置优化器和损失函数
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    # SSD 检查点保存并测量序列化开销
    ssd_checkpoint_callback = SSDCheckpointCallback(
        filepath=os.path.join(ssd_checkpoint_dir, model_name, 'ckpt-{epoch}'),
        save_weights_only=True,
        save_freq='epoch'
    )

    # NVM 检查点保存
    nvm_checkpoint_callback = NVMCheckpointCallback(nvm_checkpoint_file, mapped_len)

    # 训练并测量时间
    print(f"Training {model_name} with SSD checkpointing...")
    model.fit(dataset, epochs=3, steps_per_epoch=num_batches, callbacks=[ssd_checkpoint_callback])
    print(f"Total SSD serialization time for {model_name}: {ssd_checkpoint_callback.total_serialize_time} seconds")
    print(f"Total SSD checkpoint time for {model_name}: {ssd_checkpoint_callback.total_checkpoint_time} seconds")

    print(f"Training {model_name} with NVM checkpointing...")
    model.fit(dataset, epochs=3, steps_per_epoch=num_batches, callbacks=[nvm_checkpoint_callback])

# 创建GPT-2模型和数据集
model = TFGPT2Model.from_pretrained(gpt2_name)
dataset = create_random_dataset(batch_size, seq_length, num_batches, is_gpt2=True)

# 配置优化器和损失函数
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

# SSD 检查点保存并测量序列化开销
ssd_checkpoint_callback = SSDCheckpointCallback(
    filepath=os.path.join(ssd_checkpoint_dir, gpt2_name, 'ckpt-{epoch}'),
    save_weights_only=True,
    save_freq='epoch'
)

# NVM 检查点保存
nvm_checkpoint_callback = NVMCheckpointCallback(nvm_checkpoint_file, mapped_len)

# 训练并测量时间
print(f"Training {gpt2_name} with SSD checkpointing...")
model.fit(dataset, epochs=3, steps_per_epoch=num_batches, callbacks=[ssd_checkpoint_callback])
print(f"Total SSD serialization time for {gpt2_name}: {ssd_checkpoint_callback.total_serialize_time} seconds")
print(f"Total SSD checkpoint time for {gpt2_name}: {ssd_checkpoint_callback.total_checkpoint_time} seconds")

print(f"Training {gpt2_name} with NVM checkpointing...")
model.fit(dataset, epochs=3, steps_per_epoch=num_batches, callbacks=[nvm_checkpoint_callback])
