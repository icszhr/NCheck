import os
import time
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import random
import string

# 设置 TensorFlow 日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# 生成随机文本数据
def generate_random_texts(num_texts, min_length=5, max_length=100):
    texts = []
    for _ in range(num_texts):
        length = random.randint(min_length, max_length)
        text = ''.join(random.choices(string.ascii_lowercase + ' ', k=length))
        texts.append(text)
    return texts

# 生成训练数据
texts = generate_random_texts(100, min_length=20, max_length=50)

# 加载预训练的 GPT-2 模型和分词器
model = TFGPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 添加填充标记
tokenizer.pad_token = tokenizer.eos_token

# 编码文本
train_encodings = tokenizer(texts, truncation=True, padding=True, max_length=50, return_tensors="tf")
input_ids = train_encodings['input_ids']

# 准备 TensorFlow 数据集
train_dataset = tf.data.Dataset.from_tensor_slices((input_ids, input_ids)).shuffle(len(texts)).batch(8)

# 编译模型
optimizer = Adam(learning_rate=3e-5)
loss_fn = SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TimingCallback, self).__init__()
        self.log_file = open('timing_log.txt', 'w')

    def on_train_begin(self, logs=None):
        tf.profiler.experimental.start('logdir')

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()
        self.forward_start = None
        self.backward_start = None
        self.update_start = None

    def on_train_batch_end(self, batch, logs=None):
        batch_end_time = time.time()
        total_batch_time = batch_end_time - self.batch_start_time

        forward_time = self.forward_end - self.forward_start if self.forward_start and self.forward_end else 0
        backward_time = self.backward_end - self.backward_start if self.backward_start and self.backward_end else 0
        update_time = self.update_end - self.update_start if self.update_start and self.update_end else 0

        log_message = (
            f"Iteration {batch} - Total time: {total_batch_time:.4f}s\n"
            f"  Forward pass time: {forward_time:.4f}s\n"
            f"  Backward pass time: {backward_time:.4f}s\n"
            f"  Weight update time: {update_time:.4f}s\n"
        )

        print(log_message)
        self.log_file.write(log_message)

        # Checkpoint timing
        checkpoint_start = time.time()
        snapshot = self.model.get_weights()  # GPU to CPU snapshot
        gpu_to_cpu_time = time.time() - checkpoint_start
        log_message = f"  Checkpoint (GPU to CPU) time: {gpu_to_cpu_time:.4f}s\n"

        cpu_to_file_start = time.time()
        with open(f'checkpoint_{batch}.ckpt', 'wb') as f:
            np.save(f, snapshot)
        cpu_to_file_time = time.time() - cpu_to_file_start
        log_message += f"  Checkpoint (CPU to File) time: {cpu_to_file_time:.4f}s\n"

        print(log_message)
        self.log_file.write(log_message)
        self.log_file.flush()

    def on_train_end(self, logs=None):
        tf.profiler.experimental.stop()
        self.log_file.close()

# 训练模型并记录时间
timing_callback = TimingCallback()

with tf.profiler.experimental.Trace('train', step_num=0, _r=1):
    model.fit(train_dataset, epochs=1, steps_per_epoch=2, callbacks=[timing_callback])
