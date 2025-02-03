import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 设置 TensorFlow 日志级别
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import random
import string

# 设置 TensorFlow 日志级别
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




model.fit(train_dataset, epochs=1, steps_per_epoch=1)