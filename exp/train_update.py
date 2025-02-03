import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import time
import random
import string
import os

# 设置 TensorFlow 日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# 加载 GPT-2 模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # 添加padding token
model = TFGPT2LMHeadModel.from_pretrained("gpt2")

# 定义生成随机文本的函数
def generate_random_text(min_length=5, max_length=1000):
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(string.ascii_lowercase, k=length))

# 生成训练数据
def generate_training_data(min_length=5, max_length=10, batch_size=32):
    texts = [generate_random_text(min_length, max_length) for _ in range(batch_size)]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="tf")
    labels = inputs["input_ids"]
    return inputs, labels

# 生成大量训练数据
batch_size = 100
train_inputs, train_labels = generate_training_data(batch_size=batch_size)

# 转换为 TensorFlow 张量
train_inputs = {key: tf.constant(value) for key, value in train_inputs.items()}
train_labels = tf.constant(train_labels)

# 定义优化器和损失函数
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 自定义训练步骤
def train_step(model, inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions.logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    update_start_time = time.time()
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    update_time = time.time() - update_start_time
    return loss, update_time

# 训练模型
for epoch in range(10):
    print(f"Epoch {epoch + 1}/{10}")
    loss, update_time = train_step(model, train_inputs, train_labels)
    print(f"Loss: {loss.numpy():.4f}, Update Time: {update_time:.4f} seconds")
