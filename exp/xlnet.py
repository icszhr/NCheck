import tensorflow as tf
from transformers import XLNetTokenizer, TFXLNetModel
import time
import random
import string
import os
import numpy as np

# 设置 TensorFlow 日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def generate_random_text(min_length=5, max_length=1000):
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def generate_training_data(tokenizer, min_length=5, max_length=10, batch_size=32):
    texts = [generate_random_text(min_length, max_length) for _ in range(batch_size)]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="tf")
    labels = inputs["input_ids"]
    return inputs, labels

def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn)
    return model, optimizer, loss_fn

def benchmark_xlnet(batch_size=1):
    print("Running benchmark for XLNet...")

    tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased")
    model = TFXLNetModel.from_pretrained("xlnet-large-cased")

    tokenizer.pad_token = tokenizer.eos_token

    model, optimizer, loss_fn = compile_model(model)

    train_inputs, train_labels = generate_training_data(tokenizer, batch_size=batch_size)
    train_inputs = {key: tf.constant(value) for key, value in train_inputs.items()}
    train_labels = tf.constant(train_labels)

    # 前向传播和计算损失
    forward_pass_start_time = time.time()
    with tf.GradientTape() as tape:
        outputs = model(train_inputs, training=True)
        logits = outputs.last_hidden_state
        loss_value = loss_fn(train_labels, logits)
    forward_pass_end_time = time.time()

    # 反向传播
    backward_pass_start_time = time.time()
    gradients = tape.gradient(loss_value, model.trainable_variables)
    backward_pass_end_time = time.time()

    # 权重更新
    weight_update_start_time = time.time()
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    weight_update_end_time = time.time()

    # 模拟 GPU 到 CPU 的快照时间
    snapshot_start_time = time.time()
    model_weights_cpu = [w.numpy() for w in model.trainable_variables]  # 模拟快照到 CPU
    snapshot_end_time = time.time()

    # 模拟将模型从 CPU 持久化到 SSD
    persist_start_time = time.time()
    np.savez("xlnet_model_weights.npz", *model_weights_cpu)  # 将模型权重保存到文件
    persist_end_time = time.time()

    # 计算时间
    forward_pass_time = forward_pass_end_time - forward_pass_start_time
    backward_pass_time = backward_pass_end_time - backward_pass_start_time
    weight_update_time = weight_update_end_time - weight_update_start_time
    snapshot_time = snapshot_end_time - snapshot_start_time
    persist_time = persist_end_time - persist_start_time

    print(f"Time taken for forward pass: {forward_pass_time:.2f} seconds")
    print(f"Time taken for backward pass: {backward_pass_time:.2f} seconds")
    print(f"Time taken for weight update: {weight_update_time:.2f} seconds")
    print(f"Time taken for snapshot: {snapshot_time:.2f} seconds")
    print(f"Time taken for persist: {persist_time:.2f} seconds")

benchmark_xlnet()
