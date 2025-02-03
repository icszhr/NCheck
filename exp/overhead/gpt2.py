import tensorflow as tf
from transformers import XLNetTokenizer, TFXLNetLMHeadModel
import time
import random
import string
import os
import numpy as np

# 设置 TensorFlow 日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.get_logger().setLevel('ERROR')

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
    return model, optimizer, loss_fn

def benchmark_xlnet(batch_size=1, iterations_per_epoch=10):
    print("Running benchmark for XLNet...")
    iteration_times = []

    # 使用本地路径加载模型和分词器
    local_model_path = "xlnet-base-cased"
    try:
        tokenizer = XLNetTokenizer.from_pretrained(local_model_path)
        model = TFXLNetLMHeadModel.from_pretrained(local_model_path)
    except ValueError as e:
        print(f"Error loading model: {e}")
        return

    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, optimizer, loss_fn = compile_model(model)

    start_time = time.time()
    print(f"Starting training for 1 epoch with {iterations_per_epoch} iterations")

    for iteration in range(iterations_per_epoch):
        # 生成一个批次的训练数据
        train_inputs, train_labels = generate_training_data(tokenizer, batch_size=batch_size)
        train_inputs = {key: tf.constant(value) for key, value in train_inputs.items()}
        train_labels = tf.constant(train_labels)

        if iteration == 0:
            continue  # 跳过第一个 iteration

        # 记录整个 iteration 的开始时间
        iteration_start_time = time.time()

        # 前向传播和计算损失
        forward_pass_start_time = time.time()
        with tf.GradientTape() as tape:
            outputs = model(train_inputs, training=True)
            logits = outputs.logits
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

        iteration_end_time = time.time()

        # 计算时间
        forward_pass_time = forward_pass_end_time - forward_pass_start_time
        backward_pass_time = backward_pass_end_time - backward_pass_start_time
        weight_update_time = weight_update_end_time - weight_update_start_time
        iteration_time = iteration_end_time - iteration_start_time

        print(f"Iteration {iteration + 1} - Time taken for forward pass: {forward_pass_time:.2f} seconds")
        print(f"Iteration {iteration + 1} - Time taken for backward pass: {backward_pass_time:.2f} seconds")
        print(f"Iteration {iteration + 1} - Time taken for weight update: {weight_update_time:.2f} seconds")
        print(f"Iteration {iteration + 1} - Total time taken for iteration: {iteration_time:.2f} seconds")

        iteration_times.append(iteration_time)

    end_time = time.time()

    # 计算总的训练时间
    training_time = end_time - start_time
    print(f"Total training time for 1 epoch with XLNet (excluding first iteration):", sum(iteration_times))

    return sum(iteration_times)

# 运行基准测试并保存结果
total_time = benchmark_xlnet()
if total_time is not None:
    with open("XLNet_performance.txt", "w") as f:
        f.write(f"Total Time (excluding first iteration): {total_time:.2f} seconds\n")
    print("Benchmark results saved to XLNet_performance.txt")
else:
    print("Benchmark failed due to model loading error.")
