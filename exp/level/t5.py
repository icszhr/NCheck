import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import time
import random
import string
import os

# 设置 TensorFlow 日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

def benchmark_t5(batch_size=1, iterations_per_epoch=1):
    print("Running benchmark for T5...")
    iteration_times = []

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = TFT5ForConditionalGeneration.from_pretrained("t5-small")

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
            decoder_input_ids = tf.roll(train_labels, shift=1, axis=1)
            decoder_input_ids = tf.where(decoder_input_ids == 0, tf.fill(tf.shape(decoder_input_ids), -100), decoder_input_ids)
            outputs = model(train_inputs, decoder_input_ids=decoder_input_ids, training=True)
            logits = outputs.logits
            loss_value = loss_fn(train_labels[:, 1:], logits[:, :-1])
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
    print(f"Total training time for 1 epoch with T5 (excluding first iteration):", sum(iteration_times))

    return sum(iteration_times)

# 运行基准测试并保存结果
total_time = benchmark_t5()
with open("T5_performance.txt", "w") as f:
    f.write(f"Total Time (excluding first iteration): {total_time:.2f} seconds\n")

print("Benchmark results saved to T5_performance.txt")
